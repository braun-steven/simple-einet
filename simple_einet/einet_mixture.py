from _operator import xor
from collections import defaultdict
from typing import Sequence, List

import torch
from fast_pytorch_kmeans import KMeans
from torch import nn
from torch.utils.data import DataLoader

from simple_einet.einet import EinetConfig, Einet
from simple_einet.type_checks import check_valid


class EinetMixture(nn.Module):
    def __init__(self, n_components: int, einet_config: EinetConfig):
        super().__init__()
        self.n_components = check_valid(n_components, expected_type=int, lower_bound=1)
        self.config = einet_config

        einets = []

        for i in range(n_components):
            einets.append(Einet(einet_config))

        self.einets: Sequence[Einet] = nn.ModuleList(einets)
        self._kmeans = KMeans(n_clusters=self.n_components, mode="euclidean", verbose=1)
        self.mixture_weights = nn.Parameter(torch.empty(n_components), requires_grad=False)
        self.centroids = nn.Parameter(torch.empty(n_components, einet_config.num_features), requires_grad=False)

    @torch.no_grad()
    def initialize(self, data: torch.Tensor = None, dataloader: DataLoader = None, device=None):
        assert xor(data is not None, dataloader is not None)

        if dataloader is not None:
            # Collect data from dataloader
            l = []
            for batch in dataloader:
                x, y = batch
                l.append(x)
                if sum([d.shape[0] for d in l]) > 10000:
                    break

            data = torch.cat(l, dim=0).to(device)

        data = data.float()  # input has to be [n, d]
        predictions = self._kmeans.fit_predict(data.view(data.shape[0], -1))
        counts = torch.bincount(predictions)
        self.mixture_weights.data = counts / counts.sum()

        self.centroids.data = self._kmeans.centroids

    def _predict_cluster(self, x, marginalized_scopes: List[int] = None):
        x = x.view(x.shape[0], -1)  # input needs to be [n, d]
        if marginalized_scopes is not None:
            keep_idx = list(sorted([i for i in range(self.config.num_features) if i not in marginalized_scopes]))
            centroids = self.centroids[:, keep_idx]
            x = x[:, keep_idx]
        else:
            centroids = self.centroids
        return self._kmeans.max_sim(a=x.float(), b=centroids)[1]

    def _separate_data_by_cluster(self, x: torch.Tensor, marginalized_scope: List[int]):
        cluster_idxs = self._predict_cluster(x, marginalized_scope).tolist()

        separated_data = defaultdict(list)
        separated_idxs = defaultdict(list)
        for data_idx, cluster_idx in enumerate(cluster_idxs):
            separated_data[cluster_idx].append(x[data_idx])
            separated_idxs[cluster_idx].append(data_idx)

        return separated_idxs, separated_data

    def forward(self, x, marginalized_scope: torch.Tensor = None):
        assert self._kmeans is not None, "EinetMixture has not been initialized yet."

        separated_idxs, separated_data = self._separate_data_by_cluster(x, marginalized_scope)

        lls_result = []
        data_idxs_all = []
        for cluster_idx, data_list in separated_data.items():
            data_tensor = torch.stack(data_list, dim=0)
            lls = self.einets[cluster_idx](data_tensor)

            data_idxs = separated_idxs[cluster_idx]
            for data_idx, ll in zip(data_idxs, lls):
                lls_result.append(ll)
                data_idxs_all.append(data_idx)

        # Sort results into original order as observed in the batch
        L = [(data_idxs_all[i], i) for i in range(len(data_idxs_all))]
        L.sort()
        _, permutation = zip(*L)
        permutation = torch.tensor(permutation, device=x.device).view(-1)
        lls_result = torch.stack(lls_result)
        lls_sorted = lls_result[permutation]

        return lls_sorted

    def sample(
        self,
        num_samples: int = None,
        num_samples_per_cluster: int = None,
        class_index=None,
        evidence: torch.Tensor = None,
        is_mpe: bool = False,
        temperature_leaves: float = 1.0,
        temperature_sums: float = 1.0,
        marginalized_scopes: List[int] = None,
        seed=None,
        mpe_at_leaves: bool = False,
    ):
        assert num_samples is None or num_samples_per_cluster is None
        if num_samples is None and num_samples_per_cluster is not None:
            num_samples = num_samples_per_cluster * self.n_components

        # Check if evidence contains nans
        if evidence is not None:
            # Set n to the number of samples in the evidence
            num_samples = evidence.shape[0]
        elif num_samples is None:
            num_samples = 1

        if is_mpe:
            # Take cluster idx with largest weights
            cluster_idxs = [self.mixture_weights.argmax().item()]
        else:
            if num_samples_per_cluster is not None:
                cluster_idxs = torch.arange(self.n_components).repeat_interleave(num_samples_per_cluster).tolist()
            else:
                # Sample from categorical over weights
                cluster_idxs = (
                    torch.distributions.Categorical(probs=self.mixture_weights).sample((num_samples,)).tolist()
                )

        if evidence is None:
            # Sample without evidence
            separated_idxs = defaultdict(int)
            for cluster_idx in cluster_idxs:
                separated_idxs[cluster_idx] += 1

            samples_all = []
            for cluster_idx, num_samples_cluster in separated_idxs.items():
                samples = self.einets[cluster_idx].sample(
                    num_samples_cluster,
                    class_index=class_index,
                    evidence=evidence,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    marginalized_scopes=marginalized_scopes,
                    seed=seed,
                    mpe_at_leaves=mpe_at_leaves,
                )
                samples_all.append(samples)

            samples = torch.cat(samples_all, dim=0)

            # Sort results into original order as observed in the batch
            L = [(cluster_idxs[i], i) for i in range(len(cluster_idxs))]
            L.sort()
            _, permutation = zip(*L)
            permutation = torch.tensor(permutation, device=samples.device).view(-1)
            samples_all = torch.cat(samples_all)
            samples_sorted = samples_all[permutation]
            samples = samples_sorted

        else:
            # Sample with evidence
            separated_idxs, separated_data = self._separate_data_by_cluster(evidence, marginalized_scopes)

            samples_all = []
            evidence_idxs_all = []
            for cluster_idx, evidence_pre_cluster in separated_data.items():
                evidence_per_cluster = torch.stack(evidence_pre_cluster, dim=0)
                samples = self.einets[cluster_idx].sample(
                    evidence=evidence_per_cluster,
                    is_mpe=is_mpe,
                    temperature_leaves=temperature_leaves,
                    temperature_sums=temperature_sums,
                    marginalized_scopes=marginalized_scopes,
                )

                evidence_idxs = separated_idxs[cluster_idx]
                for evidence_idx, sample in zip(evidence_idxs, samples):
                    samples_all.append(sample)
                    evidence_idxs_all.append(evidence_idx)

            # Sort results into original order as observed in the batch
            L = [(evidence_idxs_all[i], i) for i in range(len(evidence_idxs_all))]
            L.sort()
            _, permutation = zip(*L)
            permutation = torch.tensor(permutation, device=evidence.device).view(-1)
            samples_all = torch.stack(samples_all)
            samples_sorted = samples_all[permutation]
            samples = samples_sorted

        return samples

    def mpe(
        self,
        evidence: torch.Tensor = None,
        marginalized_scopes: List[int] = None,
    ) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(evidence=evidence, is_mpe=True, marginalized_scopes=marginalized_scopes)
