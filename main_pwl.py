#!/usr/bin/env python3


import numpy as np
import tqdm
from simple_einet.dist import DataType, Domain
from simple_einet.layers.distributions.piecewise_linear import PiecewiseLinear
from torch.distributions import Binomial
import matplotlib.pyplot as plt

import torch

from simple_einet.einet import Einet, EinetConfig

BINS = 100

def make_dataset(num_features_continuous, num_features_discrete, num_clusters, num_samples):
    # Collect data and data domains
    data = []
    domains = []

    # Construct continuous features
    for i in range(num_features_continuous):
        domains.append(Domain.continuous_inf_support())
        feat_i = []

        # Create a multimodal feature
        for j in range(num_clusters):
            feat_i.append(torch.randn(num_samples) + j * 3 * torch.rand(1) + 3 * j)

        data.append(torch.cat(feat_i))

    # Construct discrete features
    for i in range(num_features_discrete):
        domains.append(Domain.discrete_range(0, BINS))
        feat_i = []

        # Create a multimodal feature
        for j in range(num_clusters):
            feat_i.append(Binomial(total_count=BINS, probs=torch.rand(1)).sample((num_samples,)).view(-1))
        data.append(torch.cat(feat_i))

    data = torch.stack(data, dim=1)
    data = data.view(data.shape[0], 1, num_features_continuous + num_features_discrete)
    data = data[torch.randperm(data.shape[0])]
    return data, domains


if __name__ == "__main__":
    torch.manual_seed(0)

    ###################
    # Hyperparameters #
    ###################

    epochs = 3
    batch_size = 128
    depth = 2
    num_sums = 20
    num_leaves = 10
    num_repetitions = 10
    lr = 0.01

    num_features = 4

    ###############
    # Einet Setup #
    ###############

    config = EinetConfig(
        num_features=num_features,
        num_channels=1,
        depth=depth,
        num_sums=num_sums,
        num_leaves=num_leaves,
        num_repetitions=num_repetitions,
        num_classes=1,
        leaf_type=PiecewiseLinear,
        # leaf_kwargs={"alpha": 0.05},
        layer_type="linsum",
        dropout=0.0,
    )

    model = Einet(config)
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))



    ##############
    # Data Setup #
    ##############

    # Simulate data
    data, domains = make_dataset(
        num_features_continuous=num_features // 2,
        num_features_discrete=num_features // 2,
        num_clusters=4,
        num_samples=1000,
    )

    ########################################
    # PiecewiseLinear Layer Initialization #
    ########################################

    model.leaf.base_leaf.initialize(data, domains=domains)

    # Init. first linsum layer weights to be the log of the mixture weights from the kmeans result in the PWL init phase
    model.layers[0].logits.data[:] = (
        model.leaf.base_leaf.mixture_weights.permute(1, 0).view(1, config.num_leaves, 1, config.num_repetitions).log()
    )


    ################
    # Optimization #
    ################
    # Optimize Einet parameters (weights and leaf params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1, verbose=True)

    model.train()
    for epoch in range(1, epochs + 1):
        # Since we don't have a train dataloader, we will loop over the data manually
        iter = range(0, len(data), batch_size)
        pbar = tqdm.tqdm(iter, desc="Train Epoch: {}".format(epoch))
        for batch_idx in pbar:
            optimizer.zero_grad()

            # Select batch
            data_batch = data[batch_idx : batch_idx + batch_size]

            # Generate outputs
            outputs = model(data_batch, cache_index=batch_idx)

            # Compute loss
            loss = -1 * outputs.mean()

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            # Logging
            if batch_idx % 10 == 0:
                pbar.set_description(
                    "Train Epoch: {} [{}/{}] Loss: {:.2f}".format(
                        epoch,
                        batch_idx,
                        len(data),
                        loss.item(),
                    )
                )
        scheduler.step()


    model.eval()

    #################
    # Visualization #
    #################

    # Generate samples
    samples = model.sample(10000)

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        # Get data subset
        if domains[i].data_type == DataType.DISCRETE:
            rng = (0, BINS + 1)
            bins = BINS + 1
            width = 1
        else:
            rng = None
            bins = 100
            width = (samples[:, :, i].max() - samples[:, :, i].min()) / bins

        # Plot histogram of data
        hist = torch.histogram(samples[:, :, i], bins=bins, density=True, range=rng)
        bin_edges = hist.bin_edges
        density = hist.hist
        if domains[i].data_type == DataType.DISCRETE:
            bin_edges -= 0.5

        # Center bars on value (e.g. bar for value 0 should have its center at value 0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, density, width=width * 0.8, alpha=0.5, label="Samples")


        if domains[i].data_type == DataType.DISCRETE:
            rng = (0, BINS + 1)
            bins = BINS + 1
            width = 1
        else:
            rng = None
            bins = 100
            width = (data[:, :, i].max() - data[:, :, i].min()) / bins

        # Plot histogram of data
        hist = torch.histogram(data[:, :, i], bins=bins, density=True, range=rng)
        bin_edges = hist.bin_edges
        density = hist.hist
        if domains[i].data_type == DataType.DISCRETE:
            bin_edges -= 0.5

        # Center bars on value (e.g. bar for value 0 should have its center at value 0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, density, width=width * 0.8, alpha=0.5, label="Data")

        # Plot PWL logprobs
        dummy = torch.full((bin_centers.shape[0], data.shape[1], data.shape[2]), np.nan)
        dummy[:, 0, i] = bin_centers
        with torch.no_grad():
            log_probs = model(dummy)
        probs = log_probs.exp().squeeze(-1).numpy()
        ax.plot(bin_centers, probs, linewidth=2, label="PWL Density")


        # MPE
        mpe = model.mpe()
        dummy = torch.full((mpe.shape[0], data.shape[1], data.shape[2]), np.nan)
        dummy[:, 0, i] = mpe[:, 0, i]
        with torch.no_grad():
            mpe_prob = model(dummy).exp().detach()
        ax.plot(mpe.squeeze()[i], mpe_prob.squeeze(), "rx", markersize=13, label="PWL MPE")

        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")

        ax.set_title(f"Feature {i} ({str(domains[i].data_type)})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"/tmp/pwl.png", dpi=300)

    # Conditional sampling example
    data_subset = data[:10]
    data[:, :, :2] = np.nan
    samples_cond = model.sample(evidence=data_subset)

    # Conditional MPE example
    mpe_cond = model.mpe(evidence=data_subset)
