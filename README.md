# Simple EinsumNetworks Implementation

This repository contains code for my personal, simplistic, EinsumNetworks implementation.

## Usage Example

``` python
from distributions import Normal
torch.manual_seed(0)

# Input dimensions
in_features = 4
batchsize = 5

# Create input sample
x = torch.randn(batchsize, in_features)

# Construct Einet
einet = Einet(K=2, D=2, R=2, in_features=in_features, leaf_cls=Normal)

# Compute log-likelihoods
lls = einet(x)
print(f"lls={lls}")
print(f"lss.shape={lls.shape}")

# Construct samples
samples = einet.sample(2)
print(f"samples={samples}")
print(f"samples.shape={samples.shape}")

# Optimize Einet parameters (weights and leaf params)
optim = torch.optim.Adam(einet.parameters(), lr=0.001)
clipper = DistributionClipper()

for _ in range(1000):
    optim.zero_grad()

    # Forward pass: log-likelihoods
    lls = einet(x)

    # Backprop NLL loss
    nlls = -1 * lls.sum()
    nlls.backward()

    # Update weights
    optim.step()

    # Clip leaf distribution parameters (e.g. std > 0.0, etc.)
    clipper(einet.leaf)
```


## Citing EinsumNetworks

``` bibtex
@inproceedings{pmlr-v119-peharz20a,
  title = {Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits},
  author = {Peharz, Robert and Lang, Steven and Vergari, Antonio and Stelzner, Karl and Molina, Alejandro and Trapp, Martin and Van Den Broeck, Guy and Kersting, Kristian and Ghahramani, Zoubin},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  pages = {7563--7574},
  year = {2020},
  editor = {III, Hal Daumé and Singh, Aarti},
  volume = {119},
  series = {Proceedings of Machine Learning Research},
  month = {13--18 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v119/peharz20a/peharz20a.pdf},
  url = {http://proceedings.mlr.press/v119/peharz20a.html},
  code = {https://github.com/cambridge-mlg/EinsumNetworks},
}
```

