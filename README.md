# Simple EinsumNetworks Implementation

This repository contains code for my personal, simplistic, EinsumNetworks implementation.

## Installation

```shell
git clone git@github.com:steven-lang/simple-einet.git
cd simple-einet
pip install .

# Or if you plan to edit the files after installation:
pip install -e .
```

## Usage Example

```python

import torch
from simple_einet.clipper import DistributionClipper
from simple_einet.distributions import RatNormal
from simple_einet.einet import Einet
from simple_einet.einet import EinetConfig

torch.manual_seed(0)

# Input dimensions
in_features = 4
batchsize = 5
out_features = 3

# Create input sample
x = torch.randn(batchsize, in_features)

# Construct Einet
einet = Einet(EinetConfig(in_features=in_features, D=2, S=2, I=2, R=3, C=out_features, dropout=0.0, leaf_base_class=RatNormal, leaf_base_kwargs={"min_sigma": 1e-5, "max_sigma": 1.0},))

# Compute log-likelihoods
lls = einet(x)
print(f"lls={lls}")
print(f"lls.shape={lls.shape}")

# Optimize Einet parameters (weights and leaf params)
optim = torch.optim.Adam(einet.parameters(), lr=0.001)

for _ in range(1000):
    optim.zero_grad()

    # Forward pass: compute log-likelihoods
    lls = einet(x)

    # Backprop negative log-likelihood loss
    nlls = -1 * lls.sum()
    nlls.backward()

    # Update weights
    optim.step()

# Construct samples
samples = einet.sample(2)
print(f"samples={samples}")
print(f"samples.shape={samples.shape}")
```

## MNIST Samples
Some samples from the `[0, 1]` class-subset of MNIST [./mnist.py]:

**Samples**

![MNIST Samples]( ./res/mnist-0-1-samples.png )

**Reconstructions (conditioned on bottom half)**

![MNIST Reconstructions]( ./res/mnist-0-1-rec.png )

## Citing EinsumNetworks

```bibtex
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
