import torch
from simple_einet.clipper import DistributionClipper
from simple_einet.distributions import Normal
from simple_einet.einet import Einet
from simple_einet.einet import EinetConfig
from torchviz import make_dot

torch.manual_seed(0)

# Input dimensions
in_features = 4
batchsize = 5
out_features = 3

# Create input sample
x = torch.randn(batchsize, in_features)

# Construct Einet
einet = Einet(EinetConfig(
    in_features=in_features,
    D=2,
    S=2,
    I=2,
    R=3,
    C=out_features,
    dropout=0.5,
    leaf_base_class=Normal,
))

# Compute log-likelihoods
lls = einet(x)
print(f"lls={lls}")
print(f"lls.shape={lls.shape}")

# dot = make_dot(lls, params=dict(einet.named_parameters()))
# dot.view(filename="einet_model_structure.pdf")

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
    clipper(einet._leaf)
