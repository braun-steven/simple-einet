#!/usr/bin/env python3

import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch

from simple_einet.distributions import RatNormal
from simple_einet.einet import Einet, EinetConfig


parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=14,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--device",
    default="cuda",
    help="Device flag. Can be either 'cpu' or 'cuda'.",
)
parser.add_argument("-K", type=int, default=3)
parser.add_argument("-D", type=int, default=1)
parser.add_argument("-R", type=int, default=3)
args = parser.parse_args()

device = torch.device(args.device)
torch.manual_seed(args.seed)

config = EinetConfig(in_features=4, D=args.D, S=args.K, I=args.K, R=args.R, C=3, leaf_base_class=RatNormal, leaf_base_kwargs={}, dropout=0.0)
model = Einet(config).to(device)
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=args.seed)
X_train = torch.tensor(X_train).float().to(device)
y_train = torch.tensor(y_train).long().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_test = torch.tensor(y_test).long().to(device)

def accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(-1)
        correct = (predictions == y).sum()
        total = y.shape[0]
        acc = correct / total * 100
        return acc

cross_entropy = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = cross_entropy(outputs, y_train)

    loss.backward()
    optimizer.step()

    acc_train = accuracy(model, X_train, y_train)
    acc_test = accuracy(model, X_test, y_test)
    print(
        "Train Epoch: {}\tLoss: {:3.2f}\t\tAccuracy Train: {:2.2f} %\t\tAccuracy Test: {:2.2f} %".format(
            epoch,
            loss.item(),
            acc_train,
            acc_test,
        )
    )
