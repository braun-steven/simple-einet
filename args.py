import argparse
import os
import pathlib

from simple_einet.data import Dist


def parse_args():

    home = os.getenv("HOME")
    data_dir = os.getenv("DATA_DIR", os.path.join(home, "data"))
    results_dir = os.getenv("RESULTS_DIR", os.path.join(home, "results"))
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset to use for training.",
    )
    parser.add_argument("--data-dir", default=data_dir, help="path to dataset")
    parser.add_argument("--results-dir", default=results_dir, help="path to results")
    parser.add_argument(
        "--mixture",
        default=1,
        type=int,
        help="Number of mixture components for an EinetMixture model (if 1 then only an Einet model is used).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=8,
        metavar="N",
        help="number of bits for each pixel (default: 8)",
    )

    parser.add_argument(
        "--num-workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--temperature-leaves",
        type=float,
        default=1.0,
        help="Temperature for leave variance during sampling.",
    )
    parser.add_argument(
        "--temperature-sums",
        type=float,
        default=1.0,
        help="Temperature for sum weights during sampling.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=1e-2,
        help="Normal distribution min sigma value.",
    )
    parser.add_argument(
        "--max-sigma",
        type=float,
        default=2.0,
        help="Normal distribution min sigma value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--classification",
        action="store_true",
        default=False,
        help="Flag for learning a discriminative task of classifying MNIST digits.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device flag. Can be either 'cpu' or 'cuda'.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug flag (less data, fewer iterations)",
    )
    parser.add_argument(
        "-S", type=int, default=10, help="Number of output sum nodes in each layer."
    )
    parser.add_argument(
        "-I", type=int, default=10, help="Number of distributions for each RV."
    )
    parser.add_argument("-D", type=int, default=3)
    parser.add_argument("-R", type=int, default=1)
    parser.add_argument("--gpu", help="GPU device id.")

    parser.add_argument(
        "--load-and-eval",
        default=None,
        type=pathlib.Path,
        help="path to a result directory with a "
        "model and stored args. if set, "
        "training is skipped and model is "
        "evaluated",
    )
    parser.add_argument(
        "--cp", action="store_true", help="Use crossproduct in einsum layer"
    )
    parser.add_argument(
        "--dist",
        type=Dist,
        choices=list(Dist),
        default=Dist.BINOMIAL,
        help="data distribution",
    )
    parser.add_argument(
        "--precision",
        "-p",
        default=32,
        help="floating point precision [16, " "bf16, 32]",
        choices=["16", "bf16", "32"],
    )
    parser.add_argument("--group-tag", type=str, help="tag for group of experiments")
    parser.add_argument("--tag", type=str, help="tag for experiment")
    parser.add_argument(
        "--wandb", action="store_true", help="enable wandb online logging"
    )
    parser.add_argument("--swa", action="store_true", help="use Stochastic Weight Averaging")

    parser.add_argument("--profiler", help="", choices=["simple", "pytorch", "advanced"])

    parser.add_argument("--log-weights", action="store_true", help="use log weights")

    # Parse args
    args = parser.parse_args()

    # If FP16/FP32 is given, convert to int (else it's "bf16", keep string)
    if args.precision == "16" or args.precision == "32":
        args.precision = int(args.precision)
    return args