import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training",
    )
    parser.add_argument(
        "--mixture",
        default=1,
        type=int,
        help="Number of mixture components for an EinetMixture model (if 1 then only an Einet model is used).",
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
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
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
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
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
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
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
        "--dataset",
        default="mnist",
        help="Dataset",
    )
    parser.add_argument("-I", type=int, default=10)
    parser.add_argument("-S", type=int, default=10)
    parser.add_argument("-D", type=int, default=3)
    parser.add_argument("-R", type=int, default=1)
    return parser.parse_args()
