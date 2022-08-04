import argparse
from omegaconf import DictConfig
import io
import json
import pathlib
import pprint

import PIL
import matplotlib
import matplotlib.pyplot as plt
import torchvision.utils
from pytorch_lightning.loggers import WandbLogger

from rtpt import RTPT

import contextlib
from torch import nn
import shutil
import traceback
import datetime
import errno
import logging
import os
import random
import time
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.backends import cudnn as cudnn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from simple_einet.data import build_dataloader, get_data_shape, Shape, generate_data


def save_args(results_dir: str, args: argparse.Namespace):
    """
    Save the arguments to a file.
    Args:
        results_dir: The directory to save the arguments to.
        args: The arguments to save.
    """
    with open(os.path.join(results_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def make_results_dir(
    results_dir: str,
    experiment_name: str,
    tag: str,
    dataset_name: str,
    debug: bool = False,
    remove_if_exists=True,
):
    """
    Returns the path to the results directory for the given experiment and dataset.

    Args:
        results_dir: Base results dir.
        experiment_name: Name of the experiment.
        tag: Tag for the experiment.
        dataset_name: Name of the dataset.
        debug: Boolean indicating whether to use the debug directory.
        remove_if_exists: If True, the results directory will be removed if it already exists.

    Returns:
        Path to the results directory.
    """
    if tag is None:
        dirname = "unnamed"
    else:
        dirname = tag

    if debug:
        dirname += "_debug"

    # Create directory
    experiment_dir = os.path.join(
        results_dir,
        "simple-einet",
        experiment_name,
        dataset_name,
        dirname,
    )

    if remove_if_exists:
        if os.path.exists(experiment_dir):
            print("Directory already exists, adding _2 to the name")
            experiment_dir += "_2"
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir + "/"


def get_data_dir(dataset_name: str):
    """
    Returns the path to the data directory for the given dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Path to the data directory.
    """
    # Get home directory
    home_dir = os.getenv("HOME")

    # Merge home directory with results, project, dataset and directories
    data_dir = os.path.join(
        home_dir, "data", "simple-einet-diff-sampling", dataset_name
    )

    # Create directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_project_name() -> str:
    return open("./PROJECT_NAME").readlines()[0].strip()


logger = logging.getLogger(get_project_name())


def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def ensure_dir(path: str):
    """
    Ensure that a directory exists.

    For 'foo/bar/baz.csv' the directories 'foo' and 'bar' will be created if not already present.

    Args:
        path (str): Directory path.
    """

    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_run_base_dir(
    result_dir: str,
    timestamp: int = None,
    tag: str = None,
    sub_dirs: List[str] = None,
    debug: bool = False,
) -> str:
    """
    Generate a base directory for each experiment run.
    Looks like this: result_dir/date_tag/sub_dir_1/.../sub_dir_n
    Args:
        result_dir (str): Experiment output directory.
        timestamp (int): Timestamp which will be inlcuded in the form of '%y%m%d_%H%M'.
        tag (str): Tag after timestamp.
        sub_dirs (List[str]): List of subdirectories that should be created.
        deubg (bool): Append '_debug' to the end.

    Returns:
        str: Directory name.
    """
    if timestamp is None:
        timestamp = time.time()

    if sub_dirs is None:
        sub_dirs = []

    # Convert time
    date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%y%m%d_%H%M")

    # Append tag if given
    if tag is None:
        base_dir = date_str
    else:
        base_dir = date_str + "_" + tag

    if debug:
        base_dir = base_dir + "_debug"

    # Create directory
    base_dir = os.path.join(result_dir, base_dir, *sub_dirs)

    mkdir_p(base_dir)
    return base_dir + "/"


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_reset_seed(worker_id: int):
    """Initialize the worker by settign a seed depending on the worker id.

    Args:
        worker_id (int): Unique worker id.
    """
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)


def num_gpus() -> int:
    """Get the number of GPUs from the 'NVIDIA_VISIBLE_DEVICES' environment variable.

    Returns:
        int: Number of GPUs.
    """
    gpus = list(range(len(os.environ["NVIDIA_VISIBLE_DEVICES"].split(","))))
    return len(gpus)


def auto_scale_workers(cfg, num_workers: int):
    """
    NOTE: Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
    When the config is defined for certain number of workers (according to
    ``cfg.solver.reference_world_size``) that's different from the number of
    workers currently in use, returns a new cfg where the total batch size
    is scaled so that the per-GPU batch size stays the same as the
    original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.
    Other config options are also scaled accordingly:
    * training steps and warmup steps are scaled inverse proportionally.
    * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.
    For example, with the original config like the following:
    .. code-block:: yaml
        IMS_PER_BATCH: 16
        BASE_LR: 0.1
        REFERENCE_WORLD_SIZE: 8
        MAX_ITER: 5000
        STEPS: (4000,)
        CHECKPOINT_PERIOD: 1000
    When this config is used on 16 GPUs instead of the reference number 8,
    calling this method will return a new config with:
    .. code-block:: yaml
        IMS_PER_BATCH: 32
        BASE_LR: 0.2
        REFERENCE_WORLD_SIZE: 16
        MAX_ITER: 2500
        STEPS: (2000,)
        CHECKPOINT_PERIOD: 500
    Note that both the original config and this new config can be trained on 16 GPUs.
    It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).
    Returns:
        CfgNode: a new config. Same as original if ``cfg.solver.reference_world_size==0``.
    """
    old_world_size = cfg.system.reference_world_size
    if old_world_size == 0 or old_world_size == num_workers:
        return cfg
    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()

    assert (
        cfg.train.batch_size % old_world_size == 0
    ), "Invalid REFERENCE_WORLD_SIZE in config!"
    scale = num_workers / old_world_size
    bs = cfg.train.batch_size = int(round(cfg.train.batch_size * scale))
    lr = cfg.solver.LR = cfg.solver.LR * scale
    max_iter = cfg.solver.max_iter = int(round(cfg.solver.max_iter / scale))
    warmup_iter = cfg.solver.scheduler.warmuplr.iters = int(
        round(cfg.solver.scheduler.warmuplr.iters / scale)
    )
    # cfg.solver.steps = tuple(int(round(s / scale)) for s in cfg.solver.steps)
    cfg.test.eval_period = int(round(cfg.test.eval_period / scale))
    cfg.checkpoints.period = int(round(cfg.checkpoints.period / scale))
    cfg.system.reference_world_size = num_workers  # maintain invariant
    logger.info(
        f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
        f"max_iter={max_iter}, warmup={warmup_iter}."
    )

    if frozen:
        cfg.freeze()
    return cfg


def detect_anomaly(losses: torch.Tensor, iteration: int):
    """Check if loss is finite .

    Args:
        losses (torch.Tensor): Loss to be checked.
        iteration (int): Current iteration.

    Raises:
        FloatingPointError: Loss was not finite.
    """
    # use a new stream so the ops don't wait for DDP
    with torch.cuda.stream(
        torch.cuda.Stream(device=losses.device)
    ) if losses.device.type == "cuda" else contextlib.nullcontext():
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!".format(iteration)
            )


def catch_exception(output_directory: str, e: Exception):
    """Catch exception and rename output directory.

    Args:
        save_path (str): Model output directory.
        e (Exception): Exception which was catched.

    Raises:
        Exception: Exception which was catched.
    """
    # Log error message
    tbstr = "".join(traceback.extract_tb(e.__traceback__).format())
    errormsg = f"Traceback:\n{tbstr}\nError: {e}"

    # Rename output dir
    src = output_directory
    if src.endswith("/"):
        src = src[:-1]
    dst = src + "_error"

    # Write error to separate file
    with open(os.path.join(output_directory, "error.txt"), "w") as f:
        f.write(errormsg)

    logger.error("Error caught!")
    logger.error(f"Moving output directory from")
    logger.error(src)
    logger.error("to")
    logger.error(dst)

    shutil.move(src, dst)
    raise e


def catch_kb_interrupt(output_directory):
    """Catch keyboard interrupt and rename output directory.

    Args:
        output_directory (str): Output directory.
    """
    # Rename output dir
    src = output_directory
    if src.endswith("/"):
        src = src[:-1]
    dst = src + "_interrupted"

    logger.error(f"Keyboard interruption catched.")
    logger.error(f"Moving output directory from")
    logger.error(src)
    logger.error("to")
    logger.error(dst)

    shutil.move(src, dst)


@torch.no_grad()
def print_num_params(model: nn.Module):
    """
    Compute the number of parameters and separate into Flow/SPN parts.

    Args:
      model (nn.Module): Model with parameters.

    """
    if type(model) == DistributedDataParallel:
        model = model.module

    # Count all parameteres
    sum_params = count_params(model)

    # Count SPN parameters
    spn_params = sum_params

    # Print
    logger.info(f"Number of parameters:")
    # logger.info(f"- Total:  {sum_params / 1e6: >8.3f}M")
    logger.info(
        f"-   SPN:  {spn_params / 1e6: >8.3f}M ({spn_params / sum_params * 100:.1f}%)"
    )
    # logger.info(f"-    NN:  {nn_params / 1e6: >8.3f}M ({nn_params / sum_params * 100:.1f}%)")


def preprocess(
    x: torch.Tensor,
    n_bits: int,
) -> torch.Tensor:
    x = reduce_bits(x, n_bits)
    # x = x.long()
    return x


def reduce_bits(image: torch.Tensor, n_bits: int) -> torch.Tensor:
    image = image * 255
    if n_bits < 8:
        image = torch.floor(image / 2 ** (8 - n_bits))

    return image


def xor(a: bool, b: bool) -> bool:
    """Perform the XOR operation between a and b."""
    return (a and not b) or (not a and b)


def loss_dict_to_str(running_loss_dict: Dict[str, float], logging_period: int) -> str:
    """Create a joined string from a dictionary mapping str->float."""
    loss_str = ", ".join(
        [
            f"{key}: {value / logging_period:.2f}"
            for key, value in running_loss_dict.items()
        ]
    )
    return loss_str


def plot_tensor(x: torch.Tensor):

    plt.figure()
    if x.dim() == 4:
        x = torchvision.utils.make_grid(x)
    plt.imshow(x.permute(1, 2, 0).detach().cpu().numpy())
    plt.show()
    plt.close()


def build_tensorboard_writer(results_dir):
    """
    Build a tensorboard writer.
    Args:
        results_dir: Directory where to save the tensorboard files.

    Returns:
        A tensorboard writer.
    """
    return SummaryWriter(os.path.join(results_dir, "tensorboard"))


def setup_experiment(
    name: str,
    args: argparse.Namespace,
    remove_if_exists: bool = True,
    with_tensorboard=True,
):
    """
    Sets up the experiment.
    Args:
        name: The name of the experiment.
    """
    print(f"Arguments: {args}")

    #
    if args.dataset == "celeba":
        args.dataset = "celeba-small"

    # Check if we want to restore from a finished experiment
    if args.load_and_eval is not None:
        # Load args
        old_dir: pathlib.Path = args.load_and_eval.expanduser()
        args_file = os.path.join(old_dir, "args.json")
        old_args = argparse.Namespace(**json.load(open(args_file)))
        old_args.load_and_eval = args.load_and_eval
        old_args.gpu = args.gpu

        print("Loading from existing directory:", old_dir)
        print("Loading with existing args:", pprint.pformat(old_args))

        results_dir = old_dir
        args = old_args
    else:
        # Create result directory
        results_dir = make_results_dir(
            results_dir=args.results_dir,
            experiment_name=name,
            tag=args.tag,
            dataset_name=args.dataset,
            remove_if_exists=remove_if_exists,
        )
        # Save args to file
        save_args(results_dir, args)
    print(f"Results directory: {results_dir}")
    # Setup tensorboard
    if with_tensorboard:
        writer = build_tensorboard_writer(results_dir)
    else:
        writer = None

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
        print("Using GPU device", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    seed_all_rng(args.seed)
    cudnn.benchmark = True

    # Image shape
    image_shape: Shape = get_data_shape(args.dataset)

    # Create RTPT object
    rtpt = RTPT(
        name_initials="SL",
        experiment_name=name + "_" + str(args.tag),
        max_iterations=args.epochs,
    )

    # Start the RTPT tracking
    rtpt.start()

    return (
        args,
        results_dir,
        writer,
        device,
        image_shape,
        rtpt,
    )

def setup_experiment(name: str, cfg: DictConfig, remove_if_exists: bool = False):
    """
    Sets up the experiment.
    Args:
        name: The name of the experiment.
    """
    # Create result directory
    results_dir = make_results_dir(
        results_dir=cfg.results_dir,
        experiment_name=name,
        tag=cfg.tag,
        dataset_name=cfg.dataset,
        remove_if_exists=remove_if_exists,
    )
    # Save args to file
    # save_args(results_dir, cfg)

    # Save args to file
    print(f"Results directory: {results_dir}")
    seed_all_rng(cfg.seed)
    cudnn.benchmark = True
    return results_dir, cfg

def anneal_tau(epoch, max_epochs):
    """Anneal the softmax temperature tau based on the epoch progress."""
    return max(0.5, np.exp(-1 / max_epochs * epoch))


def load_from_checkpoint(results_dir, load_fn, args):
    """Loads the model from a checkpoint.

    Args:
        load_fn: The function to load the model from a checkpoint.
    Returns:
        The loaded model.
    """
    ckpt_dir = os.path.join(results_dir, "tb", "version_0", "checkpoints")
    files = os.listdir(ckpt_dir)
    assert len(files) > 0, "Checkpoint directory is empty"
    ckpt_path = os.path.join(ckpt_dir, files[-1])
    model = load_fn(checkpoint_path=ckpt_path, args=args)
    return model


def save_samples(generate_samples, samples_dir, num_samples, nrow):
    for i in range(5):
        samples = generate_samples(num_samples)
        grid = torchvision.utils.make_grid(
            samples, nrow=nrow, pad_value=0.0, normalize=True
        )
        torchvision.utils.save_image(grid, os.path.join(samples_dir, f"{i}.png"))


from matplotlib.cm import tab10
from matplotlib import cm

TEXTWIDTH = 5.78853
LINEWIDTH = 0.75
ARROW_HEADWIDTH = 5
colors = tab10.colors


def get_figsize(scale: float, aspect_ratio=0.8) -> Tuple[float, float]:
    """
    Scale the default figure size to: (scale * TEXTWIDTH, scale * aspect_ratio * TEXTWIDTH).

    Args:
      scale(float): Figsize scale. Should be lower than 1.0.
      aspect_ratio(float): Aspect ratio (as scale), height to width. (Default value = 0.8)

    Returns:
      Tuple: Tuple containing (width, height) of the figure.

    """
    height = aspect_ratio * TEXTWIDTH
    widht = TEXTWIDTH
    return (scale * widht, scale * height)


def set_style():
    matplotlib.use("pgf")
    plt.style.use(["science", "grid"])  # Need SciencePlots pip package
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


def plot_distribution(model, dataset_name, logger_wandb: WandbLogger = None):
    with torch.no_grad():
        data, targets = generate_data(dataset_name, n_samples=1000)
        fig = plt.figure(figsize=get_figsize(1.0))
        data_cpu = data.cpu()
        delta = 0.05
        xmin, xmax = data_cpu[:, 0].min(), data_cpu[:, 0].max()
        ymin, ymax = data_cpu[:, 1].min(), data_cpu[:, 1].max()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        x = np.arange(xmin * 1.05, xmax * 1.05, delta)
        y = np.arange(ymin * 1.05, ymax * 1.05, delta)
        X, Y = np.meshgrid(x, y)

        Z = torch.exp(
            model(
                torch.from_numpy(np.c_[X.flatten(), Y.flatten()])
                .to(data.device)
                .float()
            ).float()
        ).cpu()
        Z = Z.view(X.shape)
        CS = plt.contourf(X, Y, Z, 100, cmap=plt.cm.viridis)
        plt.colorbar(CS)

        plt.scatter(
            *data_cpu[:500].T,
            label="Data",
            ec="black",
            lw=0.5,
            s=10,
            alpha=0.5,
            color=colors[1],
        )

        plt.xlabel("$X_0$")
        plt.ylabel("$X_1$")
        plt.title(f"Learned PDF represented by the SPN")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)

        # Add figure in numpy "image" to TensorBoard writer
        logger_wandb.log_image("distribution", images=[image])
        plt.close(fig)
