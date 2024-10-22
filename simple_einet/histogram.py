"""
Translates numpys histogram "auto" bin size estimation to PyTorch.
"""


import numpy as np
import torch
import operator

# NumPy implementations
def _hist_bin_fd_np(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)

def _hist_bin_sturges_np(x):
    return np.ptp(x) / (np.log2(x.size) + 1.0)

def _hist_bin_auto_np(x):
    fd_bw = _hist_bin_fd_np(x)
    sturges_bw = _hist_bin_sturges_np(x)
    return min(fd_bw, sturges_bw) if fd_bw else sturges_bw

# PyTorch implementations
def _hist_bin_fd_torch(x):
    iqr = torch.quantile(x, 0.75) - torch.quantile(x, 0.25)
    return 2.0 * iqr * x.size(0) ** (-1.0 / 3.0)

def _hist_bin_sturges_torch(x):
    return _ptp_torch(x) / (torch.log2(torch.tensor(x.size(0)).float()) + 1.0)

def _ptp_torch(x):
    return x.max() - x.min()

def _hist_bin_auto_torch(x):
    fd_bw = _hist_bin_fd_torch(x)
    sturges_bw = _hist_bin_sturges_torch(x)
    return min(fd_bw, sturges_bw) if fd_bw > 0 else sturges_bw


def _get_bin_edges_torch(a, range=None, weights=None):
    """
    Computes the bins used internally by `histogram` in PyTorch.

    Parameters
    ----------
    a : 1D Tensor
        Ravelled data array.
    range : tuple
        Lower and upper range of the bins.
    weights : Tensor, optional
        Ravelled weights array, or None.

    Returns
    -------
    bin_edges : Tensor
        Array of bin edges.
    uniform_bins : tuple
        The lower bound, upper bound, and number of bins for uniform binning.
    """
    # Assume bins is "auto" as per the user's request.
    if weights is not None:
        raise TypeError("Automated bin estimation is not supported for weighted data")

    first_edge, last_edge = _get_outer_edges_torch(a, range)

    # Filter the array based on the range if necessary
    if range is not None:
        a = a[(a >= first_edge) & (a <= last_edge)]

    # If the input tensor is empty after filtering, use 1 bin
    if a.numel() == 0:
        n_equal_bins = 1
    else:
        # Calculate the bin width using the Freedman-Diaconis estimator
        width = _hist_bin_auto_torch(a)
        if width > 0:
            n_equal_bins = int(torch.ceil((last_edge - first_edge) / width).item())
        else:
            # If width is zero, fall back to 1 bin
            n_equal_bins = 1

    # Compute bin edges
    bin_edges = torch.linspace(
        first_edge, last_edge, n_equal_bins + 1, dtype=torch.float32
    )

    return bin_edges, (first_edge, last_edge, n_equal_bins)

def _get_outer_edges_torch(a, range):
    """
    Determine the outer bin edges from either the data or the given range.
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in range parameter.")
        if not (torch.isfinite(torch.tensor(first_edge)) and torch.isfinite(torch.tensor(last_edge))):
            raise ValueError(f"Supplied range [{first_edge}, {last_edge}] is not finite.")
    elif a.numel() == 0:
        # Handle empty tensor case
        first_edge, last_edge = 0.0, 1.0
    else:
        first_edge, last_edge = a.min().item(), a.max().item()
        if not (torch.isfinite(torch.tensor(first_edge)) and torch.isfinite(torch.tensor(last_edge))):
            raise ValueError(f"Autodetected range [{first_edge}, {last_edge}] is not finite.")

    # Expand if the range is empty to avoid divide-by-zero errors
    if first_edge == last_edge:
        first_edge -= 0.5
        last_edge += 0.5

    return first_edge, last_edge

def _get_bin_edges_np(a, bins, range=None, weights=None):
    """
    Computes the bins used internally by `histogram`.

    Parameters
    ==========
    a : ndarray
        Ravelled data array
    bins, range
        Forwarded arguments from `histogram`.
    weights : ndarray, optional
        Ravelled weights array, or None

    Returns
    =======
    bin_edges : ndarray
        Array of bin edges
    uniform_bins : (Number, Number, int):
        The upper bound, lowerbound, and number of bins, used in the optimized
        implementation of `histogram` that works on uniform bins.
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, str):
        bin_name = bins
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if weights is not None:
            raise TypeError("Automated estimation of the number of "
                            "bins is not supported for weighted data")

        first_edge, last_edge = _get_outer_edges_np(a, range)

        # truncate the range if needed
        if range is not None:
            keep = (a >= first_edge)
            keep &= (a <= last_edge)
            if not np.logical_and.reduce(keep):
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_auto_np(a)
            if width:
                n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError(
                '`bins` must be an integer, a string, or an array') from e
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

        first_edge, last_edge = _get_outer_edges_np(a, range)

    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError(
                '`bins` must increase monotonically, when an array')

    else:
        raise ValueError('`bins` must be 1d, when an array')

    if n_equal_bins is not None:
        # gh-10322 means that type resolution rules are dependent on array
        # shapes. To avoid this causing problems, we pick a type now and stick
        # with it throughout.
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = np.linspace(
            first_edge, last_edge, n_equal_bins + 1,
            endpoint=True, dtype=bin_type)
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return bin_edges, None

def _get_outer_edges_np(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge

def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    # coerce to a single type
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong
    }
    dt = np.result_type(a, b)
    try:
        unsigned_dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned.  The input may be negative python integers so
        # ensure we pass in arrays with the initial dtype (related to NEP 50).
        return np.subtract(np.asarray(a, dtype=dt), np.asarray(b, dtype=dt),
                           casting='unsafe', dtype=unsigned_dt)


if "__main__" == __name__:
    # Generate random data
    data_np = np.random.randn(1000)  # Numpy data
    data_torch = torch.tensor(data_np, dtype=torch.float32)  # Convert to torch

    # Compare results for _hist_bin_fd
    fd_np = _hist_bin_fd_np(data_np)
    fd_torch = _hist_bin_fd_torch(data_torch).item()

    # Compare results for _hist_bin_sturges
    sturges_np = _hist_bin_sturges_np(data_np)
    sturges_torch = _hist_bin_sturges_torch(data_torch).item()

    # Compare results for _hist_bin_auto
    auto_np = _hist_bin_auto_np(data_np)
    auto_torch = _hist_bin_auto_torch(data_torch).item()

    # Print comparisons
    print(f"Freedman-Diaconis (Numpy): {fd_np}")
    print(f"Freedman-Diaconis (Torch): {fd_torch}")

    print(f"Sturges (Numpy): {sturges_np}")
    print(f"Sturges (Torch): {sturges_torch}")

    print(f"Auto (Numpy): {auto_np}")
    print(f"Auto (Torch): {auto_torch}")

    # Call the function and print the results
    bin_edges, (first_edge, last_edge, n_bins) = _get_bin_edges_torch(data_torch)

    print(f"Bin edges torch: {bin_edges}")
    print(f"Range torch: ({first_edge}, {last_edge}), Number of bins: {n_bins}")

    bin_edges, (first_edge, last_edge, n_bins) = _get_bin_edges_np(data_np, bins="auto")
    print(f"Bin edges numpy: {bin_edges}")
    print(f"Range numpy: ({first_edge}, {last_edge}), Number of bins: {n_bins}")
