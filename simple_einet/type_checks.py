import numpy as np
import torch
from typing import Any


class OutOfBoundsException(Exception):
    """
    Exception raised when a value is not within the specified bounds.

    Attributes:
        value -- the value that was out of bounds
        lower_bound -- the lower bound of the valid range
        upper_bound -- the upper bound of the valid range
    """

    def __init__(self, value, lower_bound, upper_bound):
        super().__init__(f"Value {value} was not in bounds: [{lower_bound}, {upper_bound}).")


class InvalidTypeException(Exception):
    """
    Exception raised when a value is of an invalid type.

    Attributes:
        value -- the value that was of an invalid type
        expected_type -- the expected type (or a subclass of this type)
    """

    def __init__(self, value, expected_type):
        super().__init__(
            f"Value {value} was of type {type(value)} but expected to be of type {expected_type} (or a subclass of this type) ."
        )



def _check_bounds(value: Any, expected_type, lower_bound=None, upper_bound=None):
    """
    Check if a given value is within the specified bounds.

    Args:
        value (Any): The value to check.
        expected_type (type): The expected type of the value.
        lower_bound (Any, optional): The lower bound of the value. Defaults to None.
        upper_bound (Any, optional): The upper bound of the value. Defaults to None.

    Raises:
        OutOfBoundsException: If the value is outside the specified bounds.
    """
    # Check lower bound
    if lower_bound:
        if not value >= expected_type(lower_bound):
            raise OutOfBoundsException(value, lower_bound, upper_bound)

    # Check upper bound
    if upper_bound:
        if not value < expected_type(upper_bound):
            raise OutOfBoundsException(value, lower_bound, upper_bound)



def _check_type(value: Any, expected_type):
    """
    Checks if the type of the given value matches the expected type.

    Args:
        value (Any): The value to check the type of.
        expected_type (type): The expected type of the value.

    Raises:
        Exception: If the type of the value is not supported for typecheck.
    """
    # Check if type is from torch
    if isinstance(value, torch.Tensor):
        _check_type_torch(value, expected_type)

    # Check if type is from numpy
    elif type(value).__module__ == np.__name__:
        _check_type_numpy(value, expected_type)
    elif type(value) == int or type(value) == float:
        _check_type_core(value, expected_type)
    else:
        raise Exception(f"Unsupported type ({type(value)}) for typecheck.")



def _check_type_core(value: Any, expected_type: type) -> None:
    """
    Check if the given value is of the expected type.

    Args:
        value (Any): The value to check the type of.
        expected_type (type): The expected type of the value.

    Raises:
        InvalidTypeException: If the value is not of the expected type.
    """
    if expected_type == float and not isinstance(value, float):
        raise InvalidTypeException(value, expected_type)
    elif expected_type == int and not isinstance(value, int):
        raise InvalidTypeException(value, expected_type)



def _check_type_numpy(value: Any, expected_type):
    """
    Check if the given value is of the expected type using numpy's data types.

    Args:
        value (Any): The value to check the type of.
        expected_type (type): The expected type of the value.

    Raises:
        InvalidTypeException: If the value is not of the expected type.

    """
    # Check float
    if expected_type == float:
        if not isinstance(value, np.floating):
            raise InvalidTypeException(value, expected_type)
    # Check integer
    elif expected_type == int:
        if not isinstance(value, np.integer):
            raise InvalidTypeException(value, expected_type)
    else:
        raise Exception(f"Unexpected data type, must be either int or float, but was {expected_type}")


def _check_type_torch(value: torch.Tensor, expected_type):
    """
    Check if the data type of a torch tensor matches the expected type.

    Args:
        value (torch.Tensor): The tensor to check the data type of.
        expected_type (type): The expected data type. Must be either `int` or `float`.

    Raises:
        InvalidTypeException: If the data type of the tensor does not match the expected type.

    """
    # Get torch data type
    dtype = value.dtype

    # If we expect float, check if dtype is a floating point, vice versa for int
    if expected_type == float:
        if not dtype.is_floating_point:
            raise InvalidTypeException(value, expected_type)
    elif expected_type == int:
        if dtype.is_floating_point:
            raise InvalidTypeException(value, expected_type)
    else:
        raise Exception(f"Unexpected data type, must be either int or float, but was {expected_type}")



def check_valid(value: Any, expected_type, lower_bound=None, upper_bound=None, allow_none: bool = False):
    """
    Check if a given value is valid based on its expected type and optional bounds.

    Args:
        value (Any): The value to check.
        expected_type (type): The expected type of the value.
        lower_bound (Any, optional): The lower bound for the value. Defaults to None.
        upper_bound (Any, optional): The upper bound for the value. Defaults to None.
        allow_none (bool, optional): Whether to allow None as a valid value. Defaults to False.

    Returns:
        Any: The value, if it is valid.

    Raises:
        Exception: If the value is not valid based on the expected type and/or bounds.
    """
    if allow_none and value is None:
        return value
    if not allow_none and value is None:
        raise Exception(f"Invalid input: Got None, but expected type {expected_type}.")
    # First check if the type is valid
    _check_type(value, expected_type)

    # Then check if value is inbounds
    _check_bounds(value, expected_type, lower_bound, upper_bound)

    return expected_type(value)
