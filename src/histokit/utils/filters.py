from enum import Enum

import numpy as np
from numpy.lib.stride_tricks import as_strided


def compute_padding(input_size: tuple[int, int], kernel_size: int, stride: int) -> tuple[int, int]:
    """
    Compute the required padding for a 2D array to ensure complete pooling coverage.

    Parameters:
        input_size (tuple): The dimensions of the input array (height, width).
        kernel_size (int): The size of the pooling kernel.
        stride (int): The stride of the pooling operation.

    Returns:
        tuple: Padding for height and width (pad_h, pad_w).
    """
    input_height, input_width = input_size

    # calculate the output dimensions without padding
    out_height = (input_height - kernel_size) % stride
    out_width = (input_width - kernel_size) % stride

    # compute necessary padding to make dimensions divisible by stride
    pad_h = (stride - out_height) % stride
    pad_w = (stride - out_width) % stride

    # since padding is applied equally on both sides, divide by 2
    pad_h = pad_h // 2
    pad_w = pad_w // 2

    return pad_h, pad_w


class PoolMode(Enum):
    MAX = "max"
    AVG = "avg"
    MIN = "min"

def pool2d(A: np.ndarray, kernel_size: int, stride: int, padding: int = 0, pool_mode: PoolMode = PoolMode.MAX) -> np.ndarray:
    """
    2D Pooling

    Taken from https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: PoolMode, 'max' or 'avg' or 'min'
    """
    # Padding
    A_padded = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A_padded.shape[0] - kernel_size) // stride + 1,
        (A_padded.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size_tuple = (kernel_size, kernel_size)
    A_w = as_strided(
        A_padded,
        shape=output_shape + kernel_size_tuple,
        strides=(stride * A_padded.strides[0], stride * A_padded.strides[1]) + A_padded.strides,
    )

    A_w = A_w.reshape(-1, *kernel_size_tuple)

    # Return the result of pooling
    if pool_mode == PoolMode.MAX:
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == PoolMode.AVG:
        return A_w.mean(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == PoolMode.MIN:
        return A_w.min(axis=(1, 2)).reshape(output_shape)
    else:
        raise ValueError(f"Unknown pool_mode: {pool_mode}")
