import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import stats


def compute_padding(input_size, kernel_size, stride):
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


def pool2d(A, kernel_size, stride, padding=0, pool_mode="max"):
    """
    2D Pooling

    Taken from https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )

    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "mode":
        A_w_reshape = np.reshape(A_w, (A_w.shape[0], -1))
        return stats.mode(A_w_reshape, axis=1)[0].reshape(output_shape)
