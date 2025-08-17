import numpy as np
from numpy import int64

def bilinear_interp_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This is the vectorized implementation of bilinear interpolation.
    - a is a ND array with shape [N, H1, W1, C], dtype = int64
    - b is a ND array with shape [N, H2, W2, 2], dtype = float64
    - return a ND array with shape [N, H2, W2, C], dtype = int64
    """
    N, H1, W1, C = a.shape
    N1, H2, W2, _ = b.shape
    assert N == N1

    res = np.empty((N,H2,W2,C), dtype=int64)

    x = b[:,:,:,0]
    y = b[:,:,:,1]
    x_idx = np.floor(x).astype(int64)
    y_idx = np.floor(y).astype(int64)
    x_mul = (x - x_idx)[:,:,:,None]
    y_mul = (y - y_idx)[:,:,:,None]
    n_idx = np.arange(N)[:,None,None]
    res[:] = a[n_idx, x_idx, y_idx] * (1 - x_mul) * (1 - y_mul) + \
             a[n_idx, x_idx + 1, y_idx] * x_mul * (1 - y_mul) + \
             a[n_idx, x_idx, y_idx + 1] * (1 - x_mul) * y_mul + \
             a[n_idx, x_idx + 1, y_idx + 1] * x_mul * y_mul

    return res