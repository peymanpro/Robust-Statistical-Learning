"""Shared numerical array conventions."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# A matrix is represented by a two-dimensional float64 ndarray.
Matrix = FloatArray

# A vector is represented by a one-dimensional float64 ndarray.
Vector = FloatArray