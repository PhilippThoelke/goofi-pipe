from enum import Enum

import numpy as np


class MessageType(Enum):
    ADD_PIPE = 1
    REMOVE_PIPE = 2
    DATA = 3


class DataType(Enum):
    FLOAT_1D = 1
    FLOAT_2D = 2
    FLOAT_3D = 3
    STRING = 4

    def check_type(self, value):
        if value is None:
            raise ValueError("Expected value, got None")

        if self == DataType.STRING:
            # make sure it's a string
            if not isinstance(value, str):
                raise ValueError(f"Expected string, got {type(value)}")
            return

        # all other types are numpy arrays, so make sure it's a numpy array
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(value)}")

        # make sure the number of dimensions is correct
        if self == DataType.FLOAT_1D:
            if value.ndim != 1:
                raise ValueError(f"Expected 1D array, got {value.ndim}D array")
        elif self == DataType.FLOAT_2D:
            if value.ndim != 2:
                raise ValueError(f"Expected 2D array, got {value.ndim}D array")
        elif self == DataType.FLOAT_3D:
            if value.ndim != 3:
                raise ValueError(f"Expected 3D array, got {value.ndim}D array")
        else:
            raise ValueError(f"Unknown data type {self}")
