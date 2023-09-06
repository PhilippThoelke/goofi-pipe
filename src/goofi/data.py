from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import numpy as np


class DataType(Enum):
    """
    The type of data contained in a data object. The data type determines the type of the data field in the data
    object.

    - `FLOAT_1D`: A 1D numpy array of floats.
    - `FLOAT_2D`: A 2D numpy array of floats.
    - `FLOAT_3D`: A 3D numpy array of floats.
    - `STRING`: A string.
    """

    FLOAT_1D = 1
    FLOAT_2D = 2
    FLOAT_3D = 3
    STRING = 4

    def empty(self) -> Any:
        """
        Get an empty numpy array of the correct type for this data type.
        """
        if self == DataType.FLOAT_1D:
            return np.empty(0)
        elif self == DataType.FLOAT_2D:
            return np.empty((0, 0))
        elif self == DataType.FLOAT_3D:
            return np.empty((0, 0, 0))
        elif self == DataType.STRING:
            return ""
        else:
            raise ValueError(f"Unknown data type {self}")


@dataclass
class Data:
    """
    Data objects are used to send data between nodes. They contain the data object (`data`) itself, the data type
    (`dtype`), and some metadata (`meta`). The data types are limited to the types defined in the `DataType` enum.
    The data field must be of the correct type for the data type. The metadata field is a dict that can contain
    any metadata about the data. The metadata field can be used to store information about the data, such as
    channel names, sampling frequencies, etc.

    ### Parameters
    `dtype` : DataType
        The data type of the data object.
    `data` : Any
        A data object matching the data type.
    `meta` : Dict[str, Any]
        The metadata dictionary.
    """

    dtype: DataType
    data: Any
    meta: Dict[str, Any]

    def _check_string(self):
        """General string checks."""
        if not isinstance(self.data, str):
            raise ValueError(f"Expected string, got {type(self.data)}")

    def _check_array(self):
        """General array checks."""
        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(self.data)}")

        if self.dtype == DataType.FLOAT_1D:
            # data should be a 1D array
            if self.data.ndim != 1:
                raise ValueError(f"Expected 1D array, got {self.data.ndim}D array")
        elif self.dtype == DataType.FLOAT_2D:
            # data should be a 2D array
            if self.data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {self.data.ndim}D array")
        elif self.dtype == DataType.FLOAT_3D:
            # data should be a 3D array
            if self.data.ndim != 3:
                raise ValueError(f"Expected 3D array, got {self.data.ndim}D array")
        else:
            raise NotImplementedError(f"Unknown array data type {self.dtype}")

    def __post_init__(self):
        """
        Check that the data type is valid. The data field must be of the correct type for the data type.
        """
        # general checks
        if self.dtype is None:
            raise ValueError("Expected data type, got None")
        if self.data is None:
            raise ValueError("Expected data object, got None")
        if self.meta is None or not isinstance(self.meta, dict):
            raise ValueError(f"Expected metadata dict, got {type(self.meta)}")

        # dtype-specific checks
        if self.dtype == DataType.STRING:
            self._check_string()
        elif self.dtype in [DataType.FLOAT_1D, DataType.FLOAT_2D, DataType.FLOAT_3D]:
            self._check_array()
        elif isinstance(self.dtype, DataType):
            raise RuntimeError(
                f"Data type {self.dtype} is defined but not handled in check_data. "
                "Please report this in an issue: https://github.com/PhilippThoelke/goofi-pipe/issues."
            )
        else:
            raise ValueError(f"Expected DataType, got {type(self.dtype)}")

        # TODO: add better metadata checks
