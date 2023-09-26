from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import numpy as np


class DataType(Enum):
    """
    The type of data contained in a data object. The data type determines the type of the data field in the data
    object.

    - `ARRAY`: An n-dimensional numpy array.
    - `STRING`: A string.
    - `TABLE`: A named map of data objects (i.e. a dict of str keys and Data values).
    """

    ARRAY = 0
    STRING = 1
    TABLE = 2

    def empty(self) -> Any:
        """
        Get an empty numpy array of the correct type for this data type.
        """
        if self == DataType.ARRAY:
            return np.zeros(0)
        elif self == DataType.STRING:
            return ""
        elif self == DataType.TABLE:
            return dict()
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

    def __post_init__(self):
        """
        Check that the data type is valid. The data field must be of the correct type for the data type.
        """
        # general checks
        if self.dtype is None or not isinstance(self.dtype, DataType):
            raise ValueError(f"Expected dtype of type DataType, got {type(self.dtype)}.")
        if self.meta is None or not isinstance(self.meta, dict):
            raise ValueError(f"Expected meta of type dict, got {type(self.meta)}.")

        # validate data type
        if isinstance(self.dtype, DataType) and self.dtype not in DTYPE_TO_TYPE:
            raise RuntimeError(
                f"Data type {self.dtype} is defined but not handled in check_data. "
                "Please report this bug at https://github.com/PhilippThoelke/goofi-pipe/issues."
            )
        if not isinstance(self.data, DTYPE_TO_TYPE[self.dtype]):
            raise ValueError(f"Expected data of type {DTYPE_TO_TYPE[self.dtype]}, got {type(self.data)}.")

        if self.dtype == DataType.ARRAY and self.data.ndim == 0:
            # make sure that arrays are at least 1-dimensional
            self.data = np.array([self.data])
        elif self.dtype == DataType.TABLE:
            for key, value in self.data.items():
                # make sure that table keys are strings
                if not isinstance(key, str):
                    raise ValueError(f"Expected table keys of type str, got {type(key)}.")
                # make sure that table values are Data objects
                if not isinstance(value, Data):
                    raise ValueError(f"Expected table values of type Data, got {type(value)}.")

        # TODO: add better metadata checks


DTYPE_TO_TYPE = {
    DataType.ARRAY: np.ndarray,
    DataType.STRING: str,
    DataType.TABLE: dict,
}
