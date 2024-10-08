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

    def __str__(self) -> str:
        return self.name


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

        # configure the data object by calling the appropriate configure method
        getattr(self, f"_configure_{self.dtype.name.lower()}")()

    def _configure_array(self):
        """
        Configure the data object as an array and populate the metadata.
        """
        # make sure that arrays are at least 1-dimensional
        if self.data.ndim == 0:
            self.data = np.array([self.data])

        # populate the metadata
        self.meta["shape"] = self.data.shape

        # make sure that the metadata contains a channels dict
        if "channels" in self.meta:
            assert isinstance(self.meta["channels"], dict), "Expected channels to be a dict."
        else:
            self.meta["channels"] = {}

        # check the channels metadata
        for dim in range(self.data.ndim):
            if f"dim{dim}" in self.meta["channels"]:
                assert isinstance(self.meta["channels"][f"dim{dim}"], list), f"Expected dim{dim} to be a list."
                assert len(self.meta["channels"][f"dim{dim}"]) == self.data.shape[dim], (
                    f"Expected dim{dim} to have length {self.data.shape[dim]} but got "
                    f"{len(self.meta['channels'][f'dim{dim}'])}."
                )
        for dim in self.meta["channels"].keys():
            assert dim.startswith("dim"), f"Expected channel key to start with 'dim', got {dim}."
            dim = dim[3:]
            assert dim.isdigit(), f"Expected channel key to end with a number, got {dim}."
            dim = int(dim)
            assert dim < self.data.ndim, f"Expected channel key to be less than {self.data.ndim}, got {dim}."

    def _configure_string(self):
        pass

    def _configure_table(self):
        """
        Configure the data object as a table and populate the metadata.
        """
        for key, value in self.data.items():
            # make sure that table keys are strings
            if not isinstance(key, str):
                raise ValueError(f"Expected table keys of type str, got {type(key)}.")
            # make sure that table values are Data objects
            if not isinstance(value, Data):
                raise ValueError(f"Expected table values of type Data, got {type(value)}.")


DTYPE_TO_TYPE = {
    DataType.ARRAY: (np.ndarray, np.number),
    DataType.STRING: str,
    DataType.TABLE: dict,
}
