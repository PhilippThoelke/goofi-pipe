from goofi.data import Data, DataType
from goofi.node import Node


class Transpose(Node):
    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {}

    def process(self, array: Data):
        if array is None or array.data is None:
            return None

        if array.data.ndim != 2:
            raise ValueError("Data must be 2D (TODO: support n-dimensional arrays).")

        result = array.data.T

        # transpose channel names
        ch_names = {}
        if "dim0" in array.meta["channels"]:
            ch_names["dim1"] = array.meta["channels"]["dim0"]
        if "dim1" in array.meta["channels"]:
            ch_names["dim0"] = array.meta["channels"]["dim1"]
        array.meta["channels"] = ch_names

        return {"out": (result, array.meta)}
