import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Reduce(Node):
    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"reduce": {"method": StringParam("mean", options=["mean", "median", "min", "max", "std"]), "axis": 0}}

    def process(self, array: Data):
        if array is None:
            return None

        axis = self.params.reduce.axis.value
        if axis < 0:
            axis = array.data.ndim + axis

        result = getattr(np, self.params.reduce.method.value)(array.data, axis=axis)

        if f"dim{axis}" in array.meta["channels"]:
            del array.meta["channels"][f"dim{axis}"]

        for i in range(axis, result.ndim + 1):
            if f"dim{i+1}" in array.meta["channels"]:
                array.meta["channels"][f"dim{i}"] = array.meta["channels"][f"dim{i+1}"]
                del array.meta["channels"][f"dim{i+1}"]

        return {"out": (result, array.meta)}
