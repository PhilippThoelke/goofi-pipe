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

        return {
            "out": (
                getattr(np, self.params.reduce.method.value)(array.data, axis=self.params.reduce.axis.value),
                array.meta,
            )
        }
