import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Reduce(Node):
    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"reduce": {"method": "mean", "axis": 0}}

    def process(self, array: Data):
        if array is None:
            return None

        if not hasattr(np, self.params.reduce.method.value):
            raise ValueError(f"Unknown reduce method {self.params.reduce.method.value}. Supported are all NumPy methods.")

        result = getattr(np, self.params.reduce.method.value)(array.data, axis=self.params.reduce.axis.value)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return {"out": (result, array.meta)}
