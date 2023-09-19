import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Stack(Node):
    def config_input_slots():
        return {"a": DataType.ARRAY, "b": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"stack": {"axis": 0}}

    def process(self, a: Data, b: Data):
        if a is None or b is None:
            return None

        return {"out": (np.stack([a.data, b.data], axis=self.params.stack.axis.value), {})}
