import numpy as np

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

        result = np.transpose(array.data)
        return {"out": (result, array.meta)}
