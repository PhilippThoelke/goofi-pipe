import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Clip(Node):
    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"clip": {"min": FloatParam(-1), "max": FloatParam(1)}}

    def process(self, array: Data):
        if array is None:
            return None

        result = np.clip(array.data, self.params.clip.min.value, self.params.clip.max.value)

        return {"out": (result, array.meta)}
