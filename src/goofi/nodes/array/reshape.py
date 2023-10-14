import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam

class Reshape(Node):
    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"reshape": {"new_shape_pos1": IntParam(-1, -1, 3),
                            "new_shape_pos2": IntParam(2, -1, 3), }}

    def process(self, array: Data):
        if array is None:
            return None

        a = self.params.reshape.new_shape_pos1.value
        b = self.params.reshape.new_shape_pos2.value

        result = np.reshape(array.data, (a, b))

        return {"out": (result, array.meta)}
