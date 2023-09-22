import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Math(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "math": {
                "pre_add": FloatParam(0.0, -10.0, 10.0),
                "multiply": FloatParam(1.0, -10.0, 10.0),
                "post_add": FloatParam(0.0, -10.0, 10.0),
            },
            "map": {
                "input_min": FloatParam(0.0, -10.0, 10.0),
                "input_max": FloatParam(1.0, -10.0, 10.0),
                "output_min": FloatParam(0.0, -10.0, 10.0),
                "output_max": FloatParam(1.0, -10.0, 10.0),
            },
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        signal = np.array(data.data)

        # apply math operations
        signal += self.params.math.pre_add.value
        signal *= self.params.math.multiply.value
        signal += self.params.math.post_add.value

        # rescale signal from input range to output range
        signal = self.rescale(
            signal,
            self.params.map.input_min.value,
            self.params.map.input_max.value,
            self.params.map.output_min.value,
            self.params.map.output_max.value,
        )

        return {"out": (signal, data.meta)}

    @staticmethod
    def rescale(signal, input_min, input_max, output_min, output_max):
        # rescale signal from input range to output range
        return ((signal - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
