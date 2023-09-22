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
                "add_value": FloatParam(0.0, -100.0, 100.0),
                "mult_value": FloatParam(1.0, -10.0, 10.0),
                "input_min": FloatParam(0.0, -100.0, 100.0),
                "input_max": FloatParam(1.0, -100.0, 100.0),
                "output_min": FloatParam(0.0, -100.0, 100.0),
                "output_max": FloatParam(1.0, -100.0, 100.0),
                "add_before_mult": True,
            }
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        signal = np.array(data.data)

        if self.params.math.add_before_mult.value:
            signal = (signal + self.params.math.add_value.value) * self.params.math.mult_value.value
        else:
            signal = signal * self.params.math.mult_value.value + self.params.math.add_value.value

        # rescale signal from input range to output range
        signal = self.rescale(
            signal,
            self.params.math.input_min.value,
            self.params.math.input_max.value,
            self.params.math.output_min.value,
            self.params.math.output_max.value,
        )

        return {"out": (signal, data.meta)}

    @staticmethod
    def rescale(signal, input_min, input_max, output_min, output_max):
        # rescale signal from input range to output range
        return ((signal - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
