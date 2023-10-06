from scipy.ndimage import gaussian_filter1d
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class Smooth(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "smooth": {
                "sigma": FloatParam(2.0, 0.1, 20.0, doc="standard deviation for Gaussian kernel"),
                "axis": IntParam(-1, 0, 2),
            }
        }

    def process(self, data: Data):
        if data is None:
            return None

        return {
            "out": (
                gaussian_filter1d(data.data, self.params.smooth.sigma.value, axis=self.params.smooth.axis.value),
                data.meta,
            )
        }
