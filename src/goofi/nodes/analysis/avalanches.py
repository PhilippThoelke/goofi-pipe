import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Avalanches(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "size": DataType.ARRAY,
            "duration": DataType.ARRAY,
        }

    def config_params():
        return {
            "parameters": {
                "time_bin": FloatParam(0.008, 0.0, 0.05),
            }
        }

    def setup(self):
        import edgeofpy as eop

        self.eop = eop

    def process(self, data: Data):
        if data.data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim == 1:
            # create a new axis if the data is 1D
            data.data = data.data[np.newaxis, :]
        time_bin = self.params["parameters"]["time_bin"].value
        avalanches, _, _ = self.eop.avalanche._det_avls(data.data, s_freq=data.meta["sfreq"], time=None, max_iei=time_bin)
        sizes = np.array([np.float(x["size"]) for x in avalanches])
        durations = np.array([x["dur_sec"] for x in avalanches])
        return {"size": (sizes, {}), "duration": (durations, {})}
