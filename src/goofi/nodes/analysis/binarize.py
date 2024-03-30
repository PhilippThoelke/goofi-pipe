import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class Binarize(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "bin_data": DataType.ARRAY,
        }

    def config_params():
        return {
            "parameters": {
                "threshold_type": StringParam("both", options=["both", "above", "below"]),
                "threshold": FloatParam(2.0, 0.0, 5.0),
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

        thresh_type = self.params["parameters"]["threshold_type"].value
        thresh = self.params["parameters"]["threshold"].value
        events = self.eop.binarized_events(data.data, threshold=thresh, thresh_type=thresh_type, null_value=0)
        return {"bin_data": (np.array(events), data.meta)}
