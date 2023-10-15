import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class LempelZiv(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"lzc": DataType.ARRAY}

    def config_params():
        return {
            "lempel_ziv": {
                "binarization": StringParam("mean", options=["mean", "median"]),
                "axis": -1,
            }
        }

    def setup(self):
        from antropy import lziv_complexity

        self.compute_lzc = lziv_complexity

    def process(self, data: Data):
        if data is None:
            # no data, skip processing
            return None

        if self.params.lempel_ziv.binarization.value == "mean":
            # binarize using the mean
            binarized = data.data > np.mean(data.data, axis=self.params.lempel_ziv.axis.value, keepdims=True)
        elif self.params.lempel_ziv.binarization.value == "median":
            # binarize using the median
            binarized = data.data > np.median(data.data, axis=self.params.lempel_ziv.axis.value, keepdims=True)
        else:
            raise ValueError("Unknown binarization method")

        # compute normalized Lempel-Ziv complexity
        lzc = np.apply_along_axis(self.compute_lzc, self.params.lempel_ziv.axis.value, binarized, normalize=True)

        return {"lzc": (lzc, data.meta)}
