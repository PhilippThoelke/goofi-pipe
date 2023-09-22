import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class LempelZiv(Node):

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"lzc": DataType.ARRAY}

    def config_params():
        return {"lempel_ziv": {"axis": -1}}

    def setup(self):
        from antropy import lziv_complexity
        self.lzc_fn = lziv_complexity

    def process(self, data: Data):
        if data is None:
            return None

        # TODO: implement different types of binarization (e.g. median)
        # binarize the data
        binarized = data.data > np.mean(data.data, axis=self.params.lempel_ziv.axis.value, keepdims=True)
        # compute normalized Lempel-Ziv complexity
        lzc = np.apply_along_axis(self.lzc_fn, self.params.lempel_ziv.axis.value, binarized, normalize=True)

        return {"lzc": (lzc, data.meta)}
