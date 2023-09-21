import numpy as np
from antropy import lziv_complexity

from goofi.data import Data, DataType
from goofi.node import Node


class LempelZiv(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"lzc": DataType.ARRAY}

    def config_params():
        return {"lempel_ziv": {"axis": -1}}

    def process(self, data: Data):
        if data is None:
            return None

        binarized = data.data > np.mean(data.data, axis=self.params.lempel_ziv.axis.value, keepdims=True)
        lzc = np.apply_along_axis(lziv_complexity, self.params.lempel_ziv.axis.value, binarized, normalize=True)

        print(lzc.shape)

        return {"lzc": (lzc, {})}
