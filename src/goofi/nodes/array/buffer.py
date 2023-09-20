import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam


class Buffer(Node):
    def config_input_slots():
        return {"val": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"buffer": {"size": IntParam(10, 1, 5000), "axis": -1}}

    def setup(self):
        self.buffer = None

    def process(self, val: Data):
        if val is None:
            return None

        if self.buffer is None:
            # initialize buffer
            self.buffer = np.array(val.data)
        else:
            # extend the buffer
            maxlen = self.params.buffer.size.value
            self.buffer = np.concatenate((self.buffer, val.data), axis=self.params.buffer.axis.value)
            # remove old data
            slices = [slice(None)] * self.buffer.ndim
            slices[self.params.buffer.axis.value] = slice(-maxlen, None)
            self.buffer = self.buffer[slices]

        return {"out": (self.buffer, val.meta)}
