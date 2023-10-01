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
            axis = self.params.buffer.axis.value
            try:
                self.buffer = np.concatenate((self.buffer, val.data), axis=axis)
            except ValueError:
                # data shape changed, reset buffer
                self.buffer = np.array(val.data)

            # remove old data
            if self.buffer.shape[axis] > maxlen:
                self.buffer = np.take(self.buffer, range(-maxlen, 0), axis=axis)

        return {"out": (self.buffer, val.meta)}
