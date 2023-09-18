from collections import deque

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
        return {"buffer": {"size": IntParam(10, 1, 5000), "axis": 0}}

    def setup(self):
        self.buffer = deque(maxlen=self.params.buffer.size.value)

    def process(self, val: Data):
        if val is None:
            return None

        self.buffer.append(val.data)
        return {"out": (np.stack(self.buffer, axis=self.params.buffer.axis.value), val.meta)}

    def buffer_size_changed(self, value):
        self.buffer = deque(maxlen=value)
