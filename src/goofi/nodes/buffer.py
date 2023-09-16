from collections import deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Buffer(Node):
    def config_input_slots():
        return {"a": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"buffer": {"size": 500, "axis": 0}}

    def setup(self):
        self.buffer = deque(maxlen=self.params.buffer.size.value)

    def process(self, a: Data):
        if a is None:
            return None

        self.buffer.append(a.data)

        return {"out": (np.stack(self.buffer, axis=self.params.buffer.axis.value), {})}


if __name__ == "__main__":
    ref, a = Buffer.create_local()

    print(a.process(Data(DataType.ARRAY, np.array([1, 2, 3]), {})))

    ref.terminate()
