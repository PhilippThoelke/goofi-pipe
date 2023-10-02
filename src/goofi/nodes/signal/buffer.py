import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam


class Buffer(Node):
    def config_input_slots():
        return {"val": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"buffer": {"size": IntParam(10, 1, 5000), "axis": -1, "reset": BoolParam(False, trigger=True)}}

    def setup(self):
        self.name_buffer = None
        self.buffer = None

    def process(self, val: Data):
        if val is None:
            return None

        if self.params.buffer.reset.value:
            # reset buffer
            self.buffer = None

        maxlen = self.params.buffer.size.value
        axis = self.params.buffer.axis.value
        # convert negative axis to positive
        if axis < 0:
            axis = val.data.ndim + axis

        if self.buffer is not None:
            # add dimension to buffer and val.data if axis is out of bounds
            if self.buffer.ndim < (axis + 1):
                self.buffer = np.expand_dims(self.buffer, axis=axis)
            if val.data.ndim < (axis + 1):
                val.data = np.expand_dims(val.data, axis=axis)

            try:
                # concatenate data to buffer
                self.buffer = np.concatenate((self.buffer, val.data), axis=axis)
                # update channel names
                if f"dim{axis}" in val.meta["channels"]:
                    self.name_buffer += val.meta["channels"][f"dim{axis}"]
            except ValueError:
                # data shape changed, reset buffer
                self.buffer = None

        # (re-)initialize buffer
        if self.buffer is None:
            self.buffer = np.array(val.data)
            self.name_buffer = val.meta["channels"][f"dim{axis}"] if f"dim{axis}" in val.meta["channels"] else None

        # remove data that exceeds the maximum length
        if self.buffer.shape[axis] > maxlen:
            self.buffer = np.take(self.buffer, range(-maxlen, 0), axis=axis)
            if self.name_buffer is not None:
                self.name_buffer = self.name_buffer[-maxlen:]

        # update channel names
        if self.name_buffer is not None:
            val.meta["channels"][f"dim{axis}"] = self.name_buffer

        return {"out": (self.buffer, val.meta)}
