import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class LSLOut(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_params():
        return {
            "lsl": {
                "source_name": "goofi",
                "stream_name": "stream",
            }
        }

    def setup(self):
        from mne_lsl import lsl

        self.lsl = lsl
        self.outlet = None

    def process(self, data: Data):
        if data is None or len(data.data) == 0:
            return

        if self.outlet is not None and self.outlet.n_channels != len(data.data):
            self.outlet = None

        if self.outlet is None:
            info = self.lsl.StreamInfo(
                self.params.lsl.stream_name.value,
                "Data",
                len(data.data),
                data.meta["sfreq"] if "sfreq" in data.meta else self.lsl.IRREGULAR_RATE,
                "float32",
                self.params.lsl.source_name.value,
            )
            if "dim0" in data.meta["channels"]:
                info.set_channel_names(data.meta["channels"]["dim0"])

            self.outlet = self.lsl.StreamOutlet(info)

        try:
            if data.data.ndim == 1:
                self.outlet.push_sample(data.data.astype(np.float32))
            elif data.data.ndim == 2:
                self.outlet.push_chunk(np.ascontiguousarray(data.data.T.astype(np.float32)))
            else:
                raise ValueError("Only one- and two-dimensional arrays are supported.")
        except Exception as e:
            self.outlet = None
            raise e

    def lsl_stream_name_changed(self, value: str):
        if self.outlet is not None:
            self.outlet = None
