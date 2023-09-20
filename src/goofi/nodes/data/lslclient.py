from typing import Any, Dict, Tuple

import numpy as np
from mne_realtime import LSLClient as MNE_LSLClient

from goofi.data import DataType
from goofi.node import Node


class LSLClient(Node):
    def config_params():
        return {
            "lsl_stream": {"stream_name": "goofi-stream"},
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def lsl_stream_stream_name_changed(self, value: str):
        assert value != "", "Stream name cannot be empty."
        # reinitialize the client
        self.setup()

    def setup(self):
        # stop the client if it is running
        if self.client_running:
            self.client.stop()

        # start the client
        self.client = MNE_LSLClient(host=self.params.lsl_stream.stream_name.value, wait_max=1, verbose=False)
        self.client.start()
        self.data_iter = self.client.iter_raw_buffers()

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        if not self.client_running or not hasattr(self, "data_iter"):
            return None

        # grab the next chunk of data
        data = next(self.data_iter)
        if data.size == 0:
            return None

        # TODO: return relevant metadata from self.client.info
        return {"out": (data, {})}

    @property
    def client_running(self):
        return hasattr(self, "client") and hasattr(self.client, "client")
