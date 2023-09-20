import threading
import time
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

    def client_thread(self):
        """Start the client and wait until running is set to False."""
        client = MNE_LSLClient(host=self.params.lsl_stream.stream_name.value, wait_max=0.5, verbose=False)
        try:
            client.start()
        except RuntimeError:
            self.running = False
            return

        if self.data_iterator is not None:
            raise RuntimeError("Data iterator already exists. The previous client was not stopped properly.")

        # grab the data iterator
        self.data_iterator = client.iter_raw_buffers()

        while self.running:
            time.sleep(0.1)

        client.stop()

    def setup(self, init: bool = True):
        """
        Initialize and start the client.

        ### Parameters
        `init`: bool
            Flag to indicate whether this is the first time the client is being initialized.
        """
        if init:
            self.running = False
            self.thread = None
            self.data_iterator = None
        # start the client
        self.start()

    def start(self):
        """Start a new client. If a client is already running, stop it first."""
        self.stop()

        # start the client
        self.running = True
        self.thread = threading.Thread(target=self.client_thread, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the client and wait for the thread to finish."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
            self.data_iterator = None

    def process(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """Fetch the next chunk of data from the client."""
        if not self.running or self.data_iterator is None:
            # client is not running, or not initialized yet
            return None

        # grab the next chunk of data
        data = next(self.data_iterator)
        if data.size == 0:
            return None

        # TODO: return relevant metadata from self.client.info
        return {"out": (data, {})}

    def lsl_stream_stream_name_changed(self, value: str) -> None:
        assert value != "", "Stream name cannot be empty."
        # reinitialize the client
        self.setup(init=False)

    def terminate(self) -> None:
        self.stop()
