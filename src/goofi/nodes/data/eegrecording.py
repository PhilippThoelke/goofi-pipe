import logging
import threading
import time
from typing import Any, Dict, Tuple

import mne
from mne.datasets import eegbci
from mne_realtime import MockLSLStream

from goofi.node import Node

logger = logging.getLogger(__name__)


class EEGRecording(Node):
    def config_params():
        return {"recording": {"use_example_data": True, "file_path": "", "stream_name": "goofi-stream"}}

    def stream_thread(self):
        """Load the appropriate data and start the stream. Then wait until running is set to False."""
        if self.params.recording.use_example_data.value:
            raw = mne.concatenate_raws([mne.io.read_raw(p) for p in eegbci.load_data(1, [1, 2])])
            eegbci.standardize(raw)
        else:
            # load data from file
            raw = mne.io.read_raw(self.params.recording.file_path.value, preload=True)

        stream = MockLSLStream(self.params.recording.stream_name.value, raw, "eeg")
        stream.start()

        while self.running:
            time.sleep(0.1)

        stream.stop()

    def setup(self, init: bool = True):
        """
        Load the data and start the stream.

        ### Parameters
        `init`: bool
            Flag to indicate whether this is the first time the stream is being initialized.
        """
        if init:
            self.running = True
            self.thread = None
        else:
            # stop previous stream if it exists
            self.stop()

        if self.params.recording.use_example_data.value:
            if self.params.recording.file_path.value != "":
                # both use_example_data and file_path are set
                logger.warning("Both 'use_example_data' and 'file_path' are set. Using example data.")
        elif self.params.recording.file_path.value == "":
            # either use example data or a file path must be set
            raise ValueError("File path cannot be empty if 'Use Example Data' is False.")
        assert self.params.recording.stream_name.value != "", "Stream name cannot be empty."

        # start the stream
        self.start()

    def start(self):
        """Start a new stream. If a stream is already running, stop it first."""
        self.stop()

        # start the stream
        self.running = True
        self.thread = threading.Thread(target=self.stream_thread, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the stream and wait for the thread to finish."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """Should never run, as the stream runs on its own."""
        raise NotImplementedError

    def recording_use_example_data_changed(self, _):
        """Reinitialize the stream."""
        self.setup(init=False)

    def recording_file_path_changed(self, _):
        """Reinitialize the stream."""
        self.setup(init=False)

    def recording_stream_name_changed(self, _):
        """Reinitialize the stream."""
        self.setup(init=False)
