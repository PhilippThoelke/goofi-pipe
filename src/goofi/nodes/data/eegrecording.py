import logging
from typing import Any, Dict, Tuple

import mne
from mne.datasets import eegbci
from mne_realtime import MockLSLStream

from goofi.node import Node

logger = logging.getLogger(__name__)


class EEGRecording(Node):
    def config_params():
        return {"recording": {"use_example_data": True, "file_path": "", "stream_name": "goofi-stream"}}

    def setup(self):
        """Load the data and start the stream."""
        if self.params.recording.use_example_data.value:
            if self.params.recording.file_path.value != "":
                logger.warning("Both 'use_example_data' and 'file_path' are set. Using example data.")

            # load example data
            raw = mne.concatenate_raws([mne.io.read_raw(p) for p in eegbci.load_data(1, [1, 2])])
            eegbci.standardize(raw)
        elif self.params.recording.file_path.value == "":
            # either use example data or a file path must be set
            raise ValueError("File path cannot be empty if 'Use Example Data' is False.")
        else:
            # load data from file
            raw = mne.io.read_raw(self.params.recording.file_path.value, preload=True)

        assert self.params.recording.stream_name.value != "", "Stream name cannot be empty."

        # stop the stream if it is running
        if hasattr(self, "stream"):
            self.stream.stop()
        # start the stream
        self.stream = MockLSLStream(self.params.recording.stream_name.value, raw, "eeg")
        self.stream.start()

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """Should never run, as the stream runs on its own."""
        raise NotImplementedError

    def recording_use_example_data_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def recording_file_path_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def recording_stream_name_changed(self, _):
        """Reinitialize the stream."""
        self.setup()
