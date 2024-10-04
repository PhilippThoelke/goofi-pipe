import os
import threading
import time
from typing import Any, Dict, Tuple

import mne
import pandas as pd
from mne.datasets import eegbci
from mne_lsl.player import PlayerLSL

from goofi.node import Node


class EEGRecording(Node):

    def config_params():
        return {
            "recording": {
                "use_example_data": True,
                "file_path": "",
                "source_name": "goofi",
                "stream_name": "recording",
            }
        }

    def stream_thread(self):
        """Load the appropriate data and start the stream. Then wait until running is set to False."""
        while not self.params.recording.use_example_data.value and not os.path.exists(self.params.recording.file_path.value):
            print("File path cannot be empty if 'Use Example Data' is False.")
            time.sleep(1)

        if self.params.recording.use_example_data.value:
            raw = mne.concatenate_raws(
                [mne.io.read_raw(p, preload=True, verbose=False) for p in eegbci.load_data(1, [1, 2])],
                verbose=False,
            )
            eegbci.standardize(raw)
            # scale the data for better default behavior
            raw.apply_function(lambda x: x * 1e4)
        elif self.params.recording.file_path.value.endswith(".csv"):
            # load data from csv file
            df = pd.read_csv(self.params.recording.file_path.value, index_col=0)
            df = df.select_dtypes(include=["float"])
            data = df.transpose().to_numpy()

            # TODO: make sfreq a parameter
            info = mne.create_info(ch_names=df.columns.tolist(), ch_types=["eeg"] * data.shape[0], sfreq=256)
            raw = mne.io.RawArray(data, info)
        else:
            # load data from file
            raw = mne.io.read_raw(self.params.recording.file_path.value, preload=True)

        # start the stream
        stream = PlayerLSL(
            raw,
            name=self.params.recording.stream_name.value,
            source_id=self.params.recording.source_name.value,
            annotations=False,
        )
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
                # TODO: add proper logging
                print("Both 'use_example_data' and 'file_path' are set. Using example data.")

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

    def terminate(self):
        """Stop the stream and terminate the node."""
        self.stop()
