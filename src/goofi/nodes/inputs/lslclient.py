import socket
from typing import Any, Dict, Tuple

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam


class LSLClient(Node):
    def config_params():
        return {
            "lsl_stream": {
                "source_name": "goofi",
                "stream_name": "",
                "refresh": BoolParam(False, trigger=True),
            },
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        """Initialize and start the LSL client."""
        import pylsl

        self.pylsl = pylsl

        if hasattr(self, "client"):
            self.disconnect()
        self.client = None
        self.ch_names = None

        # initialize list of streams
        self.available_streams = None
        self.lsl_stream_refresh_changed(True)

        self.connect()

    def process(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """Fetch the next chunk of data from the client."""
        if self.available_streams is None:
            self.lsl_stream_refresh_changed(True)

        if self.client is None:
            if not self.connect():
                return None

        try:
            # fetch data
            samples, _ = self.client.pull_chunk()
        except self.pylsl.LostError:
            self.setup()
            return None

        samples = np.array(samples).T

        if samples.size == 0:
            return None

        try:
            ch_info = self.client.info().desc().child("channels").child("channel")
            ch_type = self.client.info().type().lower()
            ch_names = []
            for k in range(1, self.client.info().channel_count() + 1):
                ch_names.append(ch_info.child_value("label") or "{} {:03d}".format(ch_type.upper(), k))
                ch_info = ch_info.next_sibling()
            self.ch_names = ch_names
        except Exception:
            self.setup()

        meta = {
            "sfreq": self.client.info().nominal_srate(),
            "channels": {"dim0": self.ch_names},
        }
        return {"out": (samples, meta)}

    def connect(self) -> bool:
        """Connect to the LSL stream."""
        self.lsl_stream_refresh_changed(True)
        self.disconnect()

        # find the stream
        source_name = self.params.lsl_stream.source_name.value
        stream_name = self.params.lsl_stream.stream_name.value

        matches = {}
        for info in self.available_streams:
            h, s, n = info.hostname(), info.source_id(), info.name()
            if s == source_name and (len(stream_name) == 0 or n == stream_name):
                if (s, n) in matches and h == socket.gethostname():
                    # prefer local streams
                    matches[(s, n)] = info
                elif (s, n) not in matches:
                    # otherwise, prefer the first match
                    matches[(s, n)] = info

        if len(matches) == 0:
            raise RuntimeError(f'Could not find stream "{stream_name}" from source "{source_name}".')
        elif len(matches) > 1:
            ms = {m[0]: m[1] for m in matches.keys()}
            raise RuntimeError(f'Found multiple streams matching "{stream_name}" from source "{source_name}": {ms}.')

        # connect to the stream
        self.client = self.pylsl.StreamInlet(info=list(matches.values())[0], recover=False)
        return True

    def disconnect(self) -> None:
        """Disconnect from the LSL stream."""
        if self.client is not None:
            self.client.close_stream()
            self.client = None

    def lsl_stream_refresh_changed(self, value: bool) -> None:
        self.available_streams = self.pylsl.resolve_streams()
        print("\nAvailable LSL streams:")
        for info in self.available_streams:
            print(f'  Source: "{info.source_id()}" with stream "{info.name()}" (hostname: {info.hostname()})')
        print()

    def lsl_stream_source_name_changed(self, value: str) -> None:
        assert value != "", "Host name cannot be empty."
        self.setup()

    def lsl_stream_stream_name_changed(self, value: str) -> None:
        assert value != "", "Stream name cannot be empty."
        self.setup()
