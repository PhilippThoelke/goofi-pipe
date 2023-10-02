from typing import Any, Dict, Tuple

import cv2

from goofi.data import DataType
from goofi.node import Node
from goofi.params import IntParam


class VideoStream(Node):
    def config_params():
        return {"video_stream": {"device_index": IntParam(0, 0, 10), "mirror": True}, "common": {"autotrigger": True}}

    def config_output_slots():
        return {"frame": DataType.ARRAY}

    def setup(self):
        self.cap = cv2.VideoCapture(self.params.video_stream.device_index.value)

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        _, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.params.video_stream.mirror.value:
            frame = cv2.flip(frame, 1)
        frame = frame.astype("float32") / 255.0
        return {"frame": (frame, {})}

    def video_stream_device_index_changed(self, value):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.params.video_stream.device_index.value)
