from typing import Any, Dict, Tuple

import cv2
import numpy as np
from mss import mss

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class VideoStream(Node):
    def config_params():
        return {
            "video_stream": {
                "device_index": IntParam(0, 0, 10),
                "mirror": BoolParam(True),
                "capture_mode": StringParam("screen", ["camera", "screen"]),
                "screen_region": StringParam("full", ["full", "custom"]),
                "screen_index": IntParam(1, 1, 10),
                "crop_top": IntParam(0, 0, 2000),
                "crop_down": IntParam(0, 0, 2000),
                "crop_left": IntParam(0, 0, 2000),
                "crop_right": IntParam(0, 0, 2000),
            },
            "common": {"autotrigger": BoolParam(True)},
        }

    def config_output_slots():
        return {"frame": DataType.ARRAY}

    def setup(self):
        self.cap = None
        self.capture_mode = self.params.video_stream.capture_mode.value
        if self.capture_mode == "camera":
            self.cap = cv2.VideoCapture(self.params.video_stream.device_index.value)

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        frame = None

        if self.capture_mode == "camera":
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.params.video_stream.mirror.value:
                    frame = cv2.flip(frame, 1)
        elif self.capture_mode == "screen":
            with mss() as sct:
                screen_index = self.params.video_stream.screen_index.value
                monitors = sct.monitors
                if screen_index >= len(monitors):
                    print(f"Invalid screen index: {screen_index}. Defaulting to screen 1.")
                    screen_index = 1
                monitor = monitors[screen_index]

                # Crop customization
                if self.params.video_stream.screen_region.value == "custom":
                    monitor = {
                        "top": monitor["top"] + self.params.video_stream.crop_top.value,
                        "left": monitor["left"] + self.params.video_stream.crop_left.value,
                        "width": monitor["width"]
                        - self.params.video_stream.crop_left.value
                        - self.params.video_stream.crop_right.value,
                        "height": monitor["height"]
                        - self.params.video_stream.crop_top.value
                        - self.params.video_stream.crop_down.value,
                    }

                screen_capture = sct.grab(monitor)
                frame = np.array(screen_capture)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        if frame is None:
            # Provide a fallback empty frame
            frame = np.zeros((1, 1, 3), dtype="float32")

        frame = frame.astype("float32") / 255.0

        return {"frame": (frame, {})}

    def video_stream_device_index_changed(self, value):
        if hasattr(self, "cap"):
            self.cap.release()
        self.cap = cv2.VideoCapture(value)

    def video_stream_capture_mode_changed(self, value):
        self.setup()
