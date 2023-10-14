from os import path

import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from goofi.data import Data, DataType
from goofi.node import Node


class PoseEstimation(Node):
    def config_input_slots():
        return {"image": DataType.ARRAY}

    def config_output_slots():
        return {"pose": DataType.ARRAY}

    def setup(self):
        model_path = path.join(self.assets_path, "hand_landmarker.task")
        if not path.exists(model_path):
            url = (
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            response = requests.get(url)
            # ensure the request was successful
            response.raise_for_status()
            # write the content to a file
            with open(model_path, "wb") as file:
                file.write(response.content)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, image: Data):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=(image.data * 255).astype(np.uint8))
        detection_result = self.detector.detect(mp_image)

        if len(detection_result.hand_landmarks) == 0:
            return None

        landmarks = np.array([[l.x, l.y, l.z] for l in detection_result.hand_landmarks[0]]).T
        meta = {
            "handdedness": detection_result.handedness[0][0].display_name,
            "channels": {"dim0": ["x", "y", "z"], "dim1": LANDMARK_NAMES},
        }
        return {"pose": (landmarks, meta)}


LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]
