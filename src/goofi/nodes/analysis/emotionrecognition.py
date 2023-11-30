from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam
from feat import Detector
import numpy as np
import cv2
import tempfile
import os
import logging


class EmotionRecognition(Node):
    def config_input_slots():
        return {"image": DataType.ARRAY}

    def config_output_slots():
        return {
            "emotion_probabilities": DataType.ARRAY,
            "action_units": DataType.ARRAY, 
            "main_emotion": DataType.STRING
        }

    #def config_params():
    #    return {
    #        "emotion_recognition": {
    #            "detector_config": StringParam("", doc="Configuration for py-feat detector"),
    #        }
    #    }

    def setup(self):
        logging.basicConfig(level=logging.INFO)
        # Load the py-feat detector with the specified models
        self.detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model='xgb',
            emotion_model="resmasknet",
            facepose_model="img2pose",
        )
        self.emotion_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

    def process(self, image: Data):
        if image is None or image.data is None:
            logging.warning("No image data to process.")
            return {
                "emotion_probabilities": None,
                "face_box": None,
                "landmarks": None,
                "action_units": None,
                "main_emotion": None
            }

        # Convert image data to the format expected by py-feat Detector
        if image.data.dtype != np.uint8:
            image_data = ((image.data - image.data.min()) * (255.0 / (image.data.max() - image.data.min()))).astype(np.uint8)
        else:
            image_data = image.data
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        # Detect faces and their attributes using the image data
        detected_faces = self.detector.detect_faces(image_data)
        if not detected_faces:
            logging.warning("No face detected in the image.")
            return {
                "emotion_probabilities": None,
                "action_units": None,
                "main_emotion": None
            }

        detected_landmarks = self.detector.detect_landmarks(image_data, detected_faces)
        
        # Detect action units
        if detected_landmarks:
            action_units = self.detector.detect_aus(image_data, detected_landmarks)
        else:
            action_units = None
            
        # Detect emotions
        emotions = self.detector.detect_emotions(image_data, detected_faces, detected_landmarks)

        # Process the emotions array
        if emotions is not None and len(emotions) > 0:
            emotion_probabilities = emotions[0]  # Assuming you're interested in the first detected face
            main_emotion_index = np.argmax(emotion_probabilities)
            main_emotion = self.emotion_names[main_emotion_index]
        else:
            emotion_probabilities = None
            main_emotion = None

        return {
            "emotion_probabilities": (np.array(emotion_probabilities), {}),
            "action_units": (np.array(action_units), {}),
            "main_emotion": (main_emotion, {})
        }
