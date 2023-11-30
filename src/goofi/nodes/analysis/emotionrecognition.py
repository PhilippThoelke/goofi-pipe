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
            "face_box": DataType.ARRAY,
            "landmarks": DataType.ARRAY,
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
        if image is None:
            return {
                "emotion_probabilities": None,
                "face_box": None,
                "landmarks": None,
                "action_units": None
            }

        # Check if image data is of type uint8 and scale appropriately if not
        if image.data.dtype != np.uint8:
            #logging.info("Scaling and converting image data to uint8.")
            image_data = ((image.data - image.data.min()) * (255.0 / (image.data.max() - image.data.min()))).astype(np.uint8)
        else:
            image_data = image.data
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Use a temporary file, ensuring it is deleted after use
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            cv2.imwrite(temp_image.name, image_data)
        
        assert os.path.exists(temp_image.name), f"File {temp_image.name} does not exist."

        try:
            # Detect faces and their attributes using the file path
            predictions = self.detector.detect_image(temp_image.name)
            logging.info(f"Predictions: {predictions}")
        finally:
            # Make sure to remove the temporary file after detection
            os.remove(temp_image.name)

        if predictions.facebox is None:
            logging.warning("No face detected in the image.")
            # You may want to handle this case separately
            # For example, by returning default values or by raising an exception

        # Extract and convert outputs to the appropriate types
        emotion_probabilities = predictions.emotions.values.flatten().tolist() if predictions.emotions is not None else None
        face_box = predictions.facebox.values.flatten().tolist() if predictions.facebox is not None else None
        landmarks = predictions.landmarks.values.flatten().tolist() if predictions.landmarks is not None else None
        action_units = predictions.aus.values.flatten().tolist() if predictions.aus is not None else None

        return {
            "main_emotion": (self.emotion_names[np.argmax(emotion_probabilities)], {}),
            "emotion_probabilities": (np.array(emotion_probabilities), {}),
            "face_box": (np.array(face_box), {}),
            "landmarks": (np.array(landmarks),{}),
            "action_units": (np.array(action_units), {})
        }
