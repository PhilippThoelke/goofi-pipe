import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam
from transformers import CLIPProcessor, CLIPModel
import cv2
from PIL import Image

class Embedding(Node):

    def config_input_slots():
        return {"text": DataType.STRING, "data": DataType.ARRAY}

    def config_output_slots():
        return {
            "text_embeddings": DataType.ARRAY,
            "data_embeddings": DataType.ARRAY,
        }

    def config_params():
        return {
            "embedding": {
                "model": StringParam(
                    "openai/clip-vit-base-patch32",
                    options=["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14","laion/CLIP-ViT-H-14-laion2B-s32B-b79K"],
                    doc="Model ID or name for embedding generation"
                )
            }
        }

    def setup(self):
        self.model_id = self.params["embedding"]["model"].value
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.model.eval()  # Set to evaluation mode
        self.model.to("cpu")  # Ensure model is on CPU
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def process(self, text: Data, data: Data):
        # Initialize outputs and metadata
        text_embeddings = None
        data_embeddings = None
        metadata = {}

        # Prepare text input and compute embeddings if text is provided
        if text is not None:
            input_text = text.data
            metadata["text"] = text.meta
            inputs_text = self.processor(text=input_text, return_tensors="pt", padding=True)
            outputs_text = self.model.get_text_features(**inputs_text)
            text_embeddings = outputs_text.detach().numpy()

        # Prepare image input and compute embeddings if data is provided
        if data is not None:
            input_data = data.data
            metadata["data"] = data.meta  # Preserve metadata

            # Check and preprocess numpy array
            if input_data.ndim == 4 and input_data.shape[0] == 1:  # Remove batch dimension if present
                input_data = input_data.squeeze(0)


            # Ensure image is resized and normalized
            if input_data.shape[:2] != (224, 224):  # Resize if not already 224x224
                input_data = cv2.resize(input_data, (224, 224))

            # Normalize pixel values only if necessary
            if input_data.max() > 1.0:  # Normalize to [0, 1] if data is in [0, 255]
                input_data = np.clip(input_data / 255.0, 0.0, 1.0)

            # Convert to PIL Image for CLIP compatibility
            input_data_pil = Image.fromarray((input_data * 255).astype(np.uint8))

            # Use CLIPProcessor for final preprocessing
            inputs_data = self.processor(images=input_data_pil, return_tensors="pt", padding=True)

            # Generate embeddings
            outputs_data = self.model.get_image_features(**inputs_data)
            data_embeddings = outputs_data.detach().numpy()

        # Return separate embeddings and a dictionary of metadata
        return {
            "text_embeddings": (text_embeddings, metadata),
            "data_embeddings": (data_embeddings, metadata),
        }

    def embedding_model_changed(self, _):
        """Reinitialize the stream."""
        self.setup()