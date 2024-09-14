import base64
import io

import requests
from PIL import Image

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


class Img2Txt(Node):
    def config_input_slots():
        return {"image": DataType.ARRAY}

    def config_output_slots():
        return {"generated_text": DataType.STRING}

    def config_params():
        return {
            "img_to_text": {
                "openai_key": StringParam("openai.key"),
                "model": StringParam("gpt-4-vision-preview"),
                "max_tokens": IntParam(300, 10, 1024),
                "question": StringParam("What’s in this image?"),
            }
        }

    def setup(self):
        import openai

        self.openai = openai

        key = self.params["img_to_text"]["openai_key"].value
        with open(key, "r") as f:
            self.openai.api_key = f.read().strip()

    def encode_image(self, image_array):
        if image_array.dtype != "uint8":
            # Normalize the array to 0-255 and convert to uint8
            image_array = (255 * (image_array - image_array.min()) / (image_array.max() - image_array.min())).astype("uint8")

        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(image_array)
        buffered = io.BytesIO()

        # Save the image as JPEG to the buffer
        image.save(buffered, format="JPEG")
        buffered.seek(0)

        # Encode the buffered image to base64
        return base64.b64encode(buffered.read()).decode("utf-8")

    def process(self, image: Data):
        if image.data is None:
            return None

        # Assuming image.data is a NumPy array representing an image
        model = self.params["img_to_text"]["model"].value
        max_tokens = self.params["img_to_text"]["max_tokens"].value

        base64_image = self.encode_image(image.data)

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.openai.api_key}"}
        question = self.params["img_to_text"]["question"].value
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print("STATUS", response.status_code)
        print("RESPONSE", response.json())
        if response.status_code == 200:
            generated_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"generated_text": (generated_text, {})}
        else:
            return {"generated_text": ("Error In Generating Text", {})}
