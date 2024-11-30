import base64
import io

import numpy as np
from PIL import Image

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Img2Txt(Node):
    @staticmethod
    def config_input_slots():
        return {"image": DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {"generated_text": DataType.STRING}

    @staticmethod
    def config_params():
        return {
            "img_to_text": {
                "model": StringParam(
                    "ollama:llama3.2-vision",
                    # options=["meta-llama/Llama-3.2-11B-Vision-Instruct", "gpt-4o-mini", "ollama"],
                    doc="Model ID or name for image captioning (Huggingface Llama, OpenAI, or Ollama)",
                ),
                "max_new_tokens": IntParam(30, 10, 1024, doc="Maximum number of tokens to generate"),
                "temperature": FloatParam(0.7, 0.1, 2.0, doc="Sampling temperature"),
                "prompt": StringParam("What is in this image?", doc="Prompt for image captioning"),
                "openai_key": StringParam("openai.key", doc="Path to OpenAI API key file (required for OpenAI models)"),
            }
        }

    def setup(self):
        import requests

        self.requests = requests

        self.processor = None
        self.model_instance = None
        self.openai = None
        self.ollama = None
        self.model_id = self.params.img_to_text.model.value

        if "/" in self.model_id.lower():
            self.setup_huggingface_llama()
        elif "gpt" in self.model_id.lower():
            self.setup_openai_gpt()
        elif "ollama" in self.model_id.lower():
            self.setup_ollama()
        else:
            raise ValueError(f"Invalid model selection: {self.model_id}")

    def setup_huggingface_llama(self):
        try:
            import torch
            from transformers import AutoProcessor, MllamaForConditionalGeneration

            self.model_instance = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            raise

    def setup_openai_gpt(self):
        try:
            import openai

            self.openai = openai
            key = self.params["img_to_text"]["openai_key"].value
            with open(key, "r") as f:
                self.openai.api_key = f.read().strip()
        except ImportError:
            print("Error: 'openai' library not found. Please install it using 'pip install openai'.")
            raise
        except FileNotFoundError:
            print(f"Error: OpenAI API key file not found at path: {key}")
            raise
        except Exception as e:
            print(f"Error initializing OpenAI: {e}")
            raise

    def setup_ollama(self):
        try:
            import ollama

            self.ollama = ollama
        except ImportError:
            print("Error: 'ollama' library not found. Please install it using 'pip install ollama'.")
            raise
        except ModuleNotFoundError:
            print("Error: 'ollama' library not found. Please install it using 'pip install ollama'.")
            raise

    def encode_image(self, image_array):
        if image_array.dtype != "uint8":
            image_array = (255 * (image_array - image_array.min()) / (image_array.max() - image_array.min())).astype("uint8")
        image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def process(self, image: Data):
        if image.data is None:
            return None

        image_array = image.data

        # Handle various image array shapes and types
        if image_array.ndim == 4:
            if image_array.shape[0] == 1:  # Remove batch dimension if present
                image_array = image_array.squeeze(0)
            else:
                raise ValueError(f"Unexpected image array shape: {image_array.shape}")

        if image_array.ndim == 3:
            if image_array.shape[2] == 3:  # RGB image
                pass
            elif image_array.shape[2] == 1:  # Grayscale image
                image_array = np.repeat(image_array, 3, axis=2)
            else:
                raise ValueError(f"Unexpected number of channels: {image_array.shape[2]}")
        elif image_array.ndim == 2:  # Grayscale image without channel dimension
            image_array = np.stack([image_array] * 3, axis=-1)
        else:
            raise ValueError(f"Unexpected image array dimensions: {image_array.ndim}")

        # Ensure the array is in uint8 format
        if image_array.dtype != np.uint8:
            if np.issubdtype(image_array.dtype, np.floating):
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        if "/" in self.model_id.lower():
            return self.process_huggingface_llama(image_array)
        elif "gpt" in self.model_id.lower():
            return self.process_openai_gpt(image_array)
        elif "ollama" in self.model_id.lower():
            return self.process_ollama(image_array)

    def process_huggingface_llama(self, image_array):
        if self.processor is None or self.model_instance is None:
            return {"generated_text": ("Error: Huggingface Llama model not initialized.", {})}

        # Process using Huggingface Llama
        base64_image = self.encode_image(image_array)
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": self.params.img_to_text.prompt.value}],
                }
            ],
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Wrap the image in a list
        inputs = self.processor(text=text, images=[base64_image], return_tensors="pt")

        # Move inputs to the correct device
        inputs = {k: v.to(self.model_instance.device) for k, v in inputs.items()}

        output = self.model_instance.generate(
            **inputs,
            max_new_tokens=self.params.img_to_text.max_new_tokens.value,
            temperature=self.params.img_to_text.temperature.value,
        )
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        return {"generated_text": (generated_text, {})}

    def process_openai_gpt(self, image_array):
        # Process using OpenAI GPT
        base64_image = self.encode_image(image_array)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.openai.api_key}"}
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.params.img_to_text.prompt.value},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": self.params.img_to_text.max_new_tokens.value,
            "temperature": self.params.img_to_text.temperature.value,
        }

        try:
            response = self.requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            generated_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except self.requests.exceptions.RequestException as e:
            print(f"Error during OpenAI captioning request: {e}")
            return {"generated_text": ("Error generating caption.", {})}

        return {"generated_text": (generated_text, {})}

    def process_ollama(self, image_array):
        # Process using Ollama
        base64_image = self.encode_image(image_array)
        messages = [{"role": "user", "content": self.params.img_to_text.prompt.value, "images": [base64_image]}]
        response = self.ollama.chat(
            model=self.model_id.replace("ollama:", ""),
            messages=messages,
            options={
                "num_predict": self.params.img_to_text.max_new_tokens.value,
                "temperature": self.params.img_to_text.temperature.value,
            },
        )
        generated_text = response["message"]["content"]

        return {"generated_text": (generated_text, {})}

    def img_to_text_model_changed(self, model_id):
        # reinitialize the model when the model_id changes
        if self.model_id != model_id:
            self.setup()
