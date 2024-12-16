from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


class Audio2Txt(Node):
    @staticmethod
    def config_input_slots():
        return {"audio": DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {"generated_text": DataType.STRING}

    @staticmethod
    def config_params():
        return {
            "audio_to_text": {
                "model": StringParam(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    doc="Huggingface model ID for audio captioning",
                ),
                "max_new_tokens": IntParam(30, 10, 1024, doc="Maximum number of tokens to generate"),
                "prompt": StringParam("What is in this audio?", doc="Prompt for audio captioning"),
            }
        }

    def setup(self):
        import librosa
        import torch

        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install transformers to use Qwen2-Audio models via pip install transformers")

        self.librosa = librosa
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.params["audio_to_text"]["model"].value, torch_dtype=torch.float16
            )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.params["audio_to_text"]["model"].value,
                device_map="cuda",
                torch_dtype=torch.float16,
            )
        except Exception as e:
            print(f"Error initializing Huggingface model: {e}")
            raise

    def process(self, audio: Data):
        if audio.data is None:
            return None

        audio_array = audio.data
        if audio_array.ndim != 1:
            raise ValueError(f"Unexpected audio array dimensions: {audio_array.shape}")

        # Ensure the audio is sampled at the model's expected rate
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio_array = self.librosa.resample(audio_array, orig_sr=audio.meta["sfreq"], target_sr=sampling_rate)

        max_new_tokens = self.params["audio_to_text"]["max_new_tokens"].value
        prompt = self.params["audio_to_text"]["prompt"].value

        # Prepare the conversation input
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "embedded"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(
            text=text,
            audios=[audio_array],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate text
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1) :]

        generated_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {"generated_text": (generated_text, {})}
