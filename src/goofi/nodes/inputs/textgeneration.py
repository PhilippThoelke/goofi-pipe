from os import path, environ

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam, BoolParam

class TextGeneration(Node):
    @staticmethod
    def config_input_slots():
        return {"prompt": DataType.STRING}

    @staticmethod
    def config_output_slots():
        return {"generated_text": DataType.STRING}

    @staticmethod
    def config_params():
        return {
            "text_generation": {
                "api_key": StringParam("openai.key", doc="API key for the text generation service"),
                "model": StringParam("gpt-3.5-turbo", doc="Model ID for the text generation service"),
                "temperature": FloatParam(1.0, 0.0, 2.0, doc="Temperature for text generation"),
                "max_tokens": IntParam(20, 5, 2048, doc="Maximum number of tokens to generate"),
                "keep_conversation": BoolParam(False, doc="Whether to keep conversation history")
            }
        }

    def setup(self):
        self.client = None
        self.previous_model = None
        self.messages = []
        self.load_api_key()
        self.import_libs(self.params["text_generation"]["model"].value)

    def load_api_key(self):
        self.api_key = self.params["text_generation"]["api_key"].value
        model = self.params["text_generation"]["model"].value

        if path.exists(self.api_key):
            with open(self.api_key, "r") as f:
                self.api_key = f.read().strip()
        elif model.startswith("gpt-"):
            self.api_key = environ.get("OPENAI_API_KEY", self.api_key)
        elif model.startswith("claude-"):
            self.api_key = environ.get("ANTHROPIC_API_KEY", self.api_key)
        elif model.startswith("gemini-"):
            self.api_key = environ.get("GOOGLE_API_KEY", self.api_key)

        if not self.api_key:
            raise ValueError(f"API key for {model} not found in environment variables or input parameters.")

        print(f"Loaded API key for {model}.")

    def import_libs(self, model):
        if model != self.previous_model:
            self.openai, self.anthropic, self.genai = self.dynamic_import_libs(model)
            self.previous_model = model
            self.client = None

    def dynamic_import_libs(self, model):
        if model.startswith("gpt-"):
            try:
                import openai
            except ImportError:
                raise ImportError("You need to install openai: pip install openai")
            return openai, None, None
        elif model.startswith("claude-"):
            try:
                import anthropic
            except ImportError:
                raise ImportError("You need to install anthropic: pip install anthropic")
            return None, anthropic, None
        elif model.startswith("gemini-"):
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("You need to install google-generativeai: pip install google-generativeai")
            return None, None, genai
        else:
            raise ValueError(f"Unknown model: {model}")

    def api_key_changed(self):
        self.load_api_key()
        self.client = None

    def generate_openai_response(self, messages, temp):
        if self.client is None:
            self.client = self.openai.OpenAI(api_key=self.api_key)
        response = self.client.chat.completions.create(
            model=self.params["text_generation"]["model"].value,
            messages=messages,
            temperature=temp,
            max_tokens=self.params["text_generation"]["max_tokens"].value,
        )
        return response.choices[0].message.content

    def generate_anthropic_response(self, messages, temp):
        if self.client is None:
            self.client = self.anthropic.Anthropic(api_key=self.api_key)
        response = self.client.messages.create(
            model=self.params["text_generation"]["model"].value,
            messages=messages,
            max_tokens=self.params["text_generation"]["max_tokens"].value,
            temperature=temp
        )
        return response.content[0].text

    def generate_gemini_response(self, text, temp, keep_conversation):
        if self.client is None:
            self.genai.configure(api_key=self.api_key)
            self.client = self.genai.GenerativeModel(self.params['text_generation']['model'].value)
        if not keep_conversation:
            history = []
        chat = self.client.start_chat(history=history)
        config = {
            "max_output_tokens": self.params["text_generation"]["max_tokens"].value,
            "temperature": temp,
        }
        response = chat.send_message(text, generation_config=config)
        return response.text

    def process(self, prompt: Data):
        if prompt.data is None:
            return None

        self.import_libs(self.params["text_generation"]["model"].value)
        self.load_api_key()  # Ensure API key is loaded with the current model

        prompt_ = prompt.data
        temp = self.params["text_generation"]["temperature"].value
        model = self.params["text_generation"]["model"].value
        keep_conversation = self.params["text_generation"]["keep_conversation"].value

        if keep_conversation:
            self.messages.append({"role": "user", "content": prompt_})
        else:
            self.messages = [{"role": "user", "content": prompt_}]

        if model.startswith("gpt-"):
            generated_text = self.generate_openai_response(self.messages, temp)
        elif model.startswith("claude-"):
            generated_text = self.generate_anthropic_response(self.messages, temp)
        elif model.startswith("gemini-"):
            generated_text = self.generate_gemini_response(prompt_, temp, keep_conversation)
        else:
            raise ValueError(f"Unknown model: {model}")

        if keep_conversation:
            self.messages.append({"role": "assistant", "content": generated_text})

        return {"generated_text": (generated_text, prompt.meta)}
