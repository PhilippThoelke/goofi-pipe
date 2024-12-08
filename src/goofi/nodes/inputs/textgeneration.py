import json
from os import environ, path

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


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
                "model": StringParam("gpt-4o-mini", doc="Model ID for the text generation service"),
                "system_prompt": StringParam("", doc="System prompt for text generation"),
                "temperature": FloatParam(1.0, 0.0, 2.0, doc="Temperature for text generation"),
                "max_tokens": IntParam(20, 5, 2048, doc="Maximum number of tokens to generate"),
                "keep_conversation": BoolParam(False, doc="Whether to keep conversation history"),
                "save_conversation": StringParam("", doc="Whether to save conversation history to a JSON file"),
            }
        }

    def setup(self):
        import requests

        self.requests = requests

        self.client = None
        self.previous_model = None
        self.api_key_loaded = False
        self.messages = []
        self.load_api_key()
        self.import_libs(self.params["text_generation"]["model"].value)
        self.system_prompt = self.params["text_generation"]["system_prompt"].value

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

        if not self.api_key and not model.startswith("local-"):
            raise ValueError(f"API key for {model} not found in environment variables or input parameters.")

        print(f"Loaded API key for {model}.")

    def import_libs(self, model):
        if model != self.previous_model:
            self.openai, self.anthropic, self.genai, self.ollama = self.dynamic_import_libs(model)  # Include ollama
            self.previous_model = model
            self.client = None

    def dynamic_import_libs(self, model):
        if model.startswith("gpt-"):
            try:
                import openai
            except ImportError:
                raise ImportError("You need to install openai: pip install openai")
            return openai, None, None, None
        elif model.startswith("claude-"):
            try:
                import anthropic
            except ImportError:
                raise ImportError("You need to install anthropic: pip install anthropic")
            return None, anthropic, None, None
        elif model.startswith("gemini-"):
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("You need to install google-generativeai: pip install google-generativeai\nWith Python>=3.10")
            return None, None, genai, None
        elif model.startswith("ollama-"):
            try:
                import ollama
            except ImportError:
                raise ImportError("You need to install ollama: pip install ollama")
            return None, None, None, ollama  # Return ollama
        else:
            raise ValueError(f"Unknown model: {model}")

    def api_key_changed(self):
        self.load_api_key()
        self.client = None
        self.api_key_loaded = True

    def generate_openai_response(self, messages, temp):
        if self.client is None:
            self.client = self.openai.OpenAI(api_key=self.api_key)

        payload = {
            "model": self.params["text_generation"]["model"].value,
            "messages": messages,
            "temperature": temp,
            "max_tokens": self.params["text_generation"]["max_tokens"].value,
        }

        if self.system_prompt is not None and len(messages) < 2:
            payload["messages"] = [{"role": "system", "content": self.system_prompt}] + messages

        response = self.client.chat.completions.create(**payload)
        return response.choices[0].message.content

    def generate_anthropic_response(self, messages, temp):
        if self.client is None:
            self.client = self.anthropic.Anthropic(api_key=self.api_key)

        response = self.client.messages.create(
            model=self.params["text_generation"]["model"].value,
            messages=messages,
            max_tokens=self.params["text_generation"]["max_tokens"].value,
            temperature=temp,
            system=self.system_prompt if self.system_prompt is not None else None,
        )

        return response.content[0].text

    def generate_gemini_response(self, text, temp, keep_conversation, history=None):
        if self.client is None:
            self.genai.configure(api_key=self.api_key)
            # If system prompt is provided, prepend it to the message
            self.client = self.genai.GenerativeModel(
                model_name=self.params["text_generation"]["model"].value,
                system_instruction=self.system_prompt if self.system_prompt is not None else "You are a helpful assistant.",
            )

        if not keep_conversation:
            history = []

        chat = self.client.start_chat(history=history)

        generation_config = self.genai.GenerationConfig(
            max_output_tokens=self.params["text_generation"]["max_tokens"].value, temperature=temp
        )

        # Prepare the message
        message = text

        # Send the request
        response = chat.send_message(message, generation_config=generation_config)
        # NOTE:  double check if history works this way
        try:
            history = chat.history()
        except Exception as e:
            print(f"History is None: {e}")
            history = None

        return response.text

    def generate_ollama_response(self, model, messages, temp):
        # Create an Ollama client instance
        if self.client is None:
            self.client = self.ollama

        response = self.client.chat(
            model=str(model.replace("ollama-", "")),  # remove ollama- prefix
            messages=messages,
            options={
                "system": self.system_prompt,
                "temperature": temp,
                "max_tokens": self.params["text_generation"]["max_tokens"].value,
            },
        )
        return response["message"]["content"]

    def generate_local_response(self, content):
        url = "http://127.0.0.1:5000/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "mode": "chat-instruct",  # TODO: Add mode to params
            "max_tokens": self.params["text_generation"]["max_tokens"].value,
            "temperature": self.params["text_generation"]["temperature"].value,
            "character": "Example",
            "messages": [{"role": "user", "content": content}],
        }
        response = self.requests.post(url, headers=headers, json=data, verify=False)
        return response.json()["choices"][0]["message"]["content"]

    def save_conversation_to_json(self):
        filename = f"conversation_{self.params['text_generation']['save_conversation'].value}.json"
        with open(filename, "w") as f:
            json.dump(self.messages, f, indent=4)
        print(f"Conversation saved to {filename}")

    def process(self, prompt: Data):
        if prompt is None or prompt.data is None:
            return None

        self.import_libs(self.params["text_generation"]["model"].value)

        prompt_ = prompt.data
        temp = self.params["text_generation"]["temperature"].value
        model = self.params["text_generation"]["model"].value
        keep_conversation = self.params["text_generation"]["keep_conversation"].value
        save_conversation = self.params["text_generation"]["save_conversation"].value

        if keep_conversation:
            self.messages.append({"role": "user", "content": prompt_})
        else:
            self.messages = [{"role": "user", "content": prompt_}]

        if model.startswith("gpt-"):
            if self.api_key_loaded:
                pass
            else:
                self.load_api_key()  # Ensure API key is loaded with the current model
            generated_text = self.generate_openai_response(self.messages, temp)

        elif model.startswith("claude-"):
            if self.api_key_loaded:
                pass
            else:
                self.load_api_key()
            generated_text = self.generate_anthropic_response(self.messages, temp)

        elif model.startswith("gemini-"):
            if self.api_key_loaded:
                pass
            else:
                self.load_api_key()
            generated_text = self.generate_gemini_response(prompt_, temp, keep_conversation)

        elif model.startswith("local-"):
            generated_text = self.generate_local_response(prompt_)

        elif model.startswith("ollama-"):
            generated_text = self.generate_ollama_response(model, self.messages, temp)
        else:
            raise ValueError(f"Unknown model: {model}")

        if keep_conversation:
            self.messages.append({"role": "assistant", "content": generated_text})

        if save_conversation:
            self.save_conversation_to_json()

        if self.params["text_generation"]["api_key"].value != self.api_key:
            self.api_key_changed()

        return {"generated_text": (generated_text, prompt.meta)}
