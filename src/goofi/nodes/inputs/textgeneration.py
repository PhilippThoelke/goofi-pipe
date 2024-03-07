from os import path

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class TextGeneration(Node):
    def config_input_slots():
        return {"prompt": DataType.STRING}

    def config_output_slots():
        return {"generated_text": DataType.STRING}

    def config_params():
        return {
            "text_generation": {
                "openai_key": StringParam("openai.key", doc="API key or path to a file containing it"),
                "model": StringParam("gpt-3.5-turbo"),
                "temperature": FloatParam(1.2, 0.0, 2.0),
                "max_tokens": IntParam(128, 10, 1024),
                "keep_conversation": False,
            }
        }

    def setup(self):
        self.client = None
        self.load_api_key()

        # initialize conversation history
        self.messages = []

    def process(self, prompt: Data):
        if prompt.data is None:
            return None

        model = self.params["text_generation"]["model"].value
        temperature = self.params["text_generation"]["temperature"].value
        max_tokens = self.params["text_generation"]["max_tokens"].value
        keep_conversation = self.params["text_generation"]["keep_conversation"].value

        prompt_ = prompt.data

        if keep_conversation:
            # add new message to conversation history
            user_message = {"role": "user", "content": prompt_}
            self.messages.append(user_message)
        else:
            # reset conversation history
            self.messages = [{"role": "user", "content": prompt_}]

        if self.client is None:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)

        # generate response
        response = self.client.chat.completions.create(
            model=model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generated_text = response.choices[0].message.content

        if keep_conversation:
            # store the latest response to maintain conversation
            assistant_message = {"role": "assistant", "content": generated_text}
            self.messages.append(assistant_message)

        return {"generated_text": (generated_text, prompt.meta)}

    def load_api_key(self):
        self.api_key = self.params["text_generation"]["openai_key"].value

        if path.exists(self.api_key):
            # load the API key from the file
            with open(self.api_key, "r") as f:
                self.api_key = f.read().strip()

    def text_generation_openai_key_changed(self):
        # reload the API key
        self.load_api_key()
        self.openai.api_key = self.api_key

        try:
            self.client.close()
        except Exception:
            pass

        # reset client
        self.client = None
