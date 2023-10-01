import openai
import threading
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam
from goofi.data import Data, DataType

class TextGeneration(Node):
    def config_input_slots():
        return {"prompt": DataType.STRING}

    def config_output_slots():
        return {"generated_text": DataType.STRING}

    def config_params():
        return {
            "text_generation": {
                "openai_key": StringParam("openai.key"),
                "model": StringParam("gpt-3.5-turbo"),
                "temperature": FloatParam(1.2, 0.0, 2.0),
                "max_tokens": IntParam(128, 10, 1024),
                "keep_conversation": False,  # Adding new parameter
            }
        }

    def setup(self):
        key = self.params["text_generation"]["openai_key"].value
        with open(key, "r") as f:
            openai.api_key = f.read().strip()
        self.messages = []  # Initialize conversation history

    def process(self, prompt: Data):
        if prompt.data is None:
            return None
        
        model = self.params["text_generation"]["model"].value
        temperature = self.params["text_generation"]["temperature"].value
        max_tokens = self.params["text_generation"]["max_tokens"].value
        keep_conversation = self.params["text_generation"]["keep_conversation"].value
        
        prompt_ = prompt.data
        
        # Modify the process according to the value of keep_conversation
        if keep_conversation:
            # Here you would need to manage previous messages to keep the conversation.
            user_message = {"role": "user", "content": prompt_}
            self.messages.append(user_message)
        else:
            self.messages = [{"role": 'user', "content": prompt_}]  # Reset conversation history

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return
        
        generated_text = response['choices'][0]["message"]["content"].strip()

        if keep_conversation:
            # Store the latest response to maintain conversation
            assistant_message = {"role": "assistant", "content": generated_text}
            self.messages.append(assistant_message)

        return {"generated_text": (generated_text, prompt.meta)}