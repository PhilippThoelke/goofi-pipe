import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class BufferString(Node):
    def config_input_slots():
        return {"val": DataType.STRING}

    def config_output_slots():
        return {"out": DataType.STRING}

    def config_params():
        return {
            "buffer": {
                "size": IntParam(10, 1, 5000),
                "separator": StringParam(" ", options=["[space]", ",", "[paragraph]"]),
                "reset": BoolParam(False, trigger=True)
            }
        }

    def setup(self):
        self.buffer = []

    def process(self, val: Data):
        if val is None or val.data is None:
            return None

        if self.params.buffer.reset.value:
            # reset buffer
            self.buffer = []

        maxlen = self.params.buffer.size.value
        separator = self.params.buffer.separator.value
        if separator == "[space]":
            separator = " "
        elif separator == "[paragraph]":
            separator = "\n"

        # Split the input string into words based on the separator
        words = val.data.split(separator)

        # Add words to the buffer
        self.buffer.extend(words)

        # Ensure the buffer does not exceed the maximum length
        if len(self.buffer) > maxlen:
            self.buffer = self.buffer[-maxlen:]

        # Join the buffer into a single string
        output_string = separator.join(self.buffer)

        return {"out": (output_string, val.meta)}