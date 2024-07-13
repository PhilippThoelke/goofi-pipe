from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam


class StringAwait(Node):
    def config_input_slots():
        return {"message": DataType.STRING, "trigger": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.STRING}

    def config_params():
        return {
            "string_await": {
                "require_change": BoolParam(True, doc="Only output when the message changes, and we have an unconsumed trigger")
            }
        }

    def setup(self):
        self.last_message = None

    def process(self, message: Data, trigger: Data):
        if trigger is None or message is None:
            return

        if self.params.string_await.require_change.value and self.last_message == message.data:
            return

        self.input_slots["trigger"].clear()

        self.last_message = message.data
        return {"out": (message.data, message.meta)}
