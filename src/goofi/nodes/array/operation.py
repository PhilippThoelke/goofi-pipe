from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Operation(Node):
    def config_input_slots():
        return {"a": DataType.ARRAY, "b": DataType.ARRAY}

    def config_params():
        return {"operation":{"operation": StringParam("add", options=["add", "sutract", "multiply", "divide"])}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self, a: Data, b: Data):
        if a is None or b is None:
            return None
        if self.params.operation == "add":
            out = a.data + b.data
        if self.params.operation == "subtract":
            out = a.data - b.data
        if self.params.operation == "multiply":
            out = a.data * b.data
        if self.params.operation == "divide":
            out = a.data / b.data
        return {"out": (out, {})}
