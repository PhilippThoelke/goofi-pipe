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
        if self.params.operation.operation.value == "add":
            return {"out": (a.data + b.data, {})}
        elif self.params.operation.operation.value == "subtract":
            return {"out": (a.data - b.data, {})}
        elif self.params.operation.operation.value == "multiply":
            return {"out": (a.data * b.data, {})}
        elif self.params.operation.operation.value == "divide":
            return {"out": (a.data / b.data, {})}
        else:
            raise Error
