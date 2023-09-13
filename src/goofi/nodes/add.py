from goofi.data import Data, DataType
from goofi.node import Node


class Add(Node):
    def config_input_slots():
        return {"a": DataType.ARRAY, "b": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self, a: Data, b: Data):
        if a is None or b is None:
            return None
        return {"out": (a.data + b.data, {})}
