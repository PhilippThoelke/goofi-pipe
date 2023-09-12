from goofi.connection import Connection

from goofi.data import Data, DataType
from goofi.node import Node


class Add(Node):
    def __init__(self, connection: Connection) -> None:
        super().__init__(connection)

        self.register_input("a", DataType.ARRAY)
        self.register_input("b", DataType.ARRAY)
        self.register_output("out", DataType.ARRAY)

    def process(self, a: Data, b: Data) -> Data:
        if a is None or b is None:
            return None
        return {"out": Data(DataType.ARRAY, a.data + b.data, {})}
