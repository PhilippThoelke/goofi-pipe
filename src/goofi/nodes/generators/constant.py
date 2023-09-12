from goofi.connection import Connection
from typing import Dict

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Constant(Node):
    def __init__(self, connection: Connection) -> None:
        super().__init__(connection, autotrigger=True)
        self.register_output("out", DataType.ARRAY)

    def process(self) -> Dict[str, Data]:
        return {"out": Data(DataType.ARRAY, np.ones(1), {})}
