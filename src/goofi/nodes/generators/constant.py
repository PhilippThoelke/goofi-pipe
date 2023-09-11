from multiprocessing.connection import Connection, PipeConnection
from typing import Dict, Union

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Constant(Node):
    def __init__(self, connection: Union[Connection, PipeConnection]) -> None:
        super().__init__(connection, autotrigger=True)
        self.register_output("out", DataType.FLOAT_1D)

    def process(self) -> Dict[str, Data]:
        return {"out": Data(DataType.FLOAT_1D, np.ones(1), {})}
