from typing import Dict, List, Tuple

from goofi.constants import DataType
from goofi.node import BaseNode
from goofi.pipes import Data, InputPipe, OutputPipe


class Add(BaseNode):
    def setup(self) -> Tuple[List[InputPipe], List[OutputPipe]]:
        return (
            [InputPipe("a", DataType.FLOAT_1D), InputPipe("b", DataType.FLOAT_1D)],
            [OutputPipe("sum", DataType.FLOAT_1D)],
        )

    def update(self, a: Data, b: Data) -> Dict[str, Data]:
        return {"sum": Data(DataType.FLOAT_1D, a.data + b.data)}
