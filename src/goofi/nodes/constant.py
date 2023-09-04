from typing import Dict, List, Tuple

import numpy as np

from goofi.constants import DataType
from goofi.node import BaseNode
from goofi.pipes import Data, InputPipe, OutputPipe


class Constant(BaseNode):
    def setup(self) -> Tuple[List[InputPipe], List[OutputPipe]]:
        return ([], [OutputPipe("val", DataType.FLOAT_1D)])

    def update(self) -> Dict[str, Data]:
        return {"val": Data(DataType.FLOAT_1D, np.array([1.0]))}
