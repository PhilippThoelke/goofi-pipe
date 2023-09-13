import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam


class Constant(Node):
    def config_params():
        return {"common": {"autotrigger": BoolParam(True)}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self):
        return {"out": (np.ones(1), {})}
