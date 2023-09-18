import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam


class Constant(Node):
    def config_params():
        return {"common": {"autotrigger": True}, "constant": {"value": FloatParam(0.0, -10.0, 10.0)}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self):
        return {"out": (np.ones(1) * self.params.constant.value.value, {})}
