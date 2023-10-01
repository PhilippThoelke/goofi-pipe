import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam


class ConstantArray(Node):
    def config_params():
        return {
            "constant": {"value": FloatParam(1.0, -10.0, 10.0), "shape": "1"},
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self):
        parts = [p for p in self.params.constant.shape.value.split(",") if len(p) > 0]
        shape = list(map(int, parts))
        return {"out": (np.ones(shape) * self.params.constant.value.value, {"sfreq": self.params.common.max_frequency.value})}
