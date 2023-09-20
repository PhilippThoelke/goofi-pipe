import time

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam


class Sine(Node):
    def config_params():
        return {
            "sine": {"frequency": FloatParam(1.0, 0.1, 30.0)},
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self):
        return {"out": (np.array([np.sin(time.time() * np.pi * 2 * self.params.sine.frequency.value)]), {})}
