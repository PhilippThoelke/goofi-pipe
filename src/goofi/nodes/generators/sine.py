import time

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam


class Sine(Node):
    def config_params():
        return {"common": {"autotrigger": BoolParam(True)}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self):
        return {"out": (np.array([np.sin(time.time())]), {})}
