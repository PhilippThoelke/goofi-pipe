import time

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam


class Sine(Node):
    def config_params():
        return {
            "sine": {"frequency": FloatParam(1.0, 0.1, 30.0), "sampling_frequency": FloatParam(1000.0, 1.0, 1000.0)},
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        self.last_trigger = time.time()

    def process(self):
        meta = {"sfreq": self.params.sine.sampling_frequency.value}

        t = time.time()
        dt = t - self.last_trigger
        xs = np.arange(t, t + dt, 1 / self.params.sine.sampling_frequency.value)
        data = np.sin(xs * np.pi * 2 * self.params.sine.frequency.value)

        self.last_trigger = t
        return {"out": (data, meta)}
