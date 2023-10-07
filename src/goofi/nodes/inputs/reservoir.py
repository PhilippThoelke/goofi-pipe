import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Reservoir(Node):
    def config_params():
        return {"reservoir":{"size": IntParam(100,2,1000), "function": StringParam("tanh", options=["tanh", "sigmoid"]),"add_bias": True,"mean":FloatParam(0.0, -.2,.2),"std":FloatParam(1.0, 0.7, 1.3),},"common":{"autotrigger":True}}

    def config_output_slots():
        return {"data": DataType.ARRAY}

    def setup(self):
        self.reservoir = np.zeros(self.params.reservoir.size.value)
        self.weights = np.random.uniform(-1, 1, (self.params.reservoir.size.value, self.params.reservoir.size.value))
        self.bias = np.random.uniform(-1, 1, self.params.reservoir.size.value)

    def process(self):
        w = self.weights*self.params.reservoir.std.value+self.params.reservoir.mean.value
        self.reservoir = np.dot(w, self.reservoir)
        if self.params.reservoir.add_bias.value:
            self.reservoir += self.bias

        if self.params.reservoir.function.value == "tanh":
            self.reservoir = np.tanh(self.reservoir)
        elif self.params.reservoir.function.value == "sigmoid":
            self.reservoir = 1 / (1 + np.exp(-self.reservoir))
        else:
            raise ValueError(f"Unsupported function {self.params.reservoir.function.value} for reservoir output.")
        
        # reshape into matrix of size greatest common divisor of reservoir size
        return {"data":(self.reservoir, {"sfreq":self.params.common.max_frequency.value})}
        
    def reservoir_size_changed(self, value):
        self.setup()