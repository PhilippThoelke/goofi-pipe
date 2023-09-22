import numpy as np
from mne import pick_channels

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, FloatParam, StringParam
from scipy.ndimage import gaussian_filter1d

class Smooth(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"smooth": {"sigma": FloatParam(2.0, 0.1, 20.0),
                           "axis": IntParam(-1, 0, 2)}}

    def process(self, data: Data):
        if data is None:
            return None

        axis = self.params.smooth.axis.value
        sigma = self.params.smooth.sigma.value
        
        smoothed_sig = gaussian_filter1d(data.data, sigma, axis=axis)


        return {"out": (smoothed_sig, data.meta)}
