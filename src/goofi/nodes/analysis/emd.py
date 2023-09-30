import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, FloatParam
from biotuner.peaks_extraction import EMD_eeg

class EMD(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"IMFs": DataType.ARRAY}

    def config_params():
        return {"EMD": {"nIMFs": IntParam(5, 1, 10)}}
                             

    def process(self, data: Data):
        if data is None:
            return None

        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")
        # add indices for each IMF in the meta data as strings
        data.meta["dim0"] = ["IMF"+str(i) for i in range(self.params['EMD']['nIMFs'].value+1)]
        IMFs = EMD_eeg(data.data, method="EMD_fast", graph=False, extrema_detection="simple", nIMFs=5)
        IMFs = IMFs[0:self.params['EMD']['nIMFs'].value+1]

        return {"IMFs": (IMFs, data.meta)}