from scipy.signal import welch
from numpy.fft import fft, fftfreq
from goofi.params import IntParam, FloatParam, StringParam
from goofi.data import Data, DataType
from goofi.node import Node
import numpy as np
import neurokit2 as nk


class CardiacRespiration(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"cardiac": DataType.ARRAY}

    def config_params():
        return {
            "cardiac": {
                "input_type": StringParam("ppg", options=["ppg", "ecg"]),
            }
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None
        
        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        if self.params["cardiac"]["input_type"].value == "ppg":
            print(data.data)
            print(data.meta["sfreq"])
            signal, info = nk.ppg_process(data.data, sampling_rate=data.meta["sfreq"])
            hrv_df = nk.hrv(info, sampling_rate=data.meta["sfreq"])
            
        elif self.params["cardiac"]["input_type"].value == "ecg":
            # Extract peaks
            rpeaks, info = nk.ecg_peaks(data.data, sampling_rate=data.meta["sfreq"])
            # Compute rate
            ecg_rate = nk.ecg_rate(rpeaks, sampling_rate=data.meta['sfreq'], desired_length=len(data.data))
            edr = nk.ecg_rsp(ecg_rate, sampling_rate=data.meta["sfreq"])
            print(edr)      
        
        # get array of all the features
        
        return {"cardiac": (edr, data.meta)}
