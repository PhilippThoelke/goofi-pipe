from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam
import numpy as np
import pandas as pd

class CardioRespiratoryVariability(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "MeanNN": DataType.ARRAY,
            "SDNN": DataType.ARRAY,
            "SDSD": DataType.ARRAY,
            "RMSSD": DataType.ARRAY,
            "pNN50": DataType.ARRAY,
            "LF": DataType.ARRAY,
            "HF": DataType.ARRAY,
            "LF/HF": DataType.ARRAY,
            "LZC": DataType.ARRAY,
            }

    def config_params():
        return {
            "cardiorespiratory": {
                "input_type": StringParam("ppg", options=["ppg", "ecg", "rsp"]),
            },
        }

    def setup(self):
        import neurokit2 as nk

        self.neurokit = nk

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        if self.params["cardiorespiratory"]["input_type"].value == "ppg":
            datatype = "HRV"
            _, info = self.neurokit.ppg_process(data.data, sampling_rate=data.meta["sfreq"])
            variability_df = self.neurokit.hrv(info, sampling_rate=data.meta["sfreq"])
        elif self.params["cardiorespiratory"]["input_type"].value == "ecg":
            datatype = "HRV"
            # extract peaks
            _, info = self.neurokit.ecg_peaks(data.data, sampling_rate=data.meta["sfreq"])
            # compute hrv
            variability_df = self.neurokit.hrv(info, sampling_rate=data.meta["sfreq"])
        elif self.params["cardiorespiratory"]["input_type"].value == "rsp":
            datatype = "RRV"
            _, info = self.neurokit.rsp_process(data.data, sampling_rate=data.meta["sfreq"])
            variability_df = self.neurokit.rsp_rrv(data.data, sampling_rate=data.meta["sfreq"])

        return {
            "MeanNN": (np.array(variability_df[f'{datatype}_MeanNN']), {}),
            "SDNN": (np.array(variability_df[f'{datatype}_SDNN']), {}),
            "SDSD": (np.array(variability_df[f'{datatype}_SDSD']), {}),
            "RMSSD": (np.array(variability_df[f'{datatype}_RMSSD']), {}),
            "pNN50": (np.array(variability_df[f'{datatype}_pNN50']), {}),
            "VLF": (np.array(variability_df[f'{datatype}_VLF']), {}),
            "LF": (np.array(variability_df[f'{datatype}_LF']), {}),
            "HF": (np.array(variability_df[f'{datatype}_HF']), {}),
            "LF/HF": (np.array(variability_df[f'{datatype}_LFHF']), {}),
            "LZC": (np.array(variability_df[f'{datatype}_LZC']), {}),
        }
