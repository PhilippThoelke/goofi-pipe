import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class CardioRespiratoryVariability(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "Mean": DataType.ARRAY,
            "SDNN": DataType.ARRAY,
            "SDSD": DataType.ARRAY,
            "RMSSD": DataType.ARRAY,
            "pNN50": DataType.ARRAY,
            "VLF": DataType.ARRAY,
            "LF": DataType.ARRAY,
            "HF": DataType.ARRAY,
            "LF/HF": DataType.ARRAY,
            "LZC": DataType.ARRAY,
            "Peaks": DataType.ARRAY,
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

        cleaned = None
        if self.params["cardiorespiratory"]["input_type"].value == "ppg":
            datatype = "HRV"
            ppg, info = self.neurokit.ppg_process(data.data, sampling_rate=data.meta["sfreq"])
            variability_df = self.neurokit.hrv(info, sampling_rate=data.meta["sfreq"])
            cleaned = ppg["PPG_Peaks"].values
        elif self.params["cardiorespiratory"]["input_type"].value == "ecg":
            datatype = "HRV"
            # extract peaks
            ecg, info = self.neurokit.ecg_peaks(data.data, sampling_rate=data.meta["sfreq"])
            # compute hrv
            variability_df = self.neurokit.hrv(info, sampling_rate=data.meta["sfreq"])
            cleaned = ecg["ECG_Peaks"].values
        elif self.params["cardiorespiratory"]["input_type"].value == "rsp":
            datatype = "RRV"
            rsp, info = self.neurokit.rsp_process(data.data, sampling_rate=data.meta["sfreq"])
            variability_df = self.neurokit.rsp_rrv(data.data, sampling_rate=data.meta["sfreq"])
            cleaned = rsp["RSP_Peaks"].values

        BBorNN = "BB" if datatype == "RRV" else "NN"
        pNNorBBx = "pBBx" if datatype == "RRV" else "pNN50"

        return {
            "Mean": (np.array(variability_df[f"{datatype}_Mean{BBorNN}"]), {}),
            "SDNN": (np.array(variability_df[f"{datatype}_SD{BBorNN}"]), {}),
            "SDSD": (np.array(variability_df[f"{datatype}_SDSD"]), {}),
            "RMSSD": (np.array(variability_df[f"{datatype}_RMSSD"]), {}),
            "pNN50": (np.array(variability_df[f"{datatype}_{pNNorBBx}"]), {}),
            "VLF": (np.array(variability_df[f"{datatype}_VLF"]), {}),
            "LF": (np.array(variability_df[f"{datatype}_LF"]), {}),
            "HF": (np.array(variability_df[f"{datatype}_HF"]), {}),
            "LF/HF": (np.array(variability_df[f"{datatype}_LFHF"]), {}),
            "LZC": (np.array(variability_df[f"{datatype}_LZC"]), {} if datatype == "HRV" else None),
            "Peaks": (cleaned, data.meta),
        }
