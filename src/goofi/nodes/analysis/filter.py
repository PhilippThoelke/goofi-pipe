from scipy.signal import butter, cheby1, ellip, filtfilt, lfilter, lfilter_zi
import numpy as np
from goofi.node import Node
from goofi.params import StringParam, FloatParam
from goofi.data import DataType, Data

class Filter(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"filtered_data": DataType.ARRAY}

    def config_params():
        return {
            "filter": {
                "type": StringParam("butterworth", options=["butterworth", "chebyshev", "elliptic"]),
                "method": StringParam("Causal", options=["Causal", "Zero-phase"]),
                "f_low": FloatParam(1.0, 0.01, 9999.0),
                "f_high": FloatParam(60.0, 1.0, 10000.0),
                "ripple": FloatParam(1.0, 0.1, 10.0),
                "padding": FloatParam(0.1, 0.01, 1.0)  
            }
        }
    def setup(self):
        self.filter_state = None
    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        filter_type = self.params["filter"]["type"].value
        f_low = self.params["filter"]["f_low"].value
        f_high = self.params["filter"]["f_high"].value
        ripple = self.params["filter"]["ripple"].value
        padding = self.params["filter"]["padding"].value
        method = self.params["filter"]["method"].value
        sfreq = data.meta["sfreq"]
        nyq = 0.5 * sfreq
        low = f_low / nyq
        high = f_high / nyq

        if filter_type == "butterworth":
            b, a = butter(1, [low, high], btype='band')
        elif filter_type == "chebyshev":
            b, a = cheby1(1, ripple, [low, high], btype='band')
        elif filter_type == "elliptic":
            b, a = ellip(1, ripple, ripple, [low, high], btype='band')
        

        # Edge Padding 10% of data
        
        if method == "Zero-phase":
            # Calculate the padlen as 10% of the signal length
            if data.data.ndim == 1:
                padlen = int(padding * len(data.data))
            else:  # if 2D, you will probably want to apply padding based on the time axis, usually the last axis.
                padlen = int(padding * data.data.shape[-1])
            filtered_data = filtfilt(b, a, data.data, padlen=padlen)
        
        if method == "Causal":
            if self.filter_state is None:
                if data.data.ndim == 1:
                    self.filter_state = lfilter_zi(b, a)
                else:
                    # Reshape zi to match the number of channels in data.data
                    self.filter_state = np.repeat(lfilter_zi(b, a)[..., np.newaxis], data.data.shape[-2], axis=-1)  

            # Apply causal filter and update filter state
            filtered_data, self.filter_state = lfilter(b, a, data.data, axis=-1, zi=self.filter_state * data.data[..., 0, np.newaxis])

        return {"filtered_data": (filtered_data, {**data.meta})}
