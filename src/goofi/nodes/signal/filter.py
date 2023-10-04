from scipy.signal import butter, cheby1, ellip, filtfilt, lfilter, lfilter_zi
import numpy as np
from goofi.node import Node
from goofi.params import StringParam, FloatParam, IntParam
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
                "mode": StringParam("bandpass", options=["bandpass", "notch"]),
                "order": IntParam(1, 1, 10),  # added order parameter
                "f_low": FloatParam(1.0, 0.01, 9999.0),
                "f_high": FloatParam(60.0, 1.0, 10000.0),
                "ripple": FloatParam(1.0, 0.1, 10.0),
                "padding": FloatParam(0.1, 0.01, 1.0),
                "Q": FloatParam(10.0, 0.1, 100.0),  # Quality factor
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
        order = self.params["filter"]["order"].value
        mode = self.params["filter"]["mode"].value
        Q = self.params["filter"]["Q"].value
        sfreq = data.meta["sfreq"]
        nyq = 0.5 * sfreq
        low = f_low / nyq
        high = f_high / nyq

        if mode == "bandpass":
            if filter_type == "butterworth":
                b, a = butter(order, [low, high], btype="band")
            elif filter_type == "chebyshev":
                b, a = cheby1(order, ripple, [low, high], btype="band")
            elif filter_type == "elliptic":
                b, a = ellip(order, ripple, ripple, [low, high], btype="band")

        elif mode == "notch":
            f0 = (f_high + f_low) / 2
            w0 = f0 / nyq
            bw = f0 / Q
            bw /= nyq
            
            # Ensure that the frequencies are within bounds
            w_low = np.clip(w0 - bw/2, 0.01, 0.99)  # avoiding exact 0 or 1 for stability
            w_high = np.clip(w0 + bw/2, 0.01, 0.99)
            
            if filter_type == "butterworth":
                b, a = butter(order, [w_low, w_high], btype='bandstop')
            elif filter_type == "chebyshev":
                b, a = cheby1(order, ripple, [w_low, w_high], btype='bandstop')
            elif filter_type == "elliptic":
                b, a = ellip(order, ripple, ripple, [w_low, w_high], btype='bandstop')



        if method == "Zero-phase":
            # Calculate the padlen as 10% of the signal length
            if data.data.ndim == 1:
                padlen = int(padding * len(data.data))
            else:  # if 2D, you will probably want to apply padding based on the time axis, usually the last axis.
                padlen = int(padding * data.data.shape[-1])
            filtered_data = filtfilt(b, a, data.data, padlen=padlen)

        if method == "Causal":
            # Handle initialization for 1D data
            if data.data.ndim == 1:
                zi = lfilter_zi(b, a)
                if self.filter_state is None or self.filter_state.shape != zi.shape:
                    self.filter_state = zi * data.data[0]
                filtered_data, self.filter_state = lfilter(b, a, data.data, zi=self.filter_state)

            if data.data.ndim == 2:
                num_channels = data.data.shape[0]
                zi = lfilter_zi(b, a).reshape(-1, 1)
                
                if self.filter_state is None or self.filter_state.shape[1] != num_channels:
                    self.filter_state = np.tile(zi, (1, num_channels)) * data.data[:, 0]

                filtered_data, self.filter_state = lfilter(b, a, data.data.T, axis=0, zi=self.filter_state)
                filtered_data = filtered_data.T


        return {"filtered_data": (filtered_data, {**data.meta})}
