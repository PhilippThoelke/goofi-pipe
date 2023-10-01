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
                "order": IntParam(1, 1, 10),  # added order parameter
                "f_low": FloatParam(1.0, 0.01, 9999.0),
                "f_high": FloatParam(60.0, 1.0, 10000.0),
                "ripple": FloatParam(1.0, 0.1, 10.0),
                "padding": FloatParam(0.1, 0.01, 1.0),
            }
        }

    def setup(self):
        self.filter_state = None

    def process(self, data: Data):
        # TO DO: deal with error when change in order
        if data is None or data.data is None:
            return None

        filter_type = self.params["filter"]["type"].value
        f_low = self.params["filter"]["f_low"].value
        f_high = self.params["filter"]["f_high"].value
        ripple = self.params["filter"]["ripple"].value
        padding = self.params["filter"]["padding"].value
        method = self.params["filter"]["method"].value
        order = self.params["filter"]["order"].value
        sfreq = data.meta["sfreq"]
        nyq = 0.5 * sfreq
        low = f_low / nyq
        high = f_high / nyq

        if filter_type == "butterworth":
            b, a = butter(order, [low, high], btype="band")
        elif filter_type == "chebyshev":
            b, a = cheby1(order, ripple, [low, high], btype="band")
        elif filter_type == "elliptic":
            b, a = ellip(order, ripple, ripple, [low, high], btype="band")

        # Edge Padding 10% of data

        if method == "Zero-phase":
            # Calculate the padlen as 10% of the signal length
            if data.data.ndim == 1:
                padlen = int(padding * len(data.data))
            else:  # if 2D, you will probably want to apply padding based on the time axis, usually the last axis.
                padlen = int(padding * data.data.shape[-1])
            filtered_data = filtfilt(b, a, data.data, padlen=padlen)

        if method == "Causal":
            n = max(len(a), len(b)) - 1  # Number of sections

            # Handle initialization for 1D data
            if data.data.ndim == 1:
                if self.filter_state is None or self.filter_state.ndim != 1:
                    self.filter_state = np.zeros(n)
                initial_condition = self.filter_state * data.data[0]
                filtered_data, self.filter_state = lfilter(b, a, data.data, zi=initial_condition)

            if data.data.ndim == 2:
                num_sections, num_channels = max(len(a), len(b)) - 1, data.data.shape[0]

                if self.filter_state is None or self.filter_state.ndim != 2 or self.filter_state.shape[1] != num_channels:
                    self.filter_state = np.zeros((num_sections, num_channels))

                initial_condition = (
                    self.filter_state * data.data[:, [0]]
                )  # Assuming data.data is (num_channels, num_time_points)
                filtered_data, self.filter_state = lfilter(b, a, data.data.T, axis=0, zi=initial_condition)
                filtered_data = filtered_data.T

        return {"filtered_data": (filtered_data, {**data.meta})}
