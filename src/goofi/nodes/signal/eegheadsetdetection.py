import numpy as np

from goofi.data import DataType, Data
from goofi.node import Node
from goofi.params import FloatParam
import time

class EEGHeadsetDetection(Node):
    def config_params():
        return {
            "threshold": {"average_value": FloatParam(500.0, 0.0, 10000.0, doc="Threshold for average signal value")},
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {"eeg_data": DataType.ARRAY}

    def config_output_slots():
        return {"headset_status": DataType.ARRAY}

    def process(self, eeg_data: Data):

        if not hasattr(self, 'no_data_count'):
            self.no_data_count = 0

        if eeg_data is None or eeg_data.data.size == 0:
            self.no_data_count += 1
            if self.no_data_count >= 100:
                return {"headset_status": (np.array(0), {})}  # No data for 100 times, assume headset is not connected
        else:
            self.no_data_count = 0

        # Extract EEG data and parameters
        signal = eeg_data.data
        avg_value_threshold = self.params.threshold.average_value.value

        # Compute the average of the absolute values
        average_value = np.mean(np.abs(signal))

        # Detection logic based on average value threshold
        if average_value > avg_value_threshold:
            headset_worn = np.array(1)  # Not worn (average too high)
        else:
            headset_worn = np.array(2)  # Worn (average below threshold)

        # handle the case where the EEG LSL cuts out
        self.input_slots["eeg_data"].clear()

        return {"headset_status": (headset_worn, {})}
