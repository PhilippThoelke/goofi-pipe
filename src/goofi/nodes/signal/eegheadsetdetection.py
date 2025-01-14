import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam

class EEGHeadsetDetection(Node):
    def config_params():
        return {
            "threshold": {
                "average_value": FloatParam(500.0, 0.0, 10000.0),  # Threshold for average signal value
            },
        }

    def config_input_slots():
        return {"eeg_data": DataType.ARRAY}

    def config_output_slots():
        return {"headset_status": DataType.ARRAY}

    def process(self, eeg_data=None):
        if eeg_data is None or eeg_data.data.size == 0:
            return {"headset_status": (np.array(0), {})}  # No data, assume headset is not connected

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

        return {"headset_status": (headset_worn, {})}
