import numpy as np
from math import gcd
from scipy.signal import resample_poly
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class Resample(Node):
    def config_input_slots():
        # Defining one input slot for one input signal
        return {"data": DataType.ARRAY}

    def config_output_slots():
        # Defining one output slot for the resampled signal
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "resample": {
                "new_sfreq": IntParam(1000, 10, 44100),  # New sampling frequency as a parameter
            }
        }

    

    def process(self, data: Data):
        if data is None or data.data is None:
            print("Data is None")
            return None

        # Original sampling frequency from metadata
        sf = data.meta["sfreq"]

        # Retrieve new sampling frequency parameter
        new_sfreq = self.params.resample.new_sfreq.value

        # Log the values for debugging
        #print(f"Original sfreq: {sf}, New sfreq: {new_sfreq}")

        # Calculate up and down factors based on the gcd of sf and new_sfreq
        factor = gcd(int(sf), int(new_sfreq))
        up = new_sfreq // factor
        down = sf // factor

        # Log the up and down values for debugging
        #print(f"Up: {up}, Down: {down}")

        signal = np.array(data.data)

        # Check if the signal contains any NaN or inf values
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print("Signal contains NaN or inf values")

        # Resample the signal
        resampled_signal = resample_poly(signal, up, down, padtype="line")

        # Update the 'sfreq' metadata
        data.meta["sfreq"] = new_sfreq

        return {"out": (resampled_signal, data.meta)}


