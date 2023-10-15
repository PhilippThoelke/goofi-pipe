from goofi.params import StringParam
from goofi.data import Data, DataType
from goofi.node import Node
import numpy as np


class PowerBandEEG(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "delta": DataType.ARRAY,
            "theta": DataType.ARRAY,
            "alpha": DataType.ARRAY,
            "lowbeta": DataType.ARRAY,
            "highbeta": DataType.ARRAY,
            "gamma": DataType.ARRAY,
        }

    def config_params():
        return {
            "powerband": {
                "power_type": StringParam("absolute", options=["absolute", "relative"]),
            }
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None
        bands = {
            "delta": (1, 3),
            "theta": (3, 7),
            "alpha": (7, 12),
            "lowbeta": (12, 20),
            "highbeta": (20, 30),
            "gamma": (30, 50),
        }
        power_type = self.params["powerband"]["power_type"].value
        if data.data.ndim == 1:
            freqs = np.array(data.meta["channels"]["dim0"])
            if freqs[0] == 0:
                freqs[0] = 1e-8
            del data.meta["channels"]["dim0"]
        elif data.data.ndim == 2:
            freqs = np.array(data.meta["channels"]["dim1"])
            if freqs[0] == 0:
                freqs[0] = 1e-8
            del data.meta["channels"]["dim1"]

        output = {}
        for band, (f_min, f_max) in bands.items():
            valid_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
            if data.data.ndim == 1:
                selected_psd = data.data[valid_indices]
            else:  # if 2D
                selected_psd = data.data[:, valid_indices]

            # Computing the power
            power = np.sum(selected_psd, axis=-1)
            if power_type == "relative":
                total_power = np.sum(data.data, axis=-1)
                power = power / total_power

            output[band] = (np.array(power), {"freq_min": f_min, "freq_max": f_max, **data.meta})
        return output
