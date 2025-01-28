import time
import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class Normalization(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"normalized": DataType.ARRAY}

    def config_params():
        return {
            "normalization": {
                "method": StringParam("quantile", options=["quantile", "z"]),
                "n_seconds": IntParam(30, 1, 120),
                "reset": BoolParam(trigger=True),
            }
        }

    def setup(self):
        from scipy.stats import rankdata

        self.rankdata = rankdata
        self.window = []

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        val = np.asarray(data.data)
        if val.ndim > 2:
            print("Error: Normalization only accepts 1D or 2D arrays")
            return None

        # Handle reset trigger
        if self.params["normalization"]["reset"].value:
            self.window = []
            print("Reset triggered: clearing data window.")

        # Accumulate data in the moving window
        self.window.extend(val)
        window_size = self.params["normalization"]["n_seconds"].value
        self.window = self.window[-window_size:]  # Keep only the last `n_seconds` of data

        # Perform normalization if the window has data
        if len(self.window) > 0:
            if self.params["normalization"]["method"].value == "quantile":
                normalized_value = self.quantile_transform(val)
            elif self.params["normalization"]["method"].value == "z":
                normalized_value = self.zscore(val)
        else:
            print("No data in window: returning raw input.")
            normalized_value = val

        return {"normalized": (normalized_value, data.meta)}

    def zscore(self, val):
        mean = np.mean(self.window)
        std = np.std(self.window) + 1e-8
        return (val - mean) / std

    def quantile_transform(self, val):
        if val.ndim == 2:
            normalized = np.zeros_like(val)
            for i in range(val.shape[0]):
                normalized[i, :] = self._quantile_transform_1D(val[i, :])
            return normalized
        else:
            return self._quantile_transform_1D(val)

    def _quantile_transform_1D(self, arr):
        ranks = self.rankdata(arr)
        scaled_ranks = np.clip((ranks - 1) / (len(self.window) - 1), 0, 1)
        quantiles = np.percentile(self.window, 100 * scaled_ranks)
        return (arr - np.mean(quantiles)) / (np.std(quantiles) + 1e-8)
