import time

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class StaticBaseline(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"normalized": DataType.ARRAY}

    def config_params():
        return {
            "baseline": {
                "n_seconds": IntParam(30, 1, 120),
                "method": StringParam("quantile", options=["quantile", "mean"]),
                "baseline_computation": BoolParam(trigger=True),
            }
        }

    def setup(self):
        from scipy.stats import rankdata

        self.rankdata = rankdata

        self.window = []
        self.time_origin = None

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        val = np.asarray(data.data)
        if val.ndim > 2:
            print("Error: StaticBaseline only accepts 1D or 2D arrays")
            return None

        normalized_value = np.zeros_like(val)

        # If baseline computation is triggered, reset the window and set time_origin
        if self.params["baseline"]["baseline_computation"].value:
            self.window = []
            self.time_origin = time.time()

        if self.time_origin:
            elapsed_time = time.time() - self.time_origin

            if elapsed_time < self.params["baseline"]["n_seconds"].value:
                self.window.extend(val)
            else:
                self.time_origin = None  # Reset time_origin after accumulating for n_seconds

        # Only compute the Z-score when we have enough data in our window
        if len(self.window) >= self.params["baseline"]["n_seconds"].value:
            # Compute the Z-score using the desired method
            if self.params["baseline"]["method"].value == "quantile":
                normalized_value = self.quantile_transform(val)
            elif self.params["baseline"]["method"].value == "mean":
                normalized_value = self.zscore(val)
        else:
            normalized_value = val  # You can choose to normalize or just use raw data when the window is not full

        return {"normalized": (normalized_value, data.meta)}

    def zscore(self, val):
        if val.ndim == 1:
            mean = np.mean(self.window)
            std = np.std(self.window)
            return (val - mean) / (std + 1e-8)
        elif val.ndim == 2:
            normalized = np.zeros_like(val)
            for i in range(val.shape[0]):
                mean = np.mean(self.window[i])
                std = np.std(self.window[i])
                normalized[i, :] = (val[i, :] - mean) / (std + 1e-8)
            return normalized

    def quantile_transform(self, val):
        # Check for dimension
        if val.ndim == 2:
            normalized = np.zeros_like(val)
            for i in range(val.shape[0]):
                normalized[i, :] = self._quantile_transform_1D(val[i, :])
            return normalized
        else:
            return self._quantile_transform_1D(val)

    def _quantile_transform_1D(self, arr):
        # Convert data to ranks
        ranks = self.rankdata(arr)
        # Scale ranks to [0, 1]
        scaled_ranks = (ranks - 1) / (len(arr) - 1)
        # Calculate quantile values
        quantiles = np.percentile(arr, 100 * scaled_ranks)
        # Return normalized quantile values for the input data
        return (arr - np.mean(quantiles)) / (np.std(quantiles) + 1e-8)
