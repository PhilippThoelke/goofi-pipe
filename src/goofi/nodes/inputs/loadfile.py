import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class LoadFile(Node):
    def config_output_slots():
        return {"data_output": DataType.ARRAY}

    def config_params():
        return {
            "file": {
                "filename": StringParam("earth", doc="The name of the file to load with extension"),
                "type": StringParam(
                    "spectrum",
                    options=["spectrum", "time_series", "ndarray"],
                    doc="Type of file to load, choose between 'ndarray', 'time_series' or 'spectrum'",
                ),
                "freq_multiplier": FloatParam(1.0, doc="Multiplier to adjust the frequency values"),
            },
            "common": {"autotrigger": True},
        }

    def setup(self):
        import pandas as pd

        self.pd = pd

        self.time_series = None
        self.meta = None
        self.freq_vector = None

    def process(self):
        if self.params.file.filename.value is None:
            return None
        # asset_path = self.assets_path
        file_type = self.params["file"]["type"].value
        filename = self.params["file"]["filename"].value
        extension = filename.split(".")[-1]
        if extension == "npy":
            data = np.load(f"{filename}", allow_pickle=True)
        elif extension == "txt":
            data = np.loadtxt(f"{filename}")
        elif extension == "csv":
            data = self.pd.read_csv(f"{filename}", header=None).values
        if data.shape[0] > data.shape[1]:
            data = data.T

        # Handle time_series type
        if file_type == "time_series":
            if self.time_series is None:
                assert data.shape[1] == 2, "Invalid time series shape"
                self.time_series, self.meta = data[0], data[1]
                assert isinstance(self.meta, dict), "Metadata should be a dictionary"

            return {"data_output": (self.time_series, self.meta)}

        # Handle spectrum type
        elif file_type == "spectrum":
            freq_multiplier = self.params["file"]["freq_multiplier"].value
            freq_vector = data[0] * freq_multiplier  # Multiply the frequency values
            spectrums = data[1]

            return {"data_output": (np.array(spectrums), {"freq": freq_vector})}

        elif file_type == "ndarray":
            return {"data_output": (data, {})}

        return None
