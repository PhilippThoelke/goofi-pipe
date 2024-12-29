import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class LoadFile(Node):
    def config_output_slots():
        return {"data_output": DataType.ARRAY, "string_output": DataType.STRING}

    def config_params():
        return {
            "file": {
                "filename": StringParam("earth", doc="The name of the file to load with extension"),
                "type": StringParam(
                    "spectrum", options=["spectrum", "time_series", "ndarray", "embedding_csv"], doc="Type of file to load"
                ),
                "freq_multiplier": FloatParam(1.0, doc="Multiplier to adjust the frequency values"),
                "header": 0,
                "index_column": True,
                "name_column": False,
                "select": StringParam("", doc="NumPy selection string"),
            },
            "common": {"autotrigger": True, "max_frequency": 1.0},
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

        df = None
        if extension == "npy":
            data = np.load(f"{filename}", allow_pickle=True)
        elif extension == "txt":
            data = np.loadtxt(f"{filename}")
        elif extension == "csv":
            df = self.pd.read_csv(
                f"{filename}",
                header=self.params.file.header.value,
                index_col=0 if self.params.file.index_column.value else None,
            )
            data = df.values

        # apply selection
        selection = self.params.file.select.value
        dtypes = None
        if selection:
            try:
                # Parse the selection string into a tuple of slices
                slices = tuple(
                    (
                        slice(*map(lambda x: int(x.strip()) if x.strip() else None, dim.split(":")))
                        if ":" in dim
                        else int(dim.strip())
                    )
                    for dim in selection.split(",")
                )
                data = data[slices]
                if df is not None:
                    dtypes = list(df.dtypes)[slices[1]]
                    if not isinstance(dtypes, list):
                        dtypes = [dtypes]
            except (ValueError, IndexError) as e:
                print(f"Invalid selection string: {selection}")

        if dtypes is not None and any([dtype == "object" for dtype in dtypes]):
            return {"data_output": None, "string_output": ("\n".join(data), {})}

        if self.params.file.name_column.value:
            self.meta = {"channels": {"dim0": list(df.iloc[:, 0].values)}}
            data = data[:, 1:]

        data = data.astype(np.float32)

        # Handle time_series type
        if file_type == "time_series":
            if self.time_series is None:
                assert data.shape[1] == 2, "Invalid time series shape"
                self.time_series, self.meta = data[0], data[1]
                assert isinstance(self.meta, dict), "Metadata should be a dictionary"

            return {"data_output": (self.time_series, self.meta), "string_output": None}

        # Handle spectrum type
        elif file_type == "spectrum":
            freq_multiplier = self.params["file"]["freq_multiplier"].value
            freq_vector = data[0] * freq_multiplier  # Multiply the frequency values
            spectrums = data[1]

            return {"data_output": (np.array(spectrums), {"freq": freq_vector}), "string_output": None}

        elif file_type == "ndarray":
            return {"data_output": (data, {}), "string_output": None}

        elif file_type == "embedding_csv":
            return {"data_output": (data, self.meta if self.params.file.name_column.value else {}), "string_output": None}

        return None
