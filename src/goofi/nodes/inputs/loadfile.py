from copy import deepcopy

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

        self.data_output = None
        self.string_output = None
        self.last_params = None

    def process(self):
        if self.last_params == self.params.file:
            # if the parameters are the same, return the previous output
            return {"data_output": self.data_output, "string_output": self.string_output}

        # if the parameters are different, load the file
        self.last_params = deepcopy(self.params.file)
        self.load_file()

        return {"data_output": self.data_output, "string_output": self.string_output}

    def load_file(self):
        if self.params.file.filename.value is None:
            self.data_output = None
            self.string_output = None
            return

        file_type = self.params.file.type.value
        filename = self.params.file.filename.value
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
            self.data_output = None
            self.string_output = ("\n".join(data), {})
            return

        if self.params.file.name_column.value:
            meta = {"channels": {"dim0": list(df.iloc[:, 0].values)}}
            data = data[:, 1:]

        data = data.astype(np.float32)

        # Handle time_series type
        if file_type == "time_series":
            assert data.shape[1] == 2, "Invalid time series shape"
            time_series, meta = data[0], data[1]
            assert isinstance(meta, dict), "Metadata should be a dictionary"

            self.data_output = (time_series, meta)
            self.string_output = None
            return

        # Handle spectrum type
        elif file_type == "spectrum":
            freq_multiplier = self.params["file"]["freq_multiplier"].value
            freq_vector = data[0] * freq_multiplier  # Multiply the frequency values
            spectrums = data[1]

            self.data_output = (np.array(spectrums), {"freq": freq_vector})
            self.string_output = None
            return

        elif file_type == "ndarray":
            self.data_output = (data, {})
            self.string_output = None
            return

        elif file_type == "embedding_csv":
            self.data_output = (data, meta if self.params.file.name_column.value else {})
            self.string_output = None
            return
