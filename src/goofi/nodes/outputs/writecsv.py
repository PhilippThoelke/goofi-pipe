import datetime
import os

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class WriteCsv(Node):
    @staticmethod
    def config_input_slots():
        # This node will accept a table as its input.
        return {"table_input": DataType.TABLE}

    @staticmethod
    def config_params():
        # Parameters can include the CSV filename and the write control.
        return {
            "Write": {"filename": StringParam("output.csv"), "write": False},
        }

    def setup(self):
        import pandas as pd

        self.pd = pd
        self.last_filename = None
        self.base_filename = None  # Track the base filename without timestamp
        self.written_files = set()  # Track files to ensure headers are written

    def process(self, table_input: Data):
        # Check if writing is enabled
        if not self.params["Write"]["write"].value:
            return

        table_data = table_input.data
        
        # Extract actual data content, handling multiple columns
        actual_data = {
            key: (value.data if isinstance(value, Data) else value)
            for key, value in table_data.items()
        }

        # Function to flatten data
        def flatten(data):
            if isinstance(data, (list, tuple, np.ndarray)):
                if isinstance(data, np.ndarray) and data.ndim > 1:
                    return data.flatten().tolist()
                return [item for sublist in data for item in (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]
            return [data]

        # Flatten all columns
        flattened_data = {col: flatten(values) for col, values in actual_data.items()}

        # Ensure all columns have the same length by padding with None
        max_length = max(map(len, flattened_data.values()))
        for col in flattened_data:
            flattened_data[col] += [None] * (max_length - len(flattened_data[col]))

        # Convert to DataFrame
        df = self.pd.DataFrame(flattened_data)

        # Get the filename from parameters
        filename = self.params["Write"]["filename"].value

        # Check if filename has changed, then update with timestamp
        if filename != self.base_filename:
            basename, ext = os.path.splitext(filename)
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"{basename}_{datetime_str}{ext}"
            self.last_filename = fn
            self.base_filename = filename
        else:
            fn = self.last_filename

        # Determine if headers should be written
        write_header = fn not in self.written_files

        # Append new data to CSV
        df.to_csv(fn, mode="a", header=write_header, index=False)

        # Mark file as written to prevent duplicate headers
        if write_header:
            self.written_files.add(fn)
