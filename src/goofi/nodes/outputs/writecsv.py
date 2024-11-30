import datetime
import os

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class WriteCsv(Node):
    @staticmethod
    def config_input_slots():
        # This node will accept only a table as its input.
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
        self.base_filename = None  # To track the filename without the timestamp
        self.written_files = set()  # Track files we've written headers to

    def process(self, table_input: Data):
        # Check the value of write_control.
        # If False, return without doing anything.
        if not self.params["Write"]["write"].value:
            return

        table_data = table_input.data
        # Extract the actual data content
        actual_data = {key: (value.data if isinstance(value, Data) else value) for key, value in table_data.items()}

        # Assume there's only one key (e.g., 'value1') as per your example
        if len(actual_data) != 1:
            raise ValueError("Expected exactly one column in the table input.")

        column_name, column_data = next(iter(actual_data.items()))

        # Function to flatten the data
        def flatten(data):
            if isinstance(data, (list, tuple, np.ndarray)):
                # Check if it's multi-dimensional
                if isinstance(data, np.ndarray):
                    if data.ndim > 1:
                        return data.flatten().tolist()
                    else:
                        return data.tolist()
                else:
                    # For lists or tuples, attempt to flatten only one level
                    flattened = []
                    for item in data:
                        if isinstance(item, (list, tuple, np.ndarray)):
                            flattened.extend(item)
                        else:
                            flattened.append(item)
                    return flattened
            else:
                # If data is a single scalar value
                return [data]

        # Flatten the column data
        flattened_data = flatten(column_data)

        # Create DataFrame with the flattened data
        df = self.pd.DataFrame({column_name: flattened_data})

        # Get the filename from the parameters
        filename = self.params["Write"]["filename"].value

        # Check if the filename has changed.
        if filename != self.base_filename:
            basename, ext = os.path.splitext(filename)
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"{basename}_{datetime_str}{ext}"
            self.last_filename = fn  # Update the last used filename with timestamp
            self.base_filename = filename  # Update the base filename
        else:
            fn = self.last_filename

        # Determine if headers need to be written
        write_header = fn not in self.written_files

        # Write the DataFrame to CSV
        df.to_csv(fn, mode="a", header=write_header, index=False)

        if write_header:
            self.written_files.add(fn)

        # Optionally, you can return the path to the generated CSV file as an output
        # return fn
