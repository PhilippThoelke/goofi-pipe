import datetime
import os
import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam, BoolParam, FloatParam
import json
import time

class WriteCsv(Node):
    @staticmethod
    def config_input_slots():
        # This node will accept a table as its input.
        return {"table_input": DataType.TABLE,
                "start": DataType.ARRAY,
                "stop": DataType.ARRAY,}

    @staticmethod
    def config_params():
        # Parameters can include the CSV filename, write control, and timestamp option.
        return {
            "Write": {
                "filename": StringParam("output.csv"),
                "start": BoolParam(False, trigger=True),
                "stop": BoolParam(False, trigger=True),
                'duration': FloatParam(0.0, 0.0, 100.0),
                "timestamps": BoolParam(False),  # New timestamp parameter
            },
        }

    def setup(self):
        import pandas as pd

        self.pd = pd
        self.last_filename = None
        self.base_filename = None  # Track the base filename without timestamp
        self.written_files = set()  # Track files to ensure headers are written
        self.last_values = {}  # Store the last known value for each column
        self.is_writing = False

    def process(self, table_input: Data, start: Data, stop: Data):
        if start is not None and (start.data > 0).any() or self.params["Write"]["start"].value:
            self.is_writing = True
            self.start_time = time.time()
            # Generate a new filename each time writing starts
            filename = self.params["Write"]["filename"].value
            basename, ext = os.path.splitext(filename)
            datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.last_filename = f"{basename}_{datetime_str}{ext}"
        if stop is not None or self.params["Write"]["stop"].value:
            self.is_writing = False
        
        self.input_slots["start"].clear()
        self.input_slots["stop"].clear()
        duration = self.params["Write"]["duration"].value
        if self.is_writing:
            table_data = table_input.data
            if duration > 0:
                if time.time() - self.start_time > duration:
                    self.is_writing = False
            # Extract actual data content, handling multiple columns
            actual_data = {
                key: (value.data if isinstance(value, Data) else value)
                for key, value in table_data.items()
            }

            def flatten(data):
                """Ensure lists and NumPy arrays are stored as JSON strings to keep their structure."""
                if isinstance(data, np.ndarray):
                    return json.dumps(data.tolist())  # Convert ndarray to list before serializing
                elif isinstance(data, (list, tuple)):
                    return json.dumps(data)  # Serialize list as a JSON string
                return data  # Return scalars as-is

            flattened_data = {
                col: [flatten(values)] if not isinstance(values, list) 
                else [flatten(v) for v in values]
                for col, values in actual_data.items()
            }

            # Ensure all columns have the same length by padding with None
            max_length = max(map(len, flattened_data.values()), default=0)
            for col in flattened_data:
                flattened_data[col] += [None] * (max_length - len(flattened_data[col]))

            # Replace None with the last known value
            for col in flattened_data:
                if col not in self.last_values:
                    self.last_values[col] = None  # Initialize with None if not present
                for i in range(len(flattened_data[col])):
                    if flattened_data[col][i] is None:
                        flattened_data[col][i] = self.last_values[col]
                    else:
                        self.last_values[col] = flattened_data[col][i]  # Update the last known value

            # Add timestamp column if enabled
            if self.params["Write"]["timestamps"].value:
                timestamps = [datetime.datetime.utcnow().isoformat()] * max_length
                flattened_data["timestamp"] = timestamps

            # Convert to DataFrame
            df = self.pd.DataFrame(flattened_data)

            # Get the filename from parameters
            filename = self.params["Write"]["filename"].value

            # Check if filename has changed, then update with timestamp

            fn = self.last_filename

            # Determine if headers should be written
            write_header = fn not in self.written_files

            # Append new data to CSV
            df.to_csv(fn, mode="a", header=write_header, index=False)

            # Mark file as written to prevent duplicate headers
            if write_header:
                self.written_files.add(fn)
