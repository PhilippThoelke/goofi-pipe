import os
import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam, BoolParam


class Replay(Node):
    @staticmethod
    def config_output_slots():
        return {"table_output": DataType.TABLE}

    @staticmethod
    def config_params():
        return {
            "Read": {
                "filename": StringParam("output.csv"),
                "play": BoolParam(False),
                "restart": BoolParam(False, trigger=True),
            }
        }

    def setup(self):
        self.df = None
        self.current_index = 0
        self.last_filename = None  # Track filename changes
        self.load_csv()

    def load_csv(self):
        import pandas as pd
        
        filename = self.params["Read"]["filename"].value
        if filename != self.last_filename:  # Only reload if filename changed
            if os.path.exists(filename):
                self.df = pd.read_csv(filename)
                self.current_index = 0
            else:
                self.df = None
            self.last_filename = filename

    def process(self):
        filename = self.params["Read"]["filename"].value

        # Reload CSV if filename has changed
        self.load_csv()

        if self.df is None or self.df.empty:
            return {"table_output": ({}, {})}  # Return empty table instead of None

        if not self.params["Read"]["play"].value:
            return

        # Extract the current row as a dictionary
        row_data = self.df.iloc[self.current_index].to_dict()

        # Convert each value to the appropriate Data format
        table_output = {
            key: Data(
                DataType.ARRAY if not isinstance(value, str) else DataType.STRING,
                np.array([value]) if isinstance(value, (int, float)) else value,  # Convert lists, ints, and floats to NumPy arrays
                {}
            )
            for key, value in row_data.items()
        }

        # Increment index, loop back to the start when reaching the end
        self.current_index = (self.current_index + 1) % len(self.df)

        return {"table_output": (table_output, {})}

    def read_filename_changed(self):
        self.load_csv()

    def read_restart_changed(self):
        self.current_index = 0  # Reset index when restart is triggered
