import csv
import os
import datetime
from goofi.params import StringParam
from goofi.data import Data, DataType
from goofi.node import Node


class WriteCsv(Node):
    def config_input_slots():
        # This node will accept only a table as its input.
        return {"table_input": DataType.TABLE}

    def config_params():
        # Parameters can include the CSV filename and the write control.
        return {
            "Write": {"filename": StringParam("output.csv"), "write": False},
        }

    def setup(self):
        self.last_filename = None
        self.base_filename = None  # to track the filename without the timestamp
        self.written_files = set()  # Track files we've written headers to

    def process(self, table_input: Data):
        # Check the value of write_control.
        # If False, return without doing anything.
        if not self.params["Write"]["write"].value:
            return

        table_data = table_input.data
        # Extract the actual data content
        actual_data = {key: value.data if isinstance(value, Data) else value for key, value in table_data.items()}

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

        with open(fn, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            # If the file hasn't had headers written yet, write them.
            if fn not in self.written_files:
                writer.writerow(actual_data.keys())
                self.written_files.add(fn)  # Add to our tracking set

            # Write the data values as rows.
            writer.writerow(actual_data.values())

        # Return the path to the generated CSV file as an output
        return
