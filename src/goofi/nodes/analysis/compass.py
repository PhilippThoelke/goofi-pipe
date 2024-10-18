import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node


class Compass(Node):
    """
    A Goofi node that calculates the angle from Cartesian coordinates derived from
    directional inputs (north, south, east, west).
    """

    def config_input_slots():
        return {"north": DataType.ARRAY, "south": DataType.ARRAY, "east": DataType.ARRAY, "west": DataType.ARRAY}

    def config_output_slots():
        return {
            "angle": DataType.ARRAY,
        }

    def process(self, north: Data, south: Data, east: Data, west: Data):
        """
        Processes the input directions to compute the angle of the resultant vector
        formed by the differences of west-east and north-south.
        """
        if north is None or south is None or east is None or west is None:
            return None
        # Extract the float values from Data objects
        north_val = north.data
        south_val = south.data
        east_val = east.data
        west_val = west.data

        # Calculate differences
        horizontal_diff = west_val - east_val
        vertical_diff = north_val - south_val

        # Calculate the angle in radians
        angle_radians = np.arctan2(vertical_diff, horizontal_diff)

        # Convert angle to degrees and normalize to 0-360
        angle_degrees = np.degrees(angle_radians)
        if angle_degrees < 0:
            angle_degrees += 360
        # check if sf in metadata and add it to the output metadata

        # Return the angle as a Data object
        return {"angle": (angle_degrees, {})}
