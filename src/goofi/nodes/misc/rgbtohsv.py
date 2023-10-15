import numpy as np
from goofi.node import Node
from goofi.data import DataType, Data
import colorsys


class RGBtoHSV(Node):
    def config_input_slots():
        return {"rgb_image": DataType.ARRAY}

    def config_output_slots():
        return {"hsv_image": DataType.ARRAY}

    def config_params():
        return {}  # No parameters needed for this transformation

    def process(self, rgb_image: Data):
        if rgb_image is None or rgb_image.data is None:
            return None

        # Extract RGB values
        r, g, b = rgb_image.data[..., 0], rgb_image.data[..., 1], rgb_image.data[..., 2]

        # Convert RGB to HSV
        hsv = np.vectorize(colorsys.rgb_to_hsv)(r, g, b)
        hsv_image = np.stack(hsv, axis=-1)

        return {"hsv_image": (hsv_image, {**rgb_image.meta})}
