import numpy as np
from goofi.node import Node
from goofi.data import DataType, Data
import colorsys


class HSVtoRGB(Node):
    def config_input_slots():
        return {"hsv_image": DataType.ARRAY}

    def config_output_slots():
        return {"rgb_image": DataType.ARRAY}

    def config_params():
        return {}  # No parameters needed for this transformation

    def process(self, hsv_image: Data):
        if hsv_image is None or hsv_image.data is None:
            return None

        # Extract HSV values
        h, s, v = hsv_image.data[..., 0], hsv_image.data[..., 1], hsv_image.data[..., 2]

        # Convert HSV to RGB
        rgb = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)
        rgb_image = np.stack(rgb, axis=-1)

        return {"rgb_image": (rgb_image, {**hsv_image.meta})}
