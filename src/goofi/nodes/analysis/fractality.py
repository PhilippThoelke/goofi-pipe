import neurokit2 as nk
import numpy as np
from goofi.params import StringParam, IntParam, FloatParam
from goofi.data import Data, DataType
from goofi.node import Node


class Fractality(Node):
    def config_input_slots():
        return {"data_input": DataType.ARRAY}

    def config_output_slots():
        return {"fractal_dimension": DataType.ARRAY}

    def config_params():
        return {
            "method": {
                "name": StringParam(
                    "fractal_higuchi",
                    options=[
                        "fractal_katz",
                        "fractal_petrosian",
                        "fractal_linelength",
                        "fractal_psdslope",
                        "fractal_nld",
                        "fractal_higuchi",
                        "box_counting",
                    ],
                ),
            },
            "box_counting": {
                "scales_base": FloatParam(2.0, 1.1, 10.0, doc="Base for the logspace calculation in box_counting"),
                "scales_start": FloatParam(0.01, 0.001, 1.0, doc="Start of the logspace in box_counting"),
                "scales_num": IntParam(10, 5, 100, doc="Number of steps in the logspace for box_counting"),
            },
            "fractal_higuchi": {"k_max": IntParam(10, 1, 100, doc="Maximum k value for fractal_higuchi method")},
        }

    def process(self, data_input: Data):
        if data_input is None or data_input.data is None:
            return None

        method = self.params["method"]["name"].value

        # For methods in neurokit2
        if method == "fractal_katz":
            result, _ = nk.fractal_katz(data_input.data)
        elif method == "fractal_petrosian":
            result, _ = nk.fractal_petrosian(data_input.data)
        elif method == "fractal_linelength":
            result, _ = nk.fractal_linelength(data_input.data)
        elif method == "fractal_psdslope":
            result, _ = nk.fractal_psdslope(data_input.data)
        elif method == "fractal_nld":
            result, _ = nk.fractal_nld(data_input.data)
        elif method == "fractal_higuchi":
            result, _ = nk.fractal_higuchi(data_input.data, k_max=self.params["method"]["k_max"].value)
        elif method == "box_counting":
            result = self.box_counting(data_input.data)
            print(result)

        return {"fractal_dimension": (result, {})}

    def box_counting(self, data):
        # Check if data is either 2D or 3D with shape (x, y, 3)
        if len(data.shape) not in [2, 3] or (len(data.shape) == 3 and data.shape[2] != 3):
            raise ValueError("Data for box_counting should be either 2D or have a shape of (x, y, 3)")

        # Convert from RGB to grayscale if data is 3D
        if len(data.shape) == 3:
            data = self.rgb2gray(data)
        # Threshold the data to create a binary representation
        threshold = np.mean(data)
        data = (data > threshold).astype(int)

        if len(data.shape) != 2:
            raise ValueError("Data for box_counting should be 2D after any necessary conversions")

        # Find all the non-zero pixels (modify this if your image isn't binary)
        pixels = np.column_stack(np.where(data > 0))

        # Retrieve the scales parameters
        scales_base = self.params["box_counting"]["scales_base"].value
        scales_start = self.params["box_counting"]["scales_start"].value
        scales_num = self.params["box_counting"]["scales_num"].value

        scales = np.logspace(scales_start, 1, num=scales_num, endpoint=False, base=scales_base)
        Ns = []

        for scale in scales:
            H, _ = np.histogramdd(pixels, bins=(np.arange(0, data.shape[0], scale), np.arange(0, data.shape[1], scale)))
            Ns.append(np.sum(H > 0))

        coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
        return -coeffs[0]

    def rgb2gray(self, rgb):
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        return gray / 255  # Normalize to [0,1]


# https://github.com/ChatzigeorgiouGroup/FractalDimension/blob/master/FractalDimension.py
