import colorsys

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class TuningColors(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"hue": DataType.ARRAY, "saturation": DataType.ARRAY, "value": DataType.ARRAY}

    def setup(self):
        from biotuner.biocolors import audible2visible, scale2freqs, wavelength_to_rgb
        from biotuner.biotuner_object import dyad_similarity
        from biotuner.metrics import tuning_cons_matrix

        self.audible2visible = audible2visible
        self.scale2freqs = scale2freqs
        self.wavelength_to_rgb = wavelength_to_rgb
        self.dyad_similarity = dyad_similarity
        self.tuning_cons_matrix = tuning_cons_matrix

    def process(self, data: Data):
        """
        Convert a musical scale into a list of HSV colors based on the scale's frequency values
        and their averaged consonance.
        """
        if data is None:
            return None

        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        scale = data.data  # a list of frequency ratios representing the musical scale
        fund = data.data[0]  # fundamental frequency of the scale in Hz

        min_ = 0
        max_ = 1
        # convert the scale to frequency values
        scale_freqs = self.scale2freqs(scale, fund)
        # compute the averaged consonance of each step
        scale_cons, _ = self.tuning_cons_matrix(scale, self.dyad_similarity, ratio_type="all")
        # rescale to match RGB standards (0, 255)
        scale_cons = (np.array(scale_cons) - min_) * (1 / max_ - min_) * 255
        scale_cons = scale_cons.astype("uint8").astype(float) / 255

        hsv_all = []
        for s, cons in zip(scale_freqs, scale_cons):
            # convert freq in nanometer values
            _, _, nm, octave = self.audible2visible(s)
            # convert to RGB values
            rgb = self.wavelength_to_rgb(nm)
            # convert to HSV values
            # TODO: colorsys might be slow
            hsv = colorsys.rgb_to_hsv(rgb[0] / float(255), rgb[1] / float(255), rgb[2] / float(255))
            hsv = np.array(hsv)
            # rescale
            hsv = (hsv - 0) * (1 / (1 - 0))
            # define the saturation
            hsv[1] = cons
            # define the luminance
            hsv[2] = 200 / 255
            hsv = tuple(hsv)
            hsv_all.append(hsv)

        # hsv_all is a list of HSV color tuples, one for each scale step, with hue representing the
        # frequency value, saturation representing the consonance, and luminance set to a fixed value
        hsvs = hsv_all[1:]

        return {
            "hue": (np.array([x[0] for x in hsvs]), data.meta),
            "saturation": (np.array([x[1] for x in hsvs]), data.meta),
            "value": (np.array([x[2] for x in hsvs]), data.meta),
        }
