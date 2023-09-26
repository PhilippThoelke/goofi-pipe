import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam
from biotuner.metrics import compute_subharmonics_2lists
from biotuner.biotuner_object import *

class TransitionalHarmony(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "trans_harm": DataType.ARRAY,
            "melody": DataType.ARRAY
        }

    def config_params():
        return {
            "TransitionalHarmony": {
                "n_peaks": IntParam(5, 1, 10),
                "f_min": FloatParam(2.0, 0.1, 50.0),
                "f_max": FloatParam(30.0, 1.0, 100.0),
                "precision": FloatParam(0.1, 0.01, 10.0),
                "peaks_function": "EMD",
                "delta": IntParam(50, 1, 250),
                "subharmonics": IntParam(10, 2, 100)
            }
        }

    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        data_len = len(data.data)
        data1 = data.data[0:int(data_len/2)]
        data2 = data.data[int(data_len/2):]
        bt1 = compute_biotuner(sf=data.meta["sfreq"],
                                peaks_function=self.params["TransitionalHarmony"]["peaks_function"].value,
                                )
        bt1.peaks_extraction(data1, n_peaks=self.params["TransitionalHarmony"]["n_peaks"].value,
                             min_freq=self.params["TransitionalHarmony"]["f_min"].value,
                                max_freq=self.params["TransitionalHarmony"]["f_max"].value,
                                precision=self.params["TransitionalHarmony"]["precision"].value)
        bt2 = compute_biotuner(sf=data.meta["sfreq"],
                                peaks_function=self.params["TransitionalHarmony"]["peaks_function"].value,
                                )
        bt2.peaks_extraction(data2, n_peaks=self.params["TransitionalHarmony"]["n_peaks"].value,
                             min_freq=self.params["TransitionalHarmony"]["f_min"].value,
                                max_freq=self.params["TransitionalHarmony"]["f_max"].value,
                                precision=self.params["TransitionalHarmony"]["precision"].value)
        result = compute_subharmonics_2lists(bt1.peaks,
                                            bt2.peaks,
                                            self.params["TransitionalHarmony"]["subharmonics"].value,
                                            delta_lim=self.params["TransitionalHarmony"]["delta"].value,
                                            c=2.1)
        
        common_subs, delta_t, sub_tension_final, harm_temp, pairs_melody = result

        return {
            "trans_harm": (np.array(sub_tension_final), data.meta),
            "melody": (np.array(pairs_melody), data.meta),
        }

