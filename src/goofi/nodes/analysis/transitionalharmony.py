import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class TransitionalHarmony(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"trans_harm": DataType.ARRAY, "melody": DataType.ARRAY}

    def config_params():
        return {
            "transitional_harmony": {
                "n_peaks": IntParam(5, 1, 10),
                "f_min": FloatParam(2.0, 0.1, 50.0),
                "f_max": FloatParam(30.0, 1.0, 100.0),
                "precision": FloatParam(0.1, 0.01, 10.0),
                "peaks_function": StringParam("EMD", options=["EMD", "fixed", "harmonic_recurrence", "EIMC", "cepstrum"]),
                "delta": IntParam(50, 1, 250),
                "subharmonics": IntParam(10, 2, 100),
            }
        }

    def setup(self):
        from biotuner.biotuner_object import compute_biotuner
        from biotuner.metrics import compute_subharmonics_2lists

        self.compute_biotuner = compute_biotuner
        self.compute_subharmonics_2lists = compute_subharmonics_2lists

    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        # split data in two
        data_len = len(data.data)
        data1 = data.data[: int(data_len / 2)]
        data2 = data.data[int(data_len / 2) :]

        # extract peaks from data1
        bt1 = self.compute_biotuner(
            sf=data.meta["sfreq"],
            peaks_function=self.params.transitional_harmony.peaks_function.value,
        )
        bt1.peaks_extraction(
            data1,
            n_peaks=self.params.transitional_harmony.n_peaks.value,
            min_freq=self.params.transitional_harmony.f_min.value,
            max_freq=self.params.transitional_harmony.f_max.value,
            precision=self.params.transitional_harmony.precision.value,
        )

        # extract peaks from data2
        bt2 = self.compute_biotuner(
            sf=data.meta["sfreq"],
            peaks_function=self.params.transitional_harmony.peaks_function.value,
        )
        bt2.peaks_extraction(
            data2,
            n_peaks=self.params.transitional_harmony.n_peaks.value,
            min_freq=self.params.transitional_harmony.f_min.value,
            max_freq=self.params.transitional_harmony.f_max.value,
            precision=self.params.transitional_harmony.precision.value,
        )

        # compute subharmonics between peaks
        result = self.compute_subharmonics_2lists(
            bt1.peaks,
            bt2.peaks,
            self.params.transitional_harmony.subharmonics.value,
            delta_lim=self.params.transitional_harmony.delta.value,
            c=2.1,
        )

        common_subs, delta_t, sub_tension_final, harm_temp, pairs_melody = result

        return {
            "trans_harm": (np.array(sub_tension_final), data.meta),
            "melody": (np.array(pairs_melody), data.meta),
        }
