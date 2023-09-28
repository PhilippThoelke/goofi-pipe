import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Biotuner(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "harmsim": DataType.ARRAY,
            "tenney": DataType.ARRAY,
            "subharm_tension": DataType.ARRAY,
            "cons": DataType.ARRAY,
            "peaks_ratios_tuning": DataType.ARRAY,
            "harm_tuning": DataType.ARRAY,
            "peaks": DataType.ARRAY,
            "amps": DataType.ARRAY,
            "extended_peaks": DataType.ARRAY,
        }

    def config_params():
        return {
            "biotuner": {
                "n_peaks": IntParam(5, 1, 10),
                "f_min": FloatParam(2.0, 0.1, 50.0),
                "f_max": FloatParam(30.0, 1.0, 100.0),
                "precision": FloatParam(0.1, 0.01, 10.0),
                "peaks_function": StringParam("EMD", options=["EMD", "fixed"]),
            }
        }

    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        result = biotuner_realtime(
            data.data,
            data.meta["sfreq"],
            n_peaks=self.params["biotuner"]["n_peaks"].value,
            peaks_function=self.params["biotuner"]["peaks_function"].value,
            min_freq=self.params["biotuner"]["f_min"].value,
            max_freq=self.params["biotuner"]["f_max"].value,
            precision=self.params["biotuner"]["precision"].value,
        )
        peaks, extended_peaks, metrics, tuning, harm_tuning, amps = result

        return {
            "harmsim": (np.array([metrics["harmsim"]]), data.meta),
            "tenney": (np.array([metrics["tenney"]]), data.meta),
            "subharm_tension": (np.array([metrics["subharm_tension"][0]]), data.meta),
            "cons": (np.array([metrics["cons"]]), data.meta),
            "peaks_ratios_tuning": (np.array(tuning), data.meta),
            "harm_tuning": (np.array(harm_tuning), data.meta),
            "peaks": (np.array(peaks), data.meta),
            "amps": (np.array(amps), data.meta),
            "extended_peaks": (np.array(extended_peaks), data.meta),
        }


compute_biotuner_fn, harmonic_tuning_fn = None, None


def biotuner_realtime(data, sfreq, n_peaks=5, peaks_function="EMD", min_freq=1, max_freq=65, precision=0.1):
    # import the biotuner function here to avoid loading it on startup
    global compute_biotuner_fn, harmonic_tuning_fn
    if compute_biotuner_fn is None or harmonic_tuning_fn is None:
        from biotuner.biotuner_object import compute_biotuner, harmonic_tuning

        compute_biotuner_fn = compute_biotuner
        harmonic_tuning_fn = harmonic_tuning

    # run biotuner peak extraction
    bt = compute_biotuner_fn(peaks_function=peaks_function, sf=sfreq)
    try:
        bt.peaks_extraction(
            np.array(data),
            graph=False,
            min_freq=min_freq,
            max_freq=max_freq,
            precision=precision,
            nIMFs=5,
            n_peaks=n_peaks,
            smooth_fft=2,
        )
    except UnboundLocalError:
        raise RuntimeError("No peaks found. Try increasing the length of the signal.")

    try:
        # try computing the extended peaks
        bt.peaks_extension(method="harmonic_fit")
    except TypeError:
        raise RuntimeError("Detected only one peak. The largest peak might be below the minimum frequency.")

    bt.compute_peaks_metrics(n_harm=3, delta_lim=250)
    if hasattr(bt, "all_harmonics"):
        harm_tuning = harmonic_tuning_fn(bt.all_harmonics)
    else:
        harm_tuning = [0, 1]
    peaks = bt.peaks
    amps = bt.amps
    extended_peaks = bt.extended_peaks
    metrics = bt.peaks_metrics

    if not isinstance(metrics["subharm_tension"][0], float):
        metrics["subharm_tension"] = [np.nan]
    metrics["harmsim"] = metrics["harmsim"] / 100
    # rescale tenney height from between 4 to 9 to between 0 and 1
    metrics["tenney"] = (metrics["tenney"] - 4) / 5
    tuning = bt.peaks_ratios
    return peaks, extended_peaks, metrics, tuning, harm_tuning, amps
