import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class Connectivity(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "matrix": DataType.ARRAY,
        }

    def config_params():
        return {
            "biotuner": {
                "method": "harmsim",
                "n_peaks": IntParam(5, 1, 10),
                "f_min": FloatParam(2.0, 0.1, 50.0),
                "f_max": FloatParam(30.0, 1.0, 100.0),
                "precision": FloatParam(0.1, 0.01, 10.0),
                "peaks_function": "EMD",
            }
        }

    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim != 2:
            raise ValueError("Data must be 2D")

        harm_conn = compute_conn_matrix_single(
            data.data,
            data.meta["sfreq"],
            peaks_function=self.params["biotuner"]["peaks_function"].value,
            min_freq=self.params["biotuner"]["f_min"].value,
            max_freq=self.params["biotuner"]["f_max"].value,
            precision=self.params["biotuner"]["precision"].value,
            n_peaks=self.params["biotuner"]["n_peaks"].value,
            metric=self.params["biotuner"]["method"].value,
        )

        return {"matrix": (harm_conn, data.meta)}


connectivity_fn = None


def compute_conn_matrix_single(
    data: np.ndarray,
    sfreq: float,
    peaks_function: str = "EMD",
    min_freq: float = 2.0,
    max_freq: float = 45.0,
    precision=0.1,
    n_peaks: int = 5,
    metric: str = "harmsim",
):
    # import the connectivity function here to avoid loading it on startup
    global connectivity_fn
    if connectivity_fn is None:
        from biotuner.harmonic_connectivity import harmonic_connectivity

        connectivity_fn = harmonic_connectivity

    # compute connectivity matrix
    bt_conn = connectivity_fn(
        sf=sfreq,
        data=data,
        peaks_function=peaks_function,
        precision=precision,
        min_freq=min_freq,
        max_freq=max_freq,
        n_peaks=n_peaks,
    )
    bt_conn.compute_harm_connectivity(metric=metric, save=False, graph=False)
    return bt_conn.conn_matrix
