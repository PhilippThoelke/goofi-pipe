import numpy as np
from biotuner.metrics import dyad_similarity, compute_subharmonic_tension
from biotuner.biotuner_utils import apply_power_law_remove
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam

class HarmonicSpectrum(Node):
    def config_input_slots():
        return {"psd": DataType.ARRAY}
    
    def config_output_slots():
        return {"harmonic_spectrum": DataType.ARRAY}
    
    def config_params():
        return {
            "harmonic": {
                "metric": StringParam("harmsim", options=["harmsim", "subharm_tension"]),
                "normalize": True,
                "power_law_remove": True,
                "n_harms": IntParam(3, 1, 10),
                "delta_lim": IntParam(250, 1, 300),
                "min_notes": IntParam(2, 1, 10)
            }
        }
    def setup(self):
        self.cached_freqs = None
        self.cached_harmonicity_matrix = None
        self.cached_metric = None
    def process(self, psd: Data):
        if psd is None or psd.data is None:
            return None
        
        psd_values, metadata = psd.data, psd.meta
        freqs = metadata["freq"]
        power_law_remove = self.params["harmonic"]["power_law_remove"].value
        if power_law_remove:
            psd_values = apply_power_law_remove(freqs, psd_values, power_law_remove)
            # clip the psd_values to avoid negative values by converting them to 0
            psd_values = np.clip(psd_values, 0, None)
            
        metric = self.params["harmonic"]["metric"].value
        normalize = self.params["harmonic"]["normalize"].value
        n_harms = self.params["harmonic"]["n_harms"].value
        delta_lim = self.params["harmonic"]["delta_lim"].value
        min_notes = self.params["harmonic"]["min_notes"].value
        
        if np.array_equal(self.cached_freqs, freqs) and self.cached_metric == metric:
            harmonicity_matrix = self.cached_harmonicity_matrix
        else:
            harmonicity_matrix = self.compute_harmonicity(freqs, metric, n_harms, delta_lim, min_notes)
            # Cache the freqs and harmonicity matrix for next time
            self.cached_freqs = freqs
            self.cached_harmonicity_matrix = harmonicity_matrix
            self.cached_metric = metric  # Cache the current metric
        harmonicity_values = np.zeros(len(freqs))

        total_power = np.sum(psd_values ** 2)  # Assuming psd_values is a 1-D array. If not, adapt accordingly.
        
        for i in range(len(freqs)):
            weighted_sum_harmonicity = 0
            for j in range(len(freqs)):
                if i != j:
                    weighted_sum_harmonicity += harmonicity_matrix[i, j] * (psd_values[i] * psd_values[j])
            
            harmonicity_values[i] = (weighted_sum_harmonicity / (2 * total_power)) if normalize else weighted_sum_harmonicity
        
        return {"harmonic_spectrum": (harmonicity_values, {"freq": freqs, **metadata})}
    
    def compute_harmonicity(self, freqs, metric, n_harms, delta_lim, min_notes):
        harmonicity = np.zeros((len(freqs), len(freqs)))
        
        for i, f1 in enumerate(freqs):
            for j, f2 in enumerate(freqs):
                if f1 != f2:  # avoiding division by zero and self-comparison
                    if metric == 'harmsim':
                        harmonicity[i, j] = dyad_similarity(f1 / f2)
                    elif metric == 'subharm_tension':
                        _, _, subharm, _ = compute_subharmonic_tension([f1, f2], n_harmonics=n_harms, delta_lim=delta_lim, min_notes=min_notes)
                        try:
                            harmonicity[i, j] = 1 - subharm[0]
                        except:
                            harmonicity[i, j] = None
        return harmonicity

        