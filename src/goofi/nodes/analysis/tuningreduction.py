import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam

from biotuner.scale_construction import create_mode
from biotuner.metrics import dyad_similarity, metric_denom, compute_consonance

class TuningReduction(Node):
    def config_input_slots():
        return {"tuning": DataType.ARRAY}

    def config_output_slots():
        return {
            "reduced": DataType.ARRAY,
            }

    def config_params():
        return {
            "Mode_Generation": {
                "n_steps": IntParam(5, 2, 20, doc="Number of steps in the musical mode output"),
                "function": StringParam("harmsim", options=['harmsim', 'cons', 'denom'], doc="Harmonicity methods"),
            }
        }

    def process(self, tuning: Data):
        if tuning is None:
            return None

        tuning.data = np.squeeze(tuning.data)
        if tuning.data.ndim > 1:
            raise ValueError("Data must be 1D")
        
        n_steps = self.params['Mode_Generation']['n_steps'].value
        function = self.params['Mode_Generation']['function'].value
        if function == 'harmsim':
            reduced = create_mode(tuning.data, n_steps, dyad_similarity)
        if function == 'denom':
            reduced = create_mode(tuning.data, n_steps, metric_denom)
        if function == 'cons':
            reduced = create_mode(tuning.data, n_steps, compute_consonance)
        return {
            "reduced": (np.array(reduced), tuning.meta),
        }


'''def call_function(tuning, steps, method):
    global create_mode, dyad_similarity, metric_denom, compute_consonance
    if compute_biotuner_fn is None or harmonic_tuning_fn is None:
        from biotuner.biotuner_object import compute_biotuner, harmonic_tuning

        compute_biotuner_fn = compute_biotuner
        harmonic_tuning_fn = harmonic_tuning'''
