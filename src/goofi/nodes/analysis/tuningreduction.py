import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


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
                "function": StringParam("harmsim", options=["harmsim", "cons", "denom"], doc="Harmonicity methods"),
            }
        }

    def setup(self):
        from biotuner.scale_construction import create_mode
        from biotuner.metrics import dyad_similarity, metric_denom, compute_consonance

        self.create_mode = create_mode
        self.dyad_similarity = dyad_similarity
        self.metric_denom = metric_denom
        self.compute_consonance = compute_consonance

    def process(self, tuning: Data):
        if tuning is None:
            return None

        tuning.data = np.squeeze(tuning.data)
        if tuning.data.ndim > 1:
            raise ValueError("Data must be 1D")

        n_steps = self.params["Mode_Generation"]["n_steps"].value
        function = self.params["Mode_Generation"]["function"].value
        if function == "harmsim":
            reduced = self.create_mode(tuning.data, n_steps, self.dyad_similarity)
        if function == "denom":
            reduced = self.create_mode(tuning.data, n_steps, self.metric_denom)
        if function == "cons":
            reduced = self.create_mode(tuning.data, n_steps, self.compute_consonance)
        return {
            "reduced": (np.array(reduced), tuning.meta),
        }
