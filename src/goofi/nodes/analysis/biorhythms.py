import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam


class Biorhythms(Node):
    def config_input_slots():
        return {"tuning": DataType.ARRAY}

    def config_output_slots():
        return {
            "pulses": DataType.ARRAY,
            "steps": DataType.ARRAY,
            "offsets": DataType.ARRAY,
        }

    def config_params():
        return {
            "Euclidean": {
                "n_steps_down": IntParam(2, 0, 10, doc="Number of time to divide the ratio"),
                "limit_denom": IntParam(8, 1, 100, doc="Maximum denominator of the ratio"),
                "limit_cons": FloatParam(0.5, 0.0, 1.0, doc="Minimum consonance of the ratio"),
                "limit_denom_final": IntParam(8, 1, 100, doc="Maximum denominator of the final ratio"),
                "optimize_offset": BoolParam(False),
            }
        }

    def setup(self):
        from biotuner.rhythm_construction import consonant_euclid, find_optimal_offsets

        self.consonant_euclid = consonant_euclid
        self.find_optimal_offsets = find_optimal_offsets

    def process(self, tuning: Data):
        if tuning is None:
            return None

        tuning.data = np.squeeze(tuning.data)
        if tuning.data.ndim > 1:
            raise ValueError("Data must be 1D")

        n_steps_down = self.params["Euclidean"]["n_steps_down"].value
        limit_denom = self.params["Euclidean"]["limit_denom"].value
        limit_cons = self.params["Euclidean"]["limit_cons"].value
        limit_denom_final = self.params["Euclidean"]["limit_denom_final"].value
        optimize_offset = self.params["Euclidean"]["optimize_offset"].value
        # Derive consonant euclidian rhythms from the harmonic tuning
        euclid_final, cons = self.consonant_euclid(
            list(tuning.data),
            n_steps_down=n_steps_down,
            limit_denom=limit_denom,
            limit_cons=limit_cons,
            limit_denom_final=limit_denom_final,
        )
        # Calculate the pulses and steps
        pulses = []
        steps = []
        for i in range(len(euclid_final)):
            pulses.append(euclid_final[i].count(1))
            steps.append(len(euclid_final[i]))

        if optimize_offset:
            # find the optimal offsets
            offsets = self.find_optimal_offsets(list(zip(pulses, steps)))
        else:
            offsets = [0] * len(pulses)

        return {
            "pulses": (np.array(pulses), tuning.meta),
            "steps": (np.array(steps), tuning.meta),
            "offsets": (np.array(offsets), tuning.meta),
        }
