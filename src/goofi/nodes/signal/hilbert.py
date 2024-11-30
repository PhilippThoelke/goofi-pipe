import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Hilbert(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"inst_amplitude": DataType.ARRAY, "inst_phase": DataType.ARRAY, "inst_frequency": DataType.ARRAY}

    def setup(self):
        from scipy.signal import hilbert

        self.hilbert = hilbert

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        analytic_signal = self.hilbert(data.data)
        inst_amplitude = np.abs(analytic_signal)
        inst_phase = np.angle(analytic_signal)

        # Compute the instantaneous frequency:
        delta_phase = np.diff(np.unwrap(inst_phase), axis=-1)
        inst_frequency = delta_phase / (2.0 * np.pi)

        # Pad inst_frequency to make it the same length as inst_amplitude and inst_phase.
        # Using padding with the last value to keep the size consistent with inst_amplitude and inst_phase.
        pad_value = inst_frequency[..., -1:] if inst_frequency.ndim > 1 else [inst_frequency[-1]]
        inst_frequency = np.concatenate((inst_frequency, pad_value), axis=-1)

        return {
            "inst_amplitude": (inst_amplitude, {**data.meta}),
            "inst_phase": (inst_phase, {**data.meta}),
            "inst_frequency": (inst_frequency, {**data.meta}),
        }
