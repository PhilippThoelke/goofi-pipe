import numpy as np
from scipy.signal import stft, istft
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class FrequencyShift(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "shift": {
                "frequency_shift": FloatParam(1.0, -1000.0, 1000.0, doc="Frequency shift in Hz"),
            }
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        signal = np.array(data.data).flatten()  # Assuming the input is 1D for simplicity
        sfreq = data.meta["sfreq"]

        # Perform STFT
        f, t, Zxx = stft(signal, fs=sfreq, nperseg=1024)

        # Frequency shifting
        frequency_shift = self.params["shift"]["frequency_shift"].value
        delta_bin = -int(frequency_shift * len(f) / sfreq)  # Negate delta_bin computation
        target_bins = np.arange(len(f)) + delta_bin

        # Handling bins that go out of bounds after the shift
        valid_bins = (target_bins >= 0) & (target_bins < len(f))
        Zxx_shifted = np.zeros_like(Zxx)

        # Phase correction and bin shifting
        phase_shift = (2 * np.pi * delta_bin * np.arange(Zxx.shape[1]) / (Zxx.shape[1])) % (2 * np.pi)
        Zxx_shifted[valid_bins] = Zxx[target_bins[valid_bins]] * np.exp(1j * phase_shift)

        # Inverse STFT to retrieve shifted signal
        _, shifted_signal = istft(Zxx_shifted, fs=sfreq)

        # Depending on the exact length and shifts, you might want to match the output size to the input size
        shifted_signal = shifted_signal[: len(signal)]

        return {"out": (shifted_signal, data.meta)}
