import numpy as np
from numpy.fft import ifft
from goofi.data import Data, DataType
from goofi.node import Node

class IFFT(Node):
    def config_input_slots():
        return {"spectrum": DataType.ARRAY, "phase": DataType.ARRAY}

    def config_output_slots():
        return {"reconstructed": DataType.ARRAY}

    def config_params():
        return {}  # No parameters for this node

    def process(self, spectrum: Data, phase: Data):
        # Check if the input is provided
        if spectrum is None or spectrum.data is None or phase is None or phase.data is None:
            return None

        # Create complex numbers from magnitude (spectrum) and phase
        complex_data = spectrum.data * np.exp(1j * phase.data)

        # Inverse Fourier Transform to get the time series
        time_series = np.real(ifft(complex_data))

        # For this example, I'm not adjusting or copying metadata, but you can do so as needed
        return {"reconstructed": (time_series, {})}
