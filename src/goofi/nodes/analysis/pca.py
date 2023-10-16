import numpy as np
from numpy.linalg import eig
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, BoolParam


class PCA(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "principal_components": DataType.ARRAY,
        }

    def config_params():
        return {
            "Control": {
                "buffer_size": FloatParam(5.0, 1.0, 300.0, doc="Buffer size in seconds"),
                "reset": BoolParam(False, trigger=True, doc="Reset the buffer"),
            }
        }

    def setup(self):
        self.buffer = None
        self.buffer_full = False
        self.last_principal_components = None

    def process(self, data: Data):
        if data is None:
            return None

        data_array = np.squeeze(data.data)

        if self.params.Control.reset.value:
            self.buffer = None
            self.buffer_full = False
            self.last_principal_components = None
            return None

        if data_array.ndim != 2:
            raise ValueError("Data must be 2D")

        sfreq = data.meta.get("sfreq", None)
        if sfreq is None:
            raise ValueError("Sampling frequency (sfreq) must be provided in data.meta")

        samples_to_keep = int(self.params.Control.buffer_size.value * sfreq)

        if not self.buffer_full:
            if self.buffer is None:
                self.buffer = data_array
            else:
                self.buffer = np.hstack((self.buffer, data_array))

            if self.buffer.shape[1] > samples_to_keep:
                self.buffer = self.buffer[:, -samples_to_keep:]
                if not self.buffer_full:
                    self.buffer_full = True
                else:
                    # Buffer was already full previously, so return the last computed principal components.
                    return {"principal_components": self.last_principal_components}

        if self.buffer.shape[1] < 100:
            return None
        covariance_matrix = np.cov(self.buffer)
        eigenvalues, eigenvectors = eig(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        principal_components = eigenvectors[:, idx[: data_array.shape[0]]]

        self.last_principal_components = (principal_components, data.meta)

        return {"principal_components": self.last_principal_components}
