import numpy as np
import umap
from numpy.linalg import eig

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, StringParam


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
                "method": StringParam("PCA", options=["PCA", "t-SNE", "UMAP"], doc="Dimensionality reduction method"),
                "n_components": FloatParam(2, 1, 10, doc="Number of output dimensions"),
                "tsne_perplexity": FloatParam(30.0, 5.0, 50.0, doc="t-SNE perplexity"),
                "umap_neighbors": FloatParam(15, 2, 100, doc="Number of UMAP neighbors"),
            }
        }

    def setup(self):
        from sklearn.manifold import TSNE

        self.tsne = TSNE

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

        method = self.params.Control.method.value
        n_components = int(self.params.Control.n_components.value)

        if method == "PCA":
            sfreq = data.meta.get("sfreq", None)
            if sfreq is None:
                raise ValueError("Sampling frequency (sfreq) must be provided in data.meta")

            samples_to_keep = int(self.params.Control.buffer_size.value * sfreq)
        else:
            samples_to_keep = int(self.params.Control.buffer_size.value)

        if not self.buffer_full:
            if self.buffer is None:
                self.buffer = data_array
            else:
                self.buffer = np.hstack((self.buffer, data_array))

            if self.buffer.shape[1] > samples_to_keep:
                self.buffer = self.buffer[:, -samples_to_keep:]
                self.buffer_full = True

        if self.buffer.shape[1] < 100:
            return None

        if method == "PCA":
            covariance_matrix = np.cov(self.buffer)
            eigenvalues, eigenvectors = eig(covariance_matrix)
            idx = eigenvalues.argsort()[::-1]
            principal_components = eigenvectors[:, idx[:n_components]]

            self.last_principal_components = (principal_components, data.meta)

            return {"principal_components": self.last_principal_components}

        elif method == "t-SNE":
            perplexity = self.params.Control.tsne_perplexity.value
            tsne = self.TSNE(n_components=n_components, perplexity=perplexity, init="pca", random_state=42)
            principal_components = tsne.fit_transform(self.buffer)

            last_principal_components = (principal_components, {})
            return {"principal_components": last_principal_components}

        elif method == "UMAP":
            n_neighbors = int(self.params.Control.umap_neighbors.value)
            umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
            principal_components = umap_model.fit_transform(self.buffer)

            last_principal_components = (principal_components, {})
            return {"principal_components": last_principal_components}

        else:
            raise ValueError(f"Unsupported method: {method}")

        # self.last_principal_components = (principal_components, data.meta)

        # return {"principal_components": self.last_principal_components}
