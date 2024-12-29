import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class DimensionalityReduction(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "transformed": DataType.ARRAY,
        }

    def config_params():
        return {
            "Control": {
                "reset": BoolParam(False, trigger=True, doc="Reset the buffer"),
                "method": StringParam("PCA", options=["PCA", "t-SNE", "UMAP"], doc="Dimensionality reduction method"),
                "n_components": IntParam(2, 1, 10, doc="Number of output dimensions"),
                "tsne_perplexity": FloatParam(30.0, 5.0, 50.0, doc="t-SNE perplexity"),
                "umap_neighbors": FloatParam(15, 2, 100, doc="Number of UMAP neighbors"),
            }
        }

    def setup(self):
        import umap
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        self.tsne = TSNE
        self.pca = PCA
        self.umap = umap

        self.components = None
        self.meta = None

    def process(self, data: Data):
        if data is None:
            return None

        data_array = np.squeeze(data.data)

        if self.params.Control.reset.value:
            self.components = None
            self.meta = None

        if self.components is not None:
            return {"transformed": (self.components, self.meta)}

        if data_array.ndim != 2:
            raise ValueError("Data must be 2D")

        method = self.params.Control.method.value
        n_components = int(self.params.Control.n_components.value)
        perplexity = self.params.Control.tsne_perplexity.value
        n_neighbors = int(self.params.Control.umap_neighbors.value)

        self.meta = data.meta.copy()
        if "channels" in self.meta and "dim0" in self.meta["channels"]:
            del self.meta["channels"]["dim0"]

        if method == "PCA":
            pca = self.pca(n_components=n_components)
            self.components = pca.fit_transform(data_array)
        elif method == "t-SNE":
            tsne = self.tsne(n_components=n_components, perplexity=perplexity, init="pca", random_state=42)
            self.components = tsne.fit_transform(data_array)
        elif method == "UMAP":
            umap_model = self.umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
            self.components = umap_model.fit_transform(data_array)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return {"transformed": (self.components, self.meta)}
