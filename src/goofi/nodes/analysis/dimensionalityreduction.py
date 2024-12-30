import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class DimensionalityReduction(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY, "new_data": DataType.ARRAY}

    def config_output_slots():
        return {
            "transformed": DataType.ARRAY,
            "new_components": DataType.ARRAY,
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

        self.model = None
        self.components = None
        self.meta = None

    def process(self, data: Data, new_data: Data):
        if data is None:
            return None

        data_array = np.squeeze(data.data)

        if self.params.Control.reset.value:
            self.model = None
            self.components = None
            self.meta = None

        if self.components is not None:
            new_components = None
            if new_data is not None and self.model is not None:
                new_components = self.model.transform(new_data.data)
            return {
                "transformed": (self.components, self.meta),
                "new_components": (new_components, self.meta) if new_components is not None else None,
            }

        if data_array.ndim != 2:
            raise ValueError("Data must be 2D")

        method = self.params.Control.method.value
        n_components = int(self.params.Control.n_components.value)
        perplexity = self.params.Control.tsne_perplexity.value
        n_neighbors = int(self.params.Control.umap_neighbors.value)

        self.meta = data.meta.copy()
        if "channels" in self.meta and "dim0" in self.meta["channels"]:
            del self.meta["channels"]["dim0"]

        new_components = None
        if method == "PCA":
            self.model = self.pca(n_components=n_components)
            self.components = self.model.fit_transform(data_array)

            if new_data is not None:
                new_components = self.model.transform(new_data.data)
        elif method == "t-SNE":
            self.model = self.tsne(n_components=n_components, perplexity=perplexity, init="pca", random_state=42)
            self.components = self.model.fit_transform(data_array)
        elif method == "UMAP":
            # Initialize UMAP model
            self.model = self.umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
            # Train UMAP on the original dataset
            self.model = self.model.fit(data_array)  # Save the model for future use
            self.components = self.model.transform(data_array)  # Project original data into the UMAP space

            if new_data is not None:
                new_components = self.umap_model.transform(new_data.data)  # Project new data
        else:
            raise ValueError(f"Unsupported method: {method}")

        return {
            "transformed": (self.components, self.meta),
            "new_components": (new_components, self.meta) if new_components is not None else None,
        }
