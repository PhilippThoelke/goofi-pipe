import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam


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
                "reset": BoolParam(False, trigger=True, doc="Reset the buffer"),
                "n_components": IntParam(2, 1, 10, doc="Number of output dimensions"),
            }
        }

    def setup(self):
        from sklearn.decomposition import PCA

        self.pca = PCA

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
            return {"principal_components": (self.components, self.meta)}

        if data_array.ndim != 2:
            raise ValueError("Data must be 2D")

        n_components = int(self.params.Control.n_components.value)

        self.meta = data.meta.copy()
        if "channels" in self.meta and "dim0" in self.meta["channels"]:
            del self.meta["channels"]["dim0"]

        pca = self.pca(n_components=n_components)
        pca.fit(data_array)
        self.components = pca.components_

        return {"principal_components": (self.components, self.meta)}
