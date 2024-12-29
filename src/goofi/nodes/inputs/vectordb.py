import pickle
import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam
from usearch.index import Index
import os


class VectorDB(Node):
    """
    A Goofi node for searching the top N labels from a vector embedding database.
    """

    def config_input_slots():
        return {
            "input_vector": DataType.ARRAY,
        }

    def config_output_slots():
        return {
            "top_labels": DataType.TABLE,
            "vectors": DataType.ARRAY,
        }

    def config_params():
        return {
            "Control": {
                "top_n": IntParam(10, 1, 100, doc="Number of top labels to return"),
                "database_path": StringParam("index_nouns.pkl", doc="Path to the database pickle file"),
            }
        }

    def setup(self):
        """
        Load the database index and mapping during initialization.
        """
        database_path = self.params.Control.database_path.value
        database_path = os.path.join(self.assets_path, database_path)
        try:
            with open(database_path, "rb") as f:
                index_data, self.idx2word = pickle.load(f)
            self.index = Index.restore(index_data)
        except Exception as e:
            raise RuntimeError(f"Failed to load the database from {database_path}: {e}")

    def process(self, input_vector: Data):
        """
        Search for the top N labels in the vector embedding database.

        Args:
            input_vector (Data): Input vector to search for (1D array).

        Returns:
            dict: The top N labels with their distances and vectors.
        """
        if input_vector is None:
            return None

        input_vector_array = np.squeeze(input_vector.data)

        if input_vector_array.ndim != 1:
            raise ValueError("Input vector must be 1D.")

        # Ensure the input vector matches the index dimensionality
        index_dimensionality = self.index.ndim  # Assuming self.index has an 'ndim' attribute
        if input_vector_array.shape[0] != index_dimensionality:
            raise ValueError(
                f"Input vector dimensions ({input_vector_array.shape[0]}) do not match index dimensions ({index_dimensionality})."
            )

        top_n = self.params.Control.top_n.value

        # Perform the search
        search_results = self.index.search(input_vector_array, top_n, exact=True)

        # Retrieve vectors efficiently
        vectors = [self.index.get(result.key) for result in search_results]
        vectors = np.array(vectors)

        # Get the top N labels and their distances
        top_labels = {self.idx2word[result.key]: Data(DataType.ARRAY, result.distance, {}) for result in search_results}

        return {"top_labels": (top_labels, {}), "vectors": (vectors, {})}

    def Control_database_path_changed(self, value):
        self.setup()
