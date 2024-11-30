import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class GraphMetrics(Node):
    def config_input_slots():
        return {"matrix": DataType.ARRAY}

    def config_output_slots():
        return {
            "clustering_coefficient": DataType.ARRAY,
            "characteristic_path_length": DataType.ARRAY,
            "betweenness_centrality": DataType.ARRAY,
            "degree_centrality": DataType.ARRAY,
            "assortativity": DataType.ARRAY,
            "transitivity": DataType.ARRAY,
        }

    def setup(self):
        import networkx as nx

        self.nx = nx

    def process(self, matrix: Data):
        if matrix is None:
            return None

        # Ensure data is 2D and symmetric
        if matrix.data.ndim != 2 or matrix.data.shape[0] != matrix.data.shape[1]:
            raise ValueError("Matrix must be 2D and symmetric.")

        # Create a graph from the matrix (assuming undirected graph)
        G = self.nx.from_numpy_array(matrix.data)

        # Compute metrics
        clustering_coefficients = self.nx.average_clustering(G)
        try:
            path_length = self.nx.average_shortest_path_length(G)
        except self.nx.NetworkXError:  # Handles cases where the graph is not connected
            path_length = None
        betweenness = self.nx.betweenness_centrality(G)
        betweenness = np.array(list(betweenness.values()))
        degree_centrality = self.nx.degree_centrality(G)
        degree_centrality = np.array(list(degree_centrality.values()))
        assortativity = self.nx.degree_assortativity_coefficient(G)
        transitivity = self.nx.transitivity(G)

        return {
            "clustering_coefficient": (np.array(clustering_coefficients), {}),
            "characteristic_path_length": (np.array(path_length), {}),
            "betweenness_centrality": (np.array(betweenness), matrix.meta),
            "degree_centrality": (np.array(degree_centrality), matrix.meta),
            "assortativity": (np.array(assortativity), {}),
            "transitivity": (np.array(transitivity), {}),
        }
