import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

class EigenDecomposition(Node):
    def config_input_slots():
        return {
            "matrix": DataType.ARRAY
        }

    def config_output_slots():
        return {
            "eigenvalues": DataType.ARRAY,
            "eigenvectors": DataType.ARRAY,
        }

    def config_params():
        return {
            "Eigen": {
                "laplacian" : StringParam("none", options=["none", "unnormalized", "normalized"]),
                "method": StringParam("eigh", options=["eigh", "eigh_general", "eig"]),
            }
        }

    def process(self, matrix: Data):
        
        if matrix is None:
            return None

        matrix_data = np.squeeze(matrix.data)
        if matrix_data.ndim != 2:
            raise ValueError("Matrix must be 2D")

        if self.params.Eigen.laplacian.value == "unnormalized":
            # Compute the Laplacian of the connectivity matrix
            matrix_data = laplacian(matrix_data, normed=False)
        if self.params.Eigen.laplacian.value == "normalized":
            matrix_data = laplacian(matrix_data, normed=True)
        
        method = self.params.Eigen.method.value
        
        if method == "eigh":
            eigenvalues, eigenvectors = np.linalg.eigh(matrix_data)
        if method == "eigh_general":
            eigenvalues, eigenvectors = eigh(matrix_data)
        if method == "eig":
            eigenvalues, eigenvectors = np.linalg.eig(matrix_data)

        # reordering eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        return {
            "eigenvalues": (np.array(eigenvalues), matrix.meta),
            "eigenvectors": (np.array(eigenvectors), matrix.meta)
        }
