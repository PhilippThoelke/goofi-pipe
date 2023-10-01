from sklearn.metrics import pairwise_distances

from goofi.data import Data, DataType
from goofi.node import Node


class Recurrence(Node):
    def config_input_slots():
        return {"input_array": DataType.ARRAY}

    def config_output_slots():
        return {"recurrence_matrix": DataType.ARRAY}

    def process(self, input_array: Data):
        if input_array is None:
            return None

        if input_array.data.shape[0] < 2:
            raise ValueError("Not enough data points to compute recurrence matrix.")

        # Compute pairwise distance
        distance_matrix = pairwise_distances(input_array.data)
        # Handling potential NaN or Inf values
        # if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
        #    print("Warning: The distance matrix contains NaN or Inf values. They will be replaced with a large number.")
        #    distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf)
        #    distance_matrix = np.clip(distance_matrix, a_min=None, a_max=np.finfo(np.float32).max)

        return {"recurrence_matrix": (distance_matrix, input_array.meta)}
