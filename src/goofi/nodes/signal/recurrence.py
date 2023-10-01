import numpy as np
from sklearn.metrics import pairwise_distances

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Recurrence(Node):
    def config_input_slots():
        return {"input_array": DataType.ARRAY}

    def config_output_slots():
        return {"recurrence_matrix": DataType.ARRAY}

    def config_params():
        return {
            "recurrence": {
                "threshold": FloatParam(0.1, 0.0, 10.0),  # You can adjust the bounds as per your needs
            }
        }

    def process(self, input_array: Data):
        if input_array is None or input_array.data is None:
            return None

        array = input_array.data

        if array.shape[0] < 2:
            print("Not enough data points to compute recurrence matrix.")
            return None

        threshold = self.params["recurrence"]["threshold"].value

        # Compute pairwise distance
        distance_matrix = pairwise_distances(array)
        # Handling potential NaN or Inf values
        # if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
        #    print("Warning: The distance matrix contains NaN or Inf values. They will be replaced with a large number.")
        #    distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf)
        #    distance_matrix = np.clip(distance_matrix, a_min=None, a_max=np.finfo(np.float32).max)

        return {"recurrence_matrix": (distance_matrix, input_array.meta)}
