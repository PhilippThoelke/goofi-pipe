import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class RandomArray(Node):
    """
    A Goofi node that generates a random matrix based on specified dimensions and distribution.
    The dimensions are provided as a string of comma-separated integers.
    The distribution can be specified as "uniform" or "normal".
    An option is provided to set the largest eigenvalue of the matrix to 1.
    """

    def config_params():
        return {
            "random": {
                "dimensions": "2,2",  # Default dimensions
                "distribution": StringParam("normal", options=["uniform", "normal"]),
                "normalize_eigenvalue": False,  # Default: do not normalize eigenvalue
                "reset": BoolParam(trigger=True),
            },
            "common": {
                "autotrigger": True,
            },
        }

    def config_output_slots():
        return {"random_array": DataType.ARRAY}

    def setup(self):
        self.random_array = None

        distribution = self.params.random.distribution.value
        normalize_eigenvalue = self.params.random.normalize_eigenvalue.value

        try:
            # Parse the dimensions string into a tuple of integers
            dimensions = tuple(map(int, self.params.random.dimensions.value.split(",")))

            # Generate a random array based on the specified distribution
            if distribution == "uniform":
                random_array = np.random.uniform(size=dimensions)
            elif distribution == "normal":
                random_array = np.random.normal(0, 1, size=dimensions)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}. Use 'uniform' or 'normal'.")

            # Normalize the largest eigenvalue to 1 if specified
            if normalize_eigenvalue and len(dimensions) == 2 and dimensions[0] == dimensions[1]:
                eigenvalues, _ = np.linalg.eig(random_array)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                if max_eigenvalue != 0:
                    random_array /= max_eigenvalue

            self.random_array = random_array

        except (ValueError, TypeError) as e:
            # Handle errors in parsing dimensions or generating the array
            raise ValueError("Invalid dimensions format or distribution type. Please provide a valid input.") from e

    def process(self):
        return {"random_array": (self.random_array, {})}

    def random_dimensions_changed(self, val):
        self.setup()

    def random_distribution_changed(self, val):
        self.setup()

    def random_normalize_eigenvalue_changed(self, val):
        self.setup()

    def random_reset_changed(self, val):
        self.setup()
