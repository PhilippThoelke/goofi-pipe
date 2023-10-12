import numpy as np
from scipy import stats
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Correlation(Node):
    def config_input_slots():
        # Defining two input slots for two input signals
        return {"data1": DataType.ARRAY, "data2": DataType.ARRAY}

    def config_output_slots():
        # Defining two output slots for the resampled signals
        return {"pearson": DataType.ARRAY}

    def config_params():
        return {}

    def process(self, data1: Data, data2: Data):
        if data1 is None or data1.data is None or data2 is None or data2.data is None:
            return None

        r, p = stats.pearsonr(data1.data, data2.data)

        return {"pearson": (np.array(r), data1.meta)}
