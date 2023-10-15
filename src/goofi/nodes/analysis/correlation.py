import numpy as np
from scipy import stats

from goofi.data import Data, DataType
from goofi.node import Node


class Correlation(Node):
    def config_input_slots():
        # Defining two input slots for two input signals
        return {"data1": DataType.ARRAY, "data2": DataType.ARRAY}

    def config_output_slots():
        # Defining two output slots for the resampled signals
        return {"pearson": DataType.ARRAY}

    def config_params():
        return {"correlation": {"axis": -1}}

    def process(self, data1: Data, data2: Data):
        if data1 is None or data2 is None:
            return None

        meta = data1.meta

        # broadcast data to same shape
        data1, data2 = np.broadcast_arrays(data1.data, data2.data)
        data = np.stack([data1, data2], axis=0)

        if data.ndim > 3:
            raise ValueError("Correlation only works for 1D and 2D data")

        # update axis param
        axis = self.params.correlation.axis.value
        if axis >= 0:
            axis += 1

        # calculate correlation along axis
        r = np.apply_along_axis(lambda x: stats.pearsonr(*x.reshape(2, -1))[0], axis, data)[0]

        return {"pearson": (r, meta)}
