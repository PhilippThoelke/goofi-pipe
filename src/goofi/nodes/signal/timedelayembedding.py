import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam


class TimeDelayEmbedding(Node):
    def config_input_slots():
        return {"input_array": DataType.ARRAY}

    def config_output_slots():
        return {"embedded_array": DataType.ARRAY}

    def config_params():
        return {
            "embedding": {"delay": IntParam(1, 1, 100), "embedding_dimension": IntParam(2, 2, 100), "moire_embedding": False}
        }

    def process(self, input_array: Data):
        if input_array is None or input_array.data is None:
            return None

        delay = self.params["embedding"]["delay"].value
        embedding_dimension = self.params["embedding"]["embedding_dimension"].value
        moire_embedding = self.params["embedding"]["moire_embedding"].value

        array = input_array.data
        arrays = [array]

        # Generating delayed versions
        for i in range(1, embedding_dimension):
            delayed = np.roll(array, shift=i * delay, axis=0)
            arrays.append(delayed)

        if moire_embedding:
            embedded_array = np.stack(arrays, axis=-1)
        else:
            embedded_array = np.array(arrays)

        if "dim0" in input_array.meta["channels"]:
            input_array.meta["channels"]["dim1"] = input_array.meta["channels"]["dim0"]
            del input_array.meta["channels"]["dim0"]

        return {"embedded_array": (embedded_array, input_array.meta)}
