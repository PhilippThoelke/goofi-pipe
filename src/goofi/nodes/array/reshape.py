from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Reshape(Node):
    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"reshape": {"shape": StringParam("-1")}}

    def process(self, array: Data):
        if array is None:
            return None

        shape = list(map(int, self.params.reshape.shape.value.split(",")))
        result = array.data.reshape(shape)

        # TODO: properly handle channel names
        del array.meta["channels"]
        return {"out": (result, array.meta)}
