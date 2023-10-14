import numpy as np
from copy import deepcopy
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Join(Node):
    def config_input_slots():
        return {"a": DataType.ARRAY, "b": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"join": {"method": StringParam("concatenate", options=["concatenate", "stack"]), "axis": 0}}

    def process(self, a: Data, b: Data):
        if a is None or b is None:
            return None

        if self.params.join.method.value == "concatenate":
            # concatenate a and b
            result = np.concatenate([a.data, b.data], axis=self.params.join.axis.value)
            
            if 'channels' in a.meta and 'channels' in b.meta:
                result_meta = deepcopy(a.meta)
                result_meta['channels']['dim0'] = a.meta['channels']['dim0'] + b.meta['channels']['dim0']
        elif self.params.join.method.value == "stack":
            # stack a and b
            result = np.stack([a.data, b.data], axis=self.params.join.axis.value)
            result_meta = deepcopy(a.meta)
        else:
            raise ValueError(f"Unknown join method {self.params.join.method.value}. Supported are 'concatenate' and 'stack'.")

        # TODO: properly combine metadata from both inputs
        # TODO: update metadata information after stack
        # TODO: check if inputs are compatible
        return {"out": (result, result_meta)}
