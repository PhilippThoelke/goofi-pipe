import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam
from copy import deepcopy


class Operation(Node):
    def config_input_slots():
        return {"a": DataType.ARRAY, "b": DataType.ARRAY}

    def config_params():
        return {"operation": {"operation": StringParam("add", options=["add", "subtract", "multiply", "divide", "matmul"])}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self, a: Data, b: Data):
        if a is None or b is None:
            return None

        operation = self.params.operation.operation.value
        if operation == "matmul":
            new_meta = deepcopy(a.meta)
            # matmul is n x m * m x p -> n x p
            if "dim0" in a.meta["channels"]:
                new_meta["channels"]["dim0"] = a.meta["channels"]["dim0"]
            elif "dim0" in new_meta["channels"]:
                del new_meta["channels"]["dim0"]

            if "dim1" in b.meta["channels"]:
                new_meta["channels"]["dim1"] = b.meta["channels"]["dim1"]
            elif "dim1" in new_meta["channels"]:
                del new_meta["channels"]["dim1"]
        else:
            if "channels" in a.meta and "channels" in b.meta:
                if a.meta["channels"] != b.meta["channels"]:
                    print("Channels are not the same, metadata from a is used")
                new_meta = deepcopy(a.meta)

        if operation == "add":
            result = a.data + b.data
        elif operation == "subtract":
            result = a.data - b.data
        elif operation == "multiply":
            result = a.data * b.data
        elif operation == "divide":
            result = a.data / b.data
        elif operation == "matmul":
            result = np.dot(a.data, b.data)
        else:
            raise ValueError(f"Invalid operation: {operation}")

        return {"out": (result, new_meta)}
