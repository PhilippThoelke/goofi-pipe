import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class ConstantTable(Node):

    def config_output_slots():
        return {"table": DataType.TABLE}

    def config_params():
        return {
            "param1": {
                "name": StringParam("key1", doc="Name of the parameter"),
                "value": StringParam("", doc="Value of the parameter"),
                "data_type": StringParam("ARRAY", options=["ARRAY", "STRING"], doc="Data type of the parameter"),
            },
            "param2": {
                "name": StringParam("key2", doc="Name of the parameter"),
                "value": StringParam("", doc="Value of the parameter"),
                "data_type": StringParam("ARRAY", options=["ARRAY", "STRING"], doc="Data type of the parameter"),
            },
            "param3": {
                "name": StringParam("key3", doc="Name of the parameter"),
                "value": StringParam("", doc="Value of the parameter"),
                "data_type": StringParam("ARRAY", options=["ARRAY", "STRING"], doc="Data type of the parameter"),
            },
            "param4": {
                "name": StringParam("key4", doc="Name of the parameter"),
                "value": StringParam("", doc="Value of the parameter"),
                "data_type": StringParam("ARRAY", options=["ARRAY", "STRING"], doc="Data type of the parameter"),
            },
            "param5": {
                "name": StringParam("key5", doc="Name of the parameter"),
                "value": StringParam("", doc="Value of the parameter"),
                "data_type": StringParam("ARRAY", options=["ARRAY", "STRING"], doc="Data type of the parameter"),
            },
            "common": {"autotrigger": True},
        }

    def process(self):
        table = {}
        for i in range(1, 6):
            key = self.params[f"param{i}"].name.value
            data_type = self.params[f"param{i}"].data_type.value
            value = self.params[f"param{i}"].value.value

            if value == "":
                continue

            if data_type == "ARRAY":
                table[key] = Data(DataType.ARRAY, np.fromstring(value, sep=","), {})
            elif data_type == "STRING":
                table[key] = Data(DataType.STRING, str(value), {})
        return {"table": (table, {})}
