from goofi.data import DataType
from goofi.node import Node
from goofi.params import StringParam


class ConstantString(Node):
    def config_params():
        return {
            "constant": {"value": StringParam("default_value")},
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.STRING}

    def process(self):
        return {"out": (self.params["constant"]["value"].value, {})}
