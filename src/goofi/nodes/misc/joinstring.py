from goofi.data import DataType
from goofi.node import Node


class JoinString(Node):
    def config_input_slots():
        return {
            "string1": DataType.STRING,
            "string2": DataType.STRING,
            "string3": DataType.STRING,
            "string4": DataType.STRING,
            "string5": DataType.STRING,
        }

    def config_output_slots():
        return {"output": DataType.STRING}

    def config_params():
        return {
            "join": {
                "separator": ", ",
                "string1": True,
                "string2": True,
                "string3": True,
                "string4": True,
                "string5": True,
            }
        }

    def process(self, **inputs):
        separator = self.params.join.separator.value
        selected_inputs = filter(lambda item: self.params.join[item[0]].value and item[1] is not None, inputs.items())
        output = separator.join([item[1].data for item in selected_inputs])
        return {"output": (output, {})}
