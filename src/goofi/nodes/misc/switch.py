from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam

class Switch(Node):
    def config_input_slots():
        return {
            "selector": DataType.ARRAY,  # Input 1: Selector array
            "array1": DataType.ARRAY,   # Input 2: First array input
            "array2": DataType.ARRAY,   # Input 3: Second array input
            "string1": DataType.STRING, # Input 4: First string input
            "string2": DataType.STRING, # Input 5: Second string input
        }

    def config_output_slots():
        return {
            "array_out": DataType.ARRAY,   # Output 1: Array output
            "string_out": DataType.STRING, # Output 2: String output
        }

    def config_params():
        return { "settings": {
            "mode": StringParam("array", ["array", "string"]), # Dropdown to choose mode
        }
        }

    def process(self, selector: Data, array1: Data, array2: Data, string1: Data, string2: Data):
        if selector is None or selector.data is None:
            return {"array_out": None, "string_out": None}

        selector_value = int(selector.data[0])  # Assume selector is a single-element array
        mode = self.params["settings"]["mode"].value

        if mode == "array":
            if selector_value == 1:
                return {"array_out": (array1.data, {}), "string_out": None}
            elif selector_value == 2:
                return {"array_out": (array2.data, {}), "string_out": None}
            else:
                return {"array_out": None, "string_out": None}

        elif mode == "string":
            if selector_value == 1:
                return {"array_out": None, "string_out": (string1.data, {})}
            elif selector_value == 2:
                return {"array_out": None, "string_out": (string2.data, {})}
            else:
                return {"array_out": None, "string_out": None}

        # If none of the conditions match, return None for both outputs
        return {"array_out": None, "string_out": None}
