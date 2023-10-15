from goofi.data import DataType, Data
from goofi.node import Node
from goofi.params import StringParam
import re


class FormatString(Node):
    def config_input_slots():
        slots = {}
        for i in range(1, 11):  # For 10 input strings
            slots[f"input_string_{i}"] = DataType.STRING
        return slots

    def config_output_slots():
        return {"output_string": DataType.STRING}

    def config_params():
        return {
            "pattern": {
                "key": StringParam(
                    "",
                    doc="When empty, will join strings with spaces. You can specify placeholders in the format using curly brackets",
                )
            }
        }

    def process(self, **input_strings):
        pattern = self.params["pattern"]["key"].value

        # Convert Data objects to their string representations
        for key, value in input_strings.items():
            if value is None or value.data is None:
                input_strings[key] = Data(dtype=DataType.STRING, data="", meta={})
            else:
                input_strings[key] = Data(dtype=DataType.STRING, data=value.data, meta=value.meta)

        # Handle the case with no pattern provided
        if not pattern:
            input_values = [value.data for value in input_strings.values()]
            while input_values and not input_values[-1]:
                input_values.pop()
            output = " ".join(input_values)
        else:
            # Replace all named placeholders
            for key, value in input_strings.items():
                pattern = pattern.replace(f"{{{key}}}", value.data)

            # Identify used keys in the pattern (after named replacement to catch any repeating named placeholders)
            used_keys = re.findall(r"{(input_string_\d+)}", pattern)

            # Create a list of unused values
            unused_values = [value.data for key, value in input_strings.items() if key not in used_keys]

            # Replace unnamed placeholders with unused values
            while unused_values and "{}" in pattern:
                pattern = pattern.replace("{}", unused_values.pop(0), 1)

            output = pattern

        return {"output_string": (output, {})}
