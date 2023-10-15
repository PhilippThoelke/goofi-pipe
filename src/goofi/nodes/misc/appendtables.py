from goofi.data import Data, DataType
from goofi.node import Node


class AppendTables(Node):
    def config_input_slots():
        return {"table1": DataType.TABLE, "table2": DataType.TABLE}

    def config_output_slots():
        return {"output_table": DataType.TABLE}

    def config_params():
        return {}

    def process(self, table1: Data, table2: Data):
        if table1 is None or table2 is None:
            return None

        # Combine the two tables' data dictionaries
        combined_data = {**table1.data, **table2.data}

        # For simplicity, I'm just combining the meta data dictionaries here,
        # but you might want to handle meta data merging more carefully.
        combined_meta = {**table1.meta, **table2.meta}

        return {"output_table": (combined_data, combined_meta)}
