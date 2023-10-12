from goofi.data import Data, DataType
from goofi.node import Node


class Table(Node):
    def config_input_slots():
        return {"base": DataType.TABLE, "new_entry": DataType.ARRAY}

    def config_output_slots():
        return {"table": DataType.TABLE}

    def config_params():
        return {"table": {"new_entry_key": "key"}}

    def process(self, base: Data, new_entry: Data):
        if base is None:
            # if no base is given, use an empty table
            base = Data(DataType.TABLE, DataType.TABLE.empty(), {})

        if new_entry is None:
            # if no new entry is given, return the base table
            return {"table": (base.data, base.meta)}

        # add the new entry to the base table
        assert len(self.params.table.new_entry_key.value) > 0, "New Entry Key cannot be empty."
        base.data[self.params.table.new_entry_key.value] = new_entry
        return {"table": (base.data, base.meta)}
