from goofi.data import Data, DataType
from goofi.node import Node


class TableNormalization(Node):
    def config_input_slots():
        return {"table": DataType.TABLE}

    def config_output_slots():
        return {"normalized": DataType.TABLE}

    def config_params():
        from goofi.nodes.signal.normalization import Normalization

        return Normalization.config_params()

    def setup(self):
        from goofi.nodes.signal.normalization import Normalization

        self.normalization = Normalization
        self.normalizers = {}

    def process(self, table: Data):
        params = self.params.serialize()

        result = {}
        for key, data in table.data.items():
            if key not in self.normalizers:
                # initialize a new normalizer for this key
                self.normalizers[key] = self.normalization.create_standalone()
                self.normalizers[key].setup()

            # update params of the normalizer
            self.normalizers[key].params.update(params)

            # process the data
            normalized = self.normalizers[key].process(data)["normalized"]
            result[key] = Data(data.dtype, *normalized)

        # make sure to "use" the reset trigger so it doesn't get stuck
        self.params.normalization.reset.value

        # delete normalizers for keys that are not in the table anymore
        self.normalizers = {key: norm for key, norm in self.normalizers.items() if key in table.data}

        return {"normalized": (result, table.meta)}
