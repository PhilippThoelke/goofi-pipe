import numpy as np
from mne import pick_channels

from goofi.data import Data, DataType
from goofi.node import Node


class Select(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"select": {"axis": 0, "include": "", "exclude": ""}}

    def process(self, data: Data):
        if data is None:
            return None

        axis = self.params.select.axis.value
        assert f"dim{axis}" in data.meta, f"Missing axis {axis} "

        chs = data.meta[f"dim{axis}"]

        include = self.params.select.include.value.split(",") or []
        include = [ch.strip() for ch in include if len(ch.strip()) > 0]
        exclude = self.params.select.exclude.value.split(",") or []
        exclude = [ch.strip() for ch in exclude if len(ch.strip()) > 0]

        idxs = pick_channels(chs, include=include, exclude=exclude, ordered=False)

        selected = np.squeeze(data.data[idxs])
        data.meta[f"dim{axis}"] = [chs[i] for i in idxs]

        return {"out": (selected, data.meta)}
