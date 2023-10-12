import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Select(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"select": {"axis": 0, "include": "", "exclude": ""}}

    def setup(self):
        from mne import pick_channels

        self.pick_channels = pick_channels

    def process(self, data: Data):
        if data is None:
            return None

        axis = self.params.select.axis.value
        if axis < 0:
            axis = data.data.ndim + axis

        if f"dim{axis}" in data.meta["channels"]:
            # use channel names from metadata
            chs = data.meta["channels"][f"dim{axis}"]
        else:
            # no channel names for this axis, use indices
            chs = [str(i) for i in range(data.data.shape[axis])]

        include = self.params.select.include.value.split(",") or []
        include = [ch.strip() for ch in include if len(ch.strip()) > 0]
        exclude = self.params.select.exclude.value.split(",") or []
        exclude = [ch.strip() for ch in exclude if len(ch.strip()) > 0]

        idxs = self.pick_channels(chs, include=include, exclude=exclude, ordered=False)

        if len(idxs) == 0:
            raise ValueError("No channels matched the selection.")

        selected = np.take(data.data, idxs, axis=axis)
        if f"dim{axis}" in data.meta["channels"]:
            data.meta["channels"][f"dim{axis}"] = [ch for i, ch in enumerate(data.meta["channels"][f"dim{axis}"]) if i in idxs]

        if len(idxs) == 1:
            # remove axis if only one channel is selected
            selected = np.squeeze(selected, axis=axis)
            if f"dim{axis}" in data.meta["channels"]:
                del data.meta["channels"][f"dim{axis}"]

            for i in range(axis, selected.ndim + 1):
                if f"dim{i+1}" in data.meta["channels"]:
                    data.meta["channels"][f"dim{i}"] = data.meta["channels"][f"dim{i+1}"]
                    del data.meta["channels"][f"dim{i+1}"]

        return {"out": (selected, data.meta)}
