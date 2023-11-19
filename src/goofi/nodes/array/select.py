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

        include = self.params.select.include.value.split(",") or []
        include = [ch.strip() for ch in include if len(ch.strip()) > 0]
        exclude = self.params.select.exclude.value.split(",") or []
        exclude = [ch.strip() for ch in exclude if len(ch.strip()) > 0]

        if f"dim{axis}" in data.meta["channels"]:
            # use channel names from metadata
            chs = data.meta["channels"][f"dim{axis}"]
            idxs = self.pick_channels(chs, include=include, exclude=exclude, ordered=False)
        else:
            # no channel names for this axis, use indices
            chs = [str(i) for i in range(data.data.shape[axis])]

            if include and ":" in include[0]:
                # slice notation
                if len(include) > 1:
                    raise ValueError("Only one slice can be selected at a time.")
                if len(exclude) > 0:
                    raise ValueError("Excluding channels is not supported with slice notation.")
                slice_parts = include[0].split(":")
                start, stop, step = (int(x) if x else None for x in slice_parts + [None] * (3 - len(slice_parts)))
                idxs = chs[slice(start, stop, step)]
            else:
                if len(include) == 0:
                    # include all channels
                    include = chs

                # convert to indices and shift negative indices
                include = [int(ch) if int(ch) >= 0 else data.data.shape[axis] + int(ch) for ch in include]
                exclude = [int(ch) if int(ch) >= 0 else data.data.shape[axis] + int(ch) for ch in exclude]
                idxs = [i for i in range(len(chs)) if i in include and i not in exclude]

        if len(idxs) == 0:
            raise ValueError("No channels matched the selection.")

        # select channels from data
        selected = np.take(data.data, idxs, axis=axis)

        # update channel names if present
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
