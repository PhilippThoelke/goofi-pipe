from abc import ABC, abstractmethod
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np

from goofi.data import Data, DataType
from goofi.message import Message, MessageType


class DataViewer(ABC):
    def __init__(self, dtype: DataType, content_window: int) -> None:
        self.dtype = dtype
        self.content_window = content_window

    @abstractmethod
    def update(self, data: Data) -> None:
        pass

    def __call__(self, msg: Message) -> Any:
        if not msg.type == MessageType.DATA:
            raise ValueError(f"Expected message type DATA, got {msg.type}.")
        if not msg.content["data"].dtype == self.dtype:
            raise ValueError(f"Expected data type {self.dtype}, got {msg.content['data'].dtype}.")
        # update the data viewer
        self.update(msg.content["data"])


class ArrayViewer(DataViewer):
    def __init__(self, content_window: int) -> None:
        super().__init__(DataType.ARRAY, content_window)

        self.vmin = None
        self.vmax = None
        self.margin = 0.1
        self.shrinking = 0.01

        size = dpg.get_item_user_data(content_window)
        if size is None:
            raise RuntimeError("Expected content window to have user data with size.")

        # create plot
        self.plot = dpg.add_plot(
            width=size[0],
            height=size[1],
            parent=self.content_window,
            no_menus=True,
            no_box_select=True,
            no_mouse_pos=True,
            no_highlight=True,
            no_child=True,
        )

        # add x and y axis
        axis_kwargs = dict(parent=self.plot, no_gridlines=True, no_tick_marks=True, lock_min=True, lock_max=True)
        self.xax = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True, **axis_kwargs)
        self.yax = dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels=False, **axis_kwargs)

        # container for DPG line series
        self.line_series = []

    def update(self, data: Data) -> None:
        """
        This function handles drawing numerical data to a plot. Array shapes are handled as follows:
        - 0D (single number): the data is drawn as a horizontal line.
        - 1D: the data is drawn as a simple line plot.
        - 2D: the array gets interpreted as (n_channels, n_samples), and each channel is drawn as a line plot.

        ### Parameters
        `data` : Data
            The data message.
        """
        # convert data to numpy array and copy to C order (otherwise DPG will crash for some arrays)
        array = np.squeeze(data.data).copy(order="C")

        if self.vmin is not None and self.vmax is not None:
            # apply shrinking to vmin and vmax
            self.vmin = self.vmin * (1 - self.shrinking) + self.vmax * self.shrinking
            self.vmax = self.vmax * (1 - self.shrinking) + self.vmin * self.shrinking

        # update vmin and vmax
        if self.vmin is None or np.min(array) < self.vmin:
            self.vmin = np.nanmin(array)
        if self.vmax is None or np.max(array) > self.vmax:
            self.vmax = np.nanmax(array)

        if array.ndim == 0:
            # TODO: draw 0D in a better way
            # extend value to have at least 2 elements
            array = np.array([array, array])

        if array.ndim == 1:
            # remove extra data series
            while len(self.line_series) > 1:
                dpg.delete_item(self.line_series.pop())

            xs = np.arange(array.shape[0])
            if len(self.line_series) == 0:
                # add new data series
                self.line_series.append(dpg.add_line_series(xs, array, parent=self.yax))
            else:
                # update existing data series
                dpg.set_value(self.line_series[0], [xs, array])
        elif array.ndim == 2:
            # remove extra data series
            while len(self.line_series) > array.shape[0]:
                dpg.delete_item(self.line_series.pop())

            # add new data series
            xs = np.arange(array.shape[1])

            # iterate over channels (first dimension)
            for i in range(array.shape[0]):
                if len(self.line_series) == i:
                    # add new data series
                    self.line_series.append(dpg.add_line_series(xs, array[i], parent=self.yax))
                else:
                    # update existing data series
                    dpg.set_value(self.line_series[i], [xs, array[i]])
        else:
            raise NotImplementedError("TODO: plot higher-dimensional data")

        # autoscale x and y-axis limits
        dpg.set_axis_limits(self.xax, xs.min(), xs.max())
        dpg.set_axis_limits(self.yax, self.vmin - abs(self.vmax) * self.margin, self.vmax + abs(self.vmax) * self.margin)


class StringViewer(DataViewer):
    def __init__(self, content_window: int) -> None:
        super().__init__(DataType.STRING, content_window)

        # create text item
        self.text = dpg.add_text("", parent=self.content_window)

    def update(self, data: Data) -> None:
        """
        This function handles drawing string data to a text item.

        ### Parameters
        `data` : Data
            The data message.
        """
        dpg.set_value(self.text, data.data)


class TableViewer(DataViewer):
    def __init__(self, content_window: int) -> None:
        super().__init__(DataType.TABLE, content_window)

        # create table
        self.table = dpg.add_table(parent=self.content_window, header_row=False, policy=dpg.mvTable_SizingFixedFit)
        dpg.add_table_column(parent=self.table)
        dpg.add_table_column(parent=self.table)

        # row container
        self.rows = []

    def update(self, data: Data) -> None:
        """
        This function handles drawing table data to a table item.

        ### Parameters
        `data` : Data
            The data message.
        """
        # remove extra rows
        while len(self.rows) > len(data.data):
            dpg.delete_item(self.rows.pop())
        # add missing rows
        while len(self.rows) < len(data.data):
            with dpg.table_row(parent=self.table) as row:
                key = dpg.add_text("")
                val = dpg.add_text("")
                dpg.set_item_user_data(row, (key, val))
                # store row to populate later
                self.rows.append(row)

        # populate rows with data
        for row, (key, val) in zip(self.rows, data.data.items()):
            # truncate value text
            val = str(val.data)
            if len(val) > 20:
                val = val[:20] + "..."

            # update key and value cells
            key_cell, val_cell = dpg.get_item_user_data(row)
            dpg.set_value(key_cell, key)
            dpg.set_value(val_cell, val)


DTYPE_VIEWER_MAP = {
    DataType.ARRAY: ArrayViewer,
    DataType.STRING: StringViewer,
    DataType.TABLE: TableViewer,
}
