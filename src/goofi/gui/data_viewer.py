from abc import ABC, abstractmethod
from typing import Any, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from goofi.data import Data, DataType
from goofi.message import Message, MessageType


class ViewerContainer:
    def __init__(self, dtype: DataType, content_window: int) -> None:
        self.dtype = dtype
        self.content_window = content_window

        self.viewer_idx = 0
        self.viewer = DTYPE_VIEWER_MAP[self.dtype][self.viewer_idx](self.content_window)
        self.viewer.set_size()

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=0, callback=self.clicked)
            dpg.add_mouse_wheel_handler(callback=self.wheel)

    def clicked(self, _1: int, _2: Any) -> None:
        """Content window click handler. If the content window is ctrl+clicked, the viewer is switched."""
        try:
            state = dpg.get_item_state(self.content_window)
        except SystemError:
            # failed to retrieve state, likely because the window was closed
            return

        if not "hovered" in state or not state["hovered"]:
            # window is not hovered
            return

        if dpg.is_key_down(dpg.mvKey_Control):
            # ctrl+click: switch to next viewer
            self.next_viewer()

    def wheel(self, _1: int, value: Any) -> None:
        """Content window mouse wheel handler. Increases or deacreases the size of the viewer."""
        try:
            state = dpg.get_item_state(self.content_window)
        except SystemError:
            # failed to retrieve state, likely because the window was closed
            return

        if not "hovered" in state or not state["hovered"]:
            # window is not hovered
            return

        min_size = dpg.get_item_user_data(self.content_window)

        header = dpg.get_item_parent(self.content_window)
        window = dpg.get_item_parent(header)

        width, height = dpg.get_item_width(self.content_window), dpg.get_item_height(self.content_window)
        width = max(min_size[0], width + value * 10)
        height = max(min_size[1], height + value * 10)

        dpg.set_item_width(self.content_window, width)
        dpg.set_item_height(self.content_window, height)

        header_height = dpg.get_item_state(header)["rect_size"][1]
        if header_height == 0:
            # this happens when the header is off screen, set to default height
            header_height = 15

        dpg.set_item_width(window, width)
        dpg.set_item_height(window, height + header_height + 4)  # magic + 4 to avoid scroll bar

        self.viewer.set_size()

    def next_viewer(self) -> None:
        """Switch to the next viewer."""
        dpg.delete_item(self.content_window, children_only=True)
        self.viewer_idx = (self.viewer_idx + 1) % len(DTYPE_VIEWER_MAP[self.dtype])
        self.viewer = DTYPE_VIEWER_MAP[self.dtype][self.viewer_idx](self.content_window)
        self.viewer.set_size()

    def __call__(self, msg: Message) -> Any:
        if not msg.type == MessageType.DATA:
            raise ValueError(f"Expected message type DATA, got {msg.type}.")
        if not msg.content["data"].dtype == self.dtype:
            raise ValueError(f"Expected data type {self.dtype}, got {msg.content['data'].dtype}.")

        try:
            # update the data viewer
            self.viewer.update(msg.content["data"])
        except UnsupportedViewerError as e:
            self.next_viewer()
        except Exception as e:
            print(f"Error while updating data viewer: {e}")


class DataViewer(ABC):
    def __init__(self, dtype: DataType, content_window: int) -> None:
        self.dtype = dtype
        self.content_window = content_window

    @abstractmethod
    def update(self, data: Data) -> None:
        pass

    def set_size(self) -> None:
        pass


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
            raise UnsupportedViewerError(f"Cannot handle array with {array.ndim} dimensions.")

        # autoscale x and y-axis limits
        dpg.set_axis_limits(self.xax, xs.min(), xs.max())
        dpg.set_axis_limits(self.yax, self.vmin - abs(self.vmax) * self.margin, self.vmax + abs(self.vmax) * self.margin)

    def set_size(self) -> None:
        """This function sets the size of the plot."""
        dpg.set_item_width(self.plot, dpg.get_item_width(self.content_window))
        dpg.set_item_height(self.plot, dpg.get_item_height(self.content_window))


class ImageViewer(DataViewer):
    def __init__(self, content_window: int, max_res: int = 512) -> None:
        super().__init__(DataType.ARRAY, content_window)

        config = dpg.get_item_configuration(content_window)
        res = (config["width"], config["height"])

        # set resolution to max_res
        if res[0] != max_res and res[1] != max_res:
            scale = max_res / max(res)
            res = (int(res[0] * scale), int(res[1] * scale))

        # initialize texture
        with dpg.texture_registry():
            self.texture = dpg.add_dynamic_texture(
                width=res[0], height=res[1], default_value=[0.0 for _ in range(res[0] * res[1] * 4)]
            )
        self.image = dpg.add_image(self.texture, parent=self.content_window)

    def update(self, data: Data) -> None:
        """
        This function handles drawing image data to an image item.

        ### Parameters
        `data` : Data
            The data message.
        """
        # convert data to numpy array and copy to C order (otherwise DPG will crash for some arrays)
        array = np.squeeze(data.data).copy(order="C")

        if array.ndim > 3:
            raise UnsupportedViewerError(f"Cannot handle array with {array.ndim} dimensions.")
        while array.ndim < 3:
            array = array[..., None]

        # make sure we have 4 channels
        if array.shape[2] > 4:
            raise NotImplementedError(f"Cannot handle array with {array.shape[2]} channels.")
        elif array.shape[2] == 3:
            array = np.concatenate([array, np.ones((*array.shape[:2], 1))], axis=2)
        elif array.shape[2] == 2:
            array = np.concatenate([array, np.ones((*array.shape[:2], 2))], axis=2)
        elif array.shape[2] == 1:
            array = np.concatenate([array] * 3 + [np.ones((*array.shape[:2], 1))], axis=2)

        # clamp values to [0, 1]
        array = np.clip(array, 0, 1)

        tex_config = dpg.get_item_configuration(self.texture)
        if tex_config["height"] != array.shape[1] or tex_config["width"] != array.shape[0]:
            # resize array to fit texture
            array = cv2.resize(array, (tex_config["width"], tex_config["height"]), interpolation=cv2.INTER_NEAREST)

        # update texture
        dpg.set_value(self.texture, array.flatten())

    def set_size(self) -> None:
        """This function sets the size of the image."""
        dpg.set_item_width(self.image, dpg.get_item_width(self.content_window))
        dpg.set_item_height(self.image, dpg.get_item_height(self.content_window))


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
    DataType.ARRAY: [ArrayViewer, ImageViewer],
    DataType.STRING: [StringViewer],
    DataType.TABLE: [TableViewer],
}


class UnsupportedViewerError(Exception):
    pass
