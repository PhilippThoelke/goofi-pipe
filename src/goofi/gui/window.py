import logging
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple

import dearpygui.dearpygui as dpg
import numpy as np

from goofi.data import DataType
from goofi.gui.events import KEY_MAP
from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef

logger = logging.getLogger(__name__)

DTYPE_SHAPE_MAP = {
    DataType.ARRAY: dpg.mvNode_PinShape_CircleFilled,
    DataType.STRING: dpg.mvNode_PinShape_TriangleFilled,
}


def running(func):
    def wrapper(*args, **kwargs):
        # make sure DearPyGui is running
        n_tries = 5
        while not dpg.is_dearpygui_running() and n_tries > 0:
            time.sleep(0.1)
            n_tries -= 1
        if n_tries == 0:
            raise RuntimeError("DearPyGui is not running.")
        return func(*args, **kwargs)

    return wrapper


@dataclass
class GUINode:
    item: int
    input_slots: Dict[str, int]
    output_slots: Dict[str, int]


def draw_data(node: NodeRef, data: Message, plot: List[int], minmax: List[int], margin: float = 0.1, shrinking: float = 0.001):
    """
    Draw data on a plot.

    ### Parameters
    `node` : NodeRef
        The node reference.
    `data` : Message
        The data message.
    `plot` : List[int]
        A list of at least two item tags: x-axis, y-axis, and optional data series tags.
    `minmax` : List[int]
        A list of two values: minimum and maximum values of the data.
    """
    dtype = data.content["data"].dtype
    value = np.squeeze(data.content["data"].data)

    if dtype == DataType.ARRAY:
        if minmax[0] is not None and minmax[1] is not None:
            # apply shrinking to minmax
            minmax[:] = minmax[0] * (1 - shrinking) + shrinking * minmax[1], minmax[1] * (1 - shrinking) + shrinking * minmax[0]
        # update minmax
        if minmax[0] is None or np.min(value) < minmax[0]:
            minmax[0] = np.min(value)
        if minmax[1] is None or np.max(value) > minmax[1]:
            minmax[1] = np.max(value)

        if value.ndim == 0:
            # extend value to have at least 2 elements
            value = np.array([value, value])
        if value.ndim == 1:
            # remove extra data series
            while len(plot) > 3:
                dpg.delete_item(plot.pop())

            xs = np.arange(value.shape[0])
            if len(plot) == 2:
                # add new data series
                plot.append(dpg.add_line_series(xs, value, parent=plot[1]))
            else:
                # update existing data series
                dpg.set_value(plot[2], [xs, value])

            # autoscale x-axis
            dpg.set_axis_limits(plot[0], xs.min(), xs.max())
            # set y-axis limits
            dpg.set_axis_limits(plot[1], minmax[0] - minmax[0] * margin, minmax[1] + minmax[1] * margin)
        elif value.ndim == 2:
            # remove extra data series
            while len(plot) > value.shape[0] + 2:
                dpg.delete_item(plot.pop())

            # add new data series
            xs = np.arange(value.shape[1])
            for i in range(value.shape[0]):
                if len(plot) == i + 2:
                    # add new data series
                    plot.append(dpg.add_line_series(xs, value[i], parent=plot[1]))
                else:
                    # update existing data series
                    dpg.set_value(plot[i + 2], [xs, value[i]])

            # autoscale x-axis
            dpg.set_axis_limits(plot[0], xs.min(), xs.max())
            # set y-axis limits
            dpg.set_axis_limits(plot[1], minmax[0] - minmax[0] * margin, minmax[1] + minmax[1] * margin)
        else:
            raise NotImplementedError("TODO: plot higher-dimensional data")
    else:
        raise NotImplementedError("TODO: plot non-array data")


class Window:
    """
    The graphical user interface window, implemented as a threaded singleton. The window is created
    when Window() is first called. Subsequent calls to Window() will return the same instance. The
    window thread is a daemon thread, so it will be terminated when the main thread exits, or when
    Window().close() is called.

    ### Parameters
    `manager` : Manager
        The manager instance.
    """

    _instance = None

    def __new__(cls, manager=None):
        if cls._instance is None:
            # instantiate the window thread
            logger.info("Starting graphical user interface.")
            cls._instance = super(Window, cls).__new__(cls)
            threading.Thread(target=cls._instance._initialize, args=(manager,), daemon=True).start()
        return cls._instance

    @running
    def add_node(self, node_name: str, node: NodeRef) -> None:
        with dpg.node(parent=self.node_editor, label=node_name) as node_id:
            ############### input slots ###############
            in_slots = {}
            for name, dtype in node.input_slots.items():
                slot_kwargs = dict(label=name, attribute_type=dpg.mvNode_Attr_Input, shape=DTYPE_SHAPE_MAP[dtype])
                with dpg.node_attribute(**slot_kwargs) as attr:
                    in_slots[name] = attr
                    dpg.add_text(name)

            ############### output slots ###############
            out_slots = {}
            for name, dtype in node.output_slots.items():
                slot_kwargs = dict(label=name, attribute_type=dpg.mvNode_Attr_Output, shape=DTYPE_SHAPE_MAP[dtype])
                with dpg.node_attribute(**slot_kwargs) as attr:
                    out_slots[name] = attr

                    # create plot
                    plot = dpg.add_plot(
                        label=name,
                        width=175,
                        height=125,
                        no_menus=True,
                        no_box_select=True,
                        no_mouse_pos=True,
                        no_highlight=True,
                        no_child=True,
                    )
                    # x-axis
                    xax = dpg.add_plot_axis(
                        dpg.mvXAxis,
                        parent=plot,
                        no_gridlines=True,
                        no_tick_marks=True,
                        # no_tick_labels=True,
                        lock_min=True,
                        lock_max=True,
                    )
                    # y-axis
                    yax = dpg.add_plot_axis(
                        dpg.mvYAxis,
                        parent=plot,
                        no_gridlines=True,
                        no_tick_marks=True,
                        # no_tick_labels=True,
                        lock_min=True,
                        lock_max=True,
                    )
                    node.set_message_handler(MessageType.DATA, partial(draw_data, plot=[xax, yax], minmax=[None, None]))

            # TODO: register PROCESSING_ERROR message handler and display error messages
            node.set_message_handler(MessageType.PROCESSING_ERROR, lambda node, data: logger.error(data.content["error"]))

            # add node to node list
            self.nodes[node_name] = GUINode(node_id, in_slots, out_slots)

    @running
    def remove_node(self, name: str) -> None:
        self._remove_node(self.nodes[name], notify_manager=False)

    @running
    def _remove_node(self, item: int, notify_manager: bool = True) -> None:
        name = dpg.get_item_label(item)

        # remove links associated with node
        for link in list(self.links.keys()):
            if link[0] == name or link[1] == name:
                self._remove_link(self.links[link], notify_manager=False)

        # unregister node
        del self.nodes[name]

        # delete node from gui
        dpg.delete_item(item)

        if notify_manager:
            # remove node from manager
            manager = dpg.get_item_user_data(self.window)
            manager.remove_node(name, notify_gui=False)

    @running
    def add_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str) -> None:
        # get slot IDs
        slot1 = self.nodes[node_out].output_slots[slot_out]
        slot2 = self.nodes[node_in].input_slots[slot_in]
        # add link
        self._add_link(self.node_editor, (slot1, slot2), notify_manager=False)

    @running
    def _add_link(self, sender: int, items: Tuple[int, int], notify_manager: bool = True) -> None:
        # get slot names
        slot1 = dpg.get_item_label(items[0])
        slot2 = dpg.get_item_label(items[1])
        # get node names
        node1 = dpg.get_item_label(dpg.get_item_parent(items[0]))
        node2 = dpg.get_item_label(dpg.get_item_parent(items[1]))

        # remove link first if the input slot already has a link
        for link in list(self.links.keys()):
            if link[1] == node2 and link[3] == slot2:
                self._remove_link(self.links[link], notify_manager=True)

        # register link
        self.links[(node1, node2, slot1, slot2)] = dpg.add_node_link(items[0], items[1], parent=sender)

        if notify_manager:
            # add link to manager
            manager = dpg.get_item_user_data(self.window)
            manager.add_link(node1, node2, slot1, slot2, notify_gui=False)

    def remove_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str) -> None:
        self._remove_link(self.links[(node_out, node_in, slot_out, slot_in)], notify_manager=False)

    @running
    def _remove_link(self, item: int, notify_manager: bool = True) -> None:
        conf = dpg.get_item_configuration(item)
        # get slot names
        slot1, slot2 = dpg.get_item_label(conf["attr_1"]), dpg.get_item_label(conf["attr_2"])
        # get node names
        node1 = dpg.get_item_label(dpg.get_item_parent(conf["attr_1"]))
        node2 = dpg.get_item_label(dpg.get_item_parent(conf["attr_2"]))

        if notify_manager:
            # remove link from manager
            manager = dpg.get_item_user_data(self.window)
            manager.remove_link(node1, node2, slot1, slot2, notify_gui=False)

        # unregister link
        del self.links[(node1, node2, slot1, slot2)]
        # delete link from gui
        dpg.delete_item(item)

    def link_callback(self, sender, data):
        self._add_link(sender, data)

    def delink_callback(self, _, data):
        self._remove_link(data)

    def key_press_callback(self, _, data):
        if data in KEY_MAP:
            KEY_MAP[data](self)

    def resize_callback(self, _, data):
        dpg.configure_item(self.window, width=data[0], height=data[1])

    def exit_callback(self, value):
        # TODO: open a popup to confirm exit, if state was modified after save
        self.terminate()

    def _initialize(self, manager, width=1280, height=720):
        dpg.create_context()

        # initialize dicts to map names to dpg items
        self.nodes = {}
        self.links = {}

        # create window
        self.window = dpg.add_window(
            label="window",
            width=width,
            height=height,
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            no_close=True,
            no_collapse=True,
        )
        # create node editor
        self.node_editor = dpg.add_node_editor(
            parent=self.window, callback=self.link_callback, delink_callback=self.delink_callback
        )

        # store manager in the window's user data
        dpg.set_item_user_data(self.window, manager)

        # register key-press handler
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self.key_press_callback)

        # register viewport resize handler
        dpg.set_viewport_resize_callback(self.resize_callback)

        # register exit handler
        dpg.set_exit_callback(self.exit_callback)

        # start DearPyGui
        dpg.create_viewport(title="goofi-pipe", width=width, height=height, disable_close=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

        logger.info("Graphical user interface closed.")

        # terminate manager
        manager.terminate(notify_gui=False)

    def terminate(self):
        """Stop DearPyGui and close the window."""
        dpg.stop_dearpygui()
