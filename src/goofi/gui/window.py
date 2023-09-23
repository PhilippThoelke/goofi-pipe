import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
import numpy as np

from goofi.data import DataType
from goofi.gui import events
from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef
from goofi.params import BoolParam, FloatParam, IntParam, Param, StringParam

DTYPE_SHAPE_MAP = {
    DataType.ARRAY: dpg.mvNode_PinShape_CircleFilled,
    DataType.STRING: dpg.mvNode_PinShape_TriangleFilled,
}

PARAM_WINDOW_WIDTH = 400


def format_name(name: str) -> str:
    """Format a name to be used as a node label."""
    return name.replace("_", " ").title()


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
    output_draw_handlers: Dict[str, int]
    node_ref: NodeRef


def handle_data(gui_node: GUINode, node: NodeRef, data: Message):
    """
    Handle a data message from a node. This function is registered as a message handler for the
    `MessageType.DATA` message type.

    ### Parameters
    `name` : str
        The name of the node.
    `node` : NodeRef
        The node reference.
    `data` : Message
        The data message.
    """
    try:
        gui_node.output_draw_handlers[data.content["slot_name"]](node, data)
    except ValueError as e:
        # TODO: add proper logging
        print(f"Output draw handler for slot {data.content['slot_name']} failed: {e}")


def draw_data(node: NodeRef, data: Message, plot: List[int], minmax: List[int], margin: float = 0.1, shrinking: float = 0.01):
    """
    This function handles drawing numerical data to a plot. Array shapes are handled as follows:
    - 0D (single number): the data is drawn as a horizontal line.
    - 1D: the data is drawn as a simple line plot.
    - 2D: the array gets interpreted as (n_channels, n_samples), and each channel is drawn as a line plot.

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
    value = np.squeeze(data.content["data"].data).copy(order="C")

    if dtype == DataType.ARRAY:
        if minmax[0] is not None and minmax[1] is not None:
            # apply shrinking to minmax
            minmax[:] = minmax[0] * (1 - shrinking) + shrinking * minmax[1], minmax[1] * (1 - shrinking) + shrinking * minmax[0]
        # update minmax
        if minmax[0] is None or np.min(value) < minmax[0]:
            minmax[0] = np.nanmin(value)
        if minmax[1] is None or np.max(value) > minmax[1]:
            minmax[1] = np.nanmax(value)

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
            dpg.set_axis_limits(plot[1], minmax[0] - abs(minmax[0]) * margin, minmax[1] + abs(minmax[1]) * margin)
        elif value.ndim == 2:
            # remove extra data series
            while len(plot) > value.shape[0] + 2:
                dpg.delete_item(plot.pop())

            # add new data series
            xs = np.arange(value.shape[1])

            # iterate over channels (first dimension)
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


def param_updated(_, value, user_data):
    """A callback function for updating a parameter value."""
    user_data[2].update_param(user_data[0], user_data[1], value)


def add_param(parent: int, group: str, name: str, param: Param, node: NodeRef) -> None:
    """
    Add a parameter to the GUI.

    ### Parameters
    `parent` : int
        The parent item.
    `group` : str
        The parameter group.
    `name` : str
        The parameter name.
    `param` : Param
        The parameter.
    `node` : NodeRef
        The node reference.
    """
    with dpg.group(horizontal=True, parent=parent):
        dpg.add_text(format_name(name))
        dpg.add_spacer(width=-1)

        if isinstance(param, BoolParam):
            # parameter is a bool
            if param.toggle:
                dpg.add_checkbox(default_value=param.value, callback=param_updated, user_data=(group, name, node))
            else:
                dpg.add_button(callback=param_updated, user_data=(group, name, node))
        elif isinstance(param, FloatParam):
            # parameter is a float
            dpg.add_slider_float(
                default_value=param.value,
                min_value=param.vmin,
                max_value=param.vmax,
                user_data=(group, name, node),
                callback=param_updated,
            )
        elif isinstance(param, IntParam):
            # parameter is an integer
            dpg.add_slider_int(
                default_value=param.value,
                min_value=param.vmin,
                max_value=param.vmax,
                callback=param_updated,
                user_data=(group, name, node),
            )
        elif isinstance(param, StringParam):
            # parameter is a string
            dpg.add_input_text(
                default_value=param.value,
                callback=param_updated,
                user_data=(group, name, node),
            )
        else:
            raise NotImplementedError(f"Parameter type {type(param)} not implemented.")


def add_output_slot(parent: int, name: str, closed: bool = False, size: Tuple[int, int] = (175, 100)) -> Tuple[int, int]:
    """
    Add an output slot to the GUI, which consists of a collapsable header and a plot.

    ### Parameters
    `parent` : int
        The parent item (the node attribute).
    `name` : str
        The name of the output slot.
    `closed` : bool
        Whether the headers should be closed initially.
    `size` : Tuple[int, int]
        The size of the plot.

    ### Returns
    `xax` : int
        The x-axis item.
    `yax` : int
        The y-axis item.
    """

    def header_callback(_1, _2, items: List[int]):
        """
        Event callback for the collapsable header, both for general clicks, and for toggle_open events.
        This is a workaround for the fact that toggle_close events don't exist. As toggle_open is always
        called after click, clicking on the header always closes it, so the toggle_open callback is used
        reopen it if the internal header state is open.

        ### Parameters
        `items` : List[int]
            A list of four items: a bool indicating to close the header, the header, the window, and the plot.
        """
        close, header, window, plot = items
        header_height = dpg.get_item_state(header)["rect_size"][1]

        if close:
            # header was closed, shrink window to header size
            dpg.set_item_height(window, header_height)
        else:
            # header was opened, expand window to header size plus plot size
            plot_height = dpg.get_item_state(plot)["rect_size"][1]
            dpg.set_item_height(window, header_height + plot_height + 4)  # magic + 4 to avoid scroll bar

    # NOTE: The user_data lists are used to pass data to the callbacks. The first element is a bool, which
    # is used to indicate whether the header was closed or opened. The remaining elements are the header,
    # window, and plot items. Due to the way DearPyGui registers callbacks we can not use partials here.
    user_data_open, user_data_close = [False], [True]
    # register handlers
    with dpg.item_handler_registry() as reg:
        dpg.add_item_clicked_handler(callback=header_callback, user_data=user_data_close)
        dpg.add_item_toggled_open_handler(callback=header_callback, user_data=user_data_open)

    # NOTE: we initially set the height to 1 to avoid the node reaching out of the window, which would break the
    # initial header_callback call as the header and plot sizes according to DearPyGui would be 0.
    # The child window is needed, because otherwise the header will expand to fill the entire node editor window.
    with dpg.child_window(width=size[0], height=1, parent=parent) as window:
        with dpg.collapsing_header(
            label=name, default_open=not closed, open_on_double_click=False, open_on_arrow=False, closable=False
        ) as header:
            # create plot
            plot = dpg.add_plot(
                width=size[0],
                height=size[1],
                no_menus=True,
                no_box_select=True,
                no_mouse_pos=True,
                no_highlight=True,
                no_child=True,
            )
            # add x and y axis
            axis_kwargs = dict(parent=plot, no_gridlines=True, no_tick_marks=True, lock_min=True, lock_max=True)
            xax = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True, **axis_kwargs)
            yax = dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels=False, **axis_kwargs)

            # store header, window and plot in user_data for the two callbacks
            user_data_open.extend([header, window, plot])
            user_data_close.extend([header, window, plot])
            # bind the two callbacks to the header
            dpg.bind_item_handler_registry(header, reg)

            # schedule the header callback to set up the window sizes
            initial_args = user_data_close if closed else user_data_open
            threading.Timer(0.1, header_callback, [None, None, initial_args]).start()
    return xax, yax


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
            # TODO: add proper logging
            print("Starting graphical user interface.")
            cls._instance = super(Window, cls).__new__(cls)
            threading.Thread(target=cls._instance._initialize, args=(manager,), daemon=True).start()
        return cls._instance

    @running
    def add_node(self, node_name: str, node: NodeRef) -> None:
        """
        Outward facing method to add a node to the GUI.

        ### Parameters
        `node_name` : str
            The name of the node.
        `node` : NodeRef
            The node reference.
        """
        pos = dpg.get_mouse_pos(local=False)
        with dpg.node(parent=self.node_editor, label=node_name, pos=pos) as node_id:
            ############### input slots ###############
            in_slots = {}
            for name, dtype in node.input_slots.items():
                slot_kwargs = dict(label=name, attribute_type=dpg.mvNode_Attr_Input, shape=DTYPE_SHAPE_MAP[dtype])
                with dpg.node_attribute(**slot_kwargs) as attr:
                    in_slots[name] = attr
                    dpg.add_text(name)

            ############### output slots ###############
            out_slots = {}
            output_draw_handlers = {}
            for name, dtype in node.output_slots.items():
                slot_kwargs = dict(label=name, attribute_type=dpg.mvNode_Attr_Output, shape=DTYPE_SHAPE_MAP[dtype])
                with dpg.node_attribute(**slot_kwargs) as attr:
                    out_slots[name] = attr
                    xax, yax = add_output_slot(attr, name, closed=len(node.output_slots) > 2)
                    output_draw_handlers[name] = partial(draw_data, plot=[xax, yax], minmax=[None, None])

            # TODO: register PROCESSING_ERROR message handler and display error messages
            node.set_message_handler(MessageType.PROCESSING_ERROR, lambda node, data: print(data.content["error"]))

            # add node to node list
            self.nodes[node_name] = GUINode(node_id, in_slots, out_slots, output_draw_handlers, node)
            # register node message handler
            node.set_message_handler(MessageType.DATA, partial(handle_data, self.nodes[node_name]))

    @running
    def remove_node(self, name: str) -> None:
        """
        Outward facing method to remove a node from the GUI.

        ### Parameters
        `name` : str
            The name of the node.
        """
        self._remove_node(self.nodes[name], notify_manager=False)

    @running
    def _remove_node(self, item: int, notify_manager: bool = True) -> None:
        """
        Remove a node from the GUI.

        ### Parameters
        `item` : int
            The node item.
        `notify_manager` : bool
            Whether to notify the manager to remove the node.
        """
        if item in dpg.get_selected_nodes(self.node_editor):
            # deselect node if it is selected
            self._select_node(None)

        # determine the node name
        name = dpg.get_item_label(item)

        # remove links associated with node
        for link in list(self.links.keys()):
            if link[0] == name or link[1] == name:
                self._remove_link(self.links[link], notify_manager=False)

        # unregister node by deleting it from the node list
        del self.nodes[name]

        # delete node from gui
        dpg.delete_item(item)

        if notify_manager:
            # remove node from manager
            self.manager.remove_node(name, notify_gui=False)

    @running
    def add_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str) -> None:
        """
        Outward facing method to add a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        """
        # determine slot IDs
        slot1 = self.nodes[node_out].output_slots[slot_out]
        slot2 = self.nodes[node_in].input_slots[slot_in]
        # add link internally
        self._add_link(self.node_editor, (slot1, slot2), notify_manager=False)

    @running
    def _add_link(self, sender: int, items: Tuple[int, int], notify_manager: bool = True) -> None:
        """
        Add a link between two nodes to the GUI.

        ### Parameters
        `sender` : int
            The sender item.
        `items` : Tuple[int, int]
            The two items to link.
        `notify_manager` : bool
            Whether to notify the manager to add the link.
        """
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
            self.manager.add_link(node1, node2, slot1, slot2, notify_gui=False)

    def remove_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str) -> None:
        """
        Outward facing method to remove a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        """
        self._remove_link(self.links[(node_out, node_in, slot_out, slot_in)], notify_manager=False)

    @running
    def _remove_link(self, item: int, notify_manager: bool = True) -> None:
        """
        Remove a link between two nodes from the GUI.

        ### Parameters
        `item` : int
            The link item.
        `notify_manager` : bool
            Whether to notify the manager to remove the link.
        """
        conf = dpg.get_item_configuration(item)
        # get slot names
        slot1, slot2 = dpg.get_item_label(conf["attr_1"]), dpg.get_item_label(conf["attr_2"])
        # get node names
        node1 = dpg.get_item_label(dpg.get_item_parent(conf["attr_1"]))
        node2 = dpg.get_item_label(dpg.get_item_parent(conf["attr_2"]))

        if notify_manager:
            # remove link from manager
            self.manager.remove_link(node1, node2, slot1, slot2, notify_gui=False)

        # unregister link
        del self.links[(node1, node2, slot1, slot2)]
        # delete link from gui
        dpg.delete_item(item)

    def _select_node(self, item: Optional[int]) -> None:
        """
        Register or unregister a node as selected. If a new node is selected, the parameters window is
        updated to show the parameters of that node.

        ### Parameters
        `item` : Optional[int]
            The node item. If None, deselects the currently selected node.
        """
        if self.selected_node == item:
            # do nothing if the node is already selected
            return

        # update window layout
        self.selected_node = item
        self.resize_callback(None, [dpg.get_viewport_width(), dpg.get_viewport_height()])
        # clear parameters window
        dpg.delete_item(self.parameters, children_only=True)

        if item is None:
            # node deselected, hide parameters window
            dpg.configure_item(self.parameters, show=False)
            return

        # get node reference
        node = self.nodes[dpg.get_item_label(item)].node_ref
        with dpg.tab_bar(parent=self.parameters):
            for group in node.params:
                with dpg.tab(label=format_name(group)) as tab:
                    for name, param in node.params[group].items():
                        add_param(tab, group, name, param, node)

        # show parameters window
        dpg.configure_item(self.parameters, show=True)

    def link_callback(self, sender: int, items: Tuple[int, int]) -> None:
        """Callback from DearPyGui that two nodes were connected."""
        self._add_link(sender, items)

    def delink_callback(self, _, item: int) -> None:
        """Callback from DearPyGui that a link was removed."""
        self._remove_link(item)

    def resize_callback(self, _, size: Tuple[int, int]) -> None:
        """Callback from DearPyGui that the viewport was resized."""
        # resize window to fill viewport
        dpg.configure_item(self.window, width=size[0], height=size[1])

        if self.selected_node is None:
            # no node selected, resize node editor to fill viewport
            dpg.configure_item(self.node_editor, width=size[0])
        else:
            # node selected, resize node editor to fill viewport minus parameters window
            dpg.configure_item(self.node_editor, width=size[0] - PARAM_WINDOW_WIDTH)

    def exit_callback(self, value):
        """Callback from DearPyGui that the close button was pressed."""
        # TODO: open a popup to confirm exit, if state was modified after save
        self.terminate()

    def _initialize(self, manager, width=1280, height=720):
        """Initialize the window and launch the event loop (blocking)."""
        dpg.create_context()
        self.manager = manager

        # initialize dicts to map names to dpg items
        self.nodes = {}
        self.links = {}
        self.selected_node = None
        self.create_node_window = None
        self.last_create_node_tab=0

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
        with dpg.group(horizontal=True, parent=self.window):
            # create node editor
            self.node_editor = dpg.add_node_editor(callback=self.link_callback, delink_callback=self.delink_callback)
            # add parameters window
            self.parameters = dpg.add_child_window(label="Parameters", autosize_x=True)

        # register key-press handler
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=events.key_press_callback, user_data=self)
            dpg.add_mouse_click_handler(callback=events.click_callback, user_data=self)
            dpg.add_mouse_double_click_handler(callback=events.double_click_callback, user_data=self)

        # register viewport resize handler
        dpg.set_viewport_resize_callback(self.resize_callback)

        # register exit handler
        dpg.set_exit_callback(self.exit_callback)

        # set up window
        self.resize_callback(None, [width, height])

        # start DearPyGui
        dpg.create_viewport(title="goofi-pipe", width=width, height=height, disable_close=True)
        # TODO: set the goofi-pipe icon as the window icon
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

        # TODO: add proper logging
        print("Shutting down graphical user interface.")

        # terminate manager
        manager.terminate(notify_gui=False)

    def terminate(self):
        """Stop DearPyGui and close the window."""
        dpg.stop_dearpygui()
