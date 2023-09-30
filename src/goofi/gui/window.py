import os
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg

from goofi.data import DataType
from goofi.gui import events
from goofi.gui.data_viewer import DTYPE_VIEWER_MAP
from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef
from goofi.params import BoolParam, FloatParam, IntParam, Param, StringParam

DTYPE_SHAPE_MAP = {
    DataType.ARRAY: dpg.mvNode_PinShape_CircleFilled,
    DataType.STRING: dpg.mvNode_PinShape_TriangleFilled,
    DataType.TABLE: dpg.mvNode_PinShape_QuadFilled,
}

PARAM_WINDOW_WIDTH = 400


def format_name(name: str) -> str:
    """Format a name to be used as a node label."""
    return name.replace("_", " ").title()


def running(func):
    """Decorator to make sure the window is running before calling a function."""

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


def update_minimap(func):
    """Decorator to update the minimap state after the node editor is updated."""

    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

        # update minimap state
        if len(args[0].nodes) > 0:
            dpg.configure_item(args[0].node_editor, minimap=True)
        else:
            dpg.configure_item(args[0].node_editor, minimap=False)

    return wrapper


@dataclass
class GUINode:
    item: int
    input_slots: Dict[str, int]
    output_slots: Dict[str, int]
    output_draw_handlers: Dict[str, int]
    node_ref: NodeRef


def handle_data(gui_node: GUINode, node: NodeRef, message: Message):
    """
    Handle a data message from a node. This function is registered as a message handler for the
    `MessageType.DATA` message type.

    ### Parameters
    `gui_node` : GUINode
        The GUI node instance.
    `node` : NodeRef
        The node reference.
    `message` : Message
        The data message.
    """
    try:
        gui_node.output_draw_handlers[message.content["slot_name"]](message)
    except ValueError as e:
        # TODO: add proper logging
        print(f"Output draw handler for slot {message.content['slot_name']} failed: {e}")


def param_updated(_, value, user_data):
    """A callback function for updating a parameter value."""
    group, name, node = user_data[:3]

    if len(user_data) == 5:
        # the parameter includes multiple input widgets, update all of them
        input_group, value_type = user_data[3:]
        for child in dpg.get_item_children(input_group)[1]:
            dpg.set_value(child, value_type(value))

        # make sure the value has the correct type
        value = value_type(value)

    # send the updated parameter to the node
    node.update_param(group, name, value)


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
    with dpg.table_row(parent=parent):
        with dpg.table_cell():
            dpg.add_text(format_name(name))

        with dpg.table_cell():
            if isinstance(param, BoolParam):
                # parameter is a bool
                if param.toggle:
                    dpg.add_checkbox(default_value=param.value, callback=param_updated, user_data=(group, name, node))
                else:
                    dpg.add_button(callback=param_updated, user_data=(group, name, node))
            elif isinstance(param, FloatParam):
                with dpg.group(horizontal=True) as input_group:
                    # parameter is a float
                    dpg.add_input_text(
                        width=50,
                        scientific=True,
                        default_value=str(param.value),
                        callback=param_updated,
                        user_data=(group, name, node, input_group, float),
                    )
                    dpg.add_slider_float(
                        default_value=param.value,
                        min_value=param.vmin,
                        max_value=param.vmax,
                        user_data=(group, name, node, input_group, float),
                        callback=param_updated,
                    )
            elif isinstance(param, IntParam):
                with dpg.group(horizontal=True) as input_group:
                    # parameter is an integer
                    dpg.add_input_text(
                        width=50,
                        default_value=str(param.value),
                        callback=param_updated,
                        user_data=(group, name, node, input_group, int),
                    )
                    dpg.add_slider_int(
                        default_value=param.value,
                        min_value=param.vmin,
                        max_value=param.vmax,
                        callback=param_updated,
                        user_data=(group, name, node, input_group, int),
                    )
            elif isinstance(param, StringParam):
                # parameter is a string
                if param.options is None:
                    # `options` is not set, use an unconstrained text input
                    dpg.add_input_text(
                        default_value=param.value,
                        callback=param_updated,
                        user_data=(group, name, node),
                    )
                else:
                    # `options` is set, use a dropdown menu
                    dpg.add_combo(
                        default_value=param.value,
                        items=param.options,
                        callback=param_updated,
                        user_data=(group, name, node),
                    )
            else:
                raise NotImplementedError(f"Parameter type {type(param)} not implemented.")


def add_output_slot(parent: int, name: str, closed: bool = False, size: Tuple[int, int] = (175, 100)) -> int:
    """
    Add an output slot to the GUI, which consists of a collapsable header and a content window for the data viewer.

    ### Parameters
    `parent` : int
        The parent item (the node attribute).
    `name` : str
        The name of the output slot.
    `closed` : bool
        Whether the headers should be closed initially.
    `size` : Tuple[int, int]
        The size of the content area.

    ### Returns
    `content` : int
        The content window item.
    """

    def header_callback(_1, _2, items: List[int]):
        """
        Event callback for the collapsable header, both for general clicks, and for toggle_open events.
        This is a workaround for the fact that toggle_close events don't exist. As toggle_open is always
        called after click, clicking on the header always closes it, so the toggle_open callback is used
        reopen it if the internal header state is open.

        ### Parameters
        `items` : List[int]
            A list of four items: a bool indicating to close the header, the header, the window, and the content window.
        """
        close, header, window, content = items
        header_height = dpg.get_item_state(header)["rect_size"][1]

        if header_height == 0:
            # this happens when the header is off screen, set to default height
            header_height = 15

        if close:
            # header was closed, shrink window to header size
            dpg.set_item_height(window, header_height)
        else:
            # header was opened, expand window to header size plus cotent size
            content_height = dpg.get_item_state(content)["rect_size"][1]
            if content_height == 0:
                # this happens when the content window is off screen, set to default height
                content_height = size[1]

            dpg.set_item_height(window, header_height + content_height + 4)  # magic + 4 to avoid scroll bar

    # NOTE: The user_data lists are used to pass data to the callbacks. The first element is a bool, which
    # is used to indicate whether the header was closed or opened. The remaining elements are the header,
    # window, and content window items. Due to the way DearPyGui registers callbacks we can not use partials here.
    user_data_open, user_data_close = [False], [True]
    # register handlers
    with dpg.item_handler_registry() as reg:
        dpg.add_item_clicked_handler(callback=header_callback, user_data=user_data_close)
        dpg.add_item_toggled_open_handler(callback=header_callback, user_data=user_data_open)

    # NOTE: we initially set the height to 1 to avoid the node reaching out of the window, which would break the
    # initial header_callback call as the header and content sizes according to DearPyGui would be 0.
    # The child window is needed, because otherwise the header will expand to fill the entire node editor window.
    with dpg.child_window(width=size[0], height=1, parent=parent) as window:
        # add collapsable content area for the data viewer
        header = dpg.add_collapsing_header(label=name, default_open=not closed, open_on_arrow=False, closable=False)
        content = dpg.add_child_window(width=size[0], height=size[1], user_data=size, parent=header)

    # store header, window and content window in user_data for the two callbacks
    user_data_open.extend([header, window, content])
    user_data_close.extend([header, window, content])
    # bind the two callbacks to the header
    dpg.bind_item_handler_registry(header, reg)

    # schedule the header callback to set up the window sizes
    initial_args = user_data_close if closed else user_data_open
    threading.Timer(0.1, header_callback, [None, None, initial_args]).start()
    return content


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
    @update_minimap
    def add_node(
        self,
        node_name: str,
        node: NodeRef,
        pos: Optional[Tuple[int, int]] = None,
        offset: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Outward facing method to add a node to the GUI.

        ### Parameters
        `node_name` : str
            The name of the node.
        `node` : NodeRef
            The node reference.
        `pos` : Optional[Tuple[int, int]]
            Position of the node. If None, uses the mouse position.
        `offset` : Optional[Tuple[int, int]]
            Offsets the node position by the given amount.
        """
        if pos is None:
            pos = self._node_editor_mouse_pos()
        if offset is not None:
            pos = (pos[0] + offset[0], pos[1] + offset[1])

        with dpg.node(parent=self.node_editor, label=node_name, pos=pos, user_data=node) as node_id:
            ############### input slots ###############
            in_slots = {}
            for name, dtype in node.input_slots.items():
                # create input slot
                in_slots[name] = dpg.add_node_attribute(
                    label=name, attribute_type=dpg.mvNode_Attr_Input, shape=DTYPE_SHAPE_MAP[dtype], user_data=dtype
                )
                # simply add a text label
                dpg.add_text(name, parent=in_slots[name])

            ############### output slots ###############
            out_slots = {}
            output_draw_handlers = {}
            for name, dtype in node.output_slots.items():
                # create output slot
                out_slots[name] = dpg.add_node_attribute(
                    label=name, attribute_type=dpg.mvNode_Attr_Output, shape=DTYPE_SHAPE_MAP[dtype], user_data=dtype
                )
                # create content window for data viewer (initialize closed if more than two output slots)
                content = add_output_slot(out_slots[name], name, closed=len(node.output_slots) > 2)
                # create data viewer
                output_draw_handlers[name] = DTYPE_VIEWER_MAP[dtype](content)

            # add node to node list
            self.nodes[node_name] = GUINode(node_id, in_slots, out_slots, output_draw_handlers, node)

            # TODO: register PROCESSING_ERROR message handler and display error messages
            node.set_message_handler(MessageType.PROCESSING_ERROR, lambda node, data: print(data.content["error"]))

            # register data message handler to update the data viewers
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
    @update_minimap
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

        # make sure the data types match
        dtype1 = dpg.get_item_user_data(items[0])
        dtype2 = dpg.get_item_user_data(items[1])
        if dtype1 != dtype2:
            # TODO: add proper logging
            print(f"Data types of slots {slot1} ({dtype1.name}) and {slot2} ({dtype2.name}) do not match.")
            return

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

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a goofi-pipe from a file.

        ### Parameters
        `path` : Optional[str]
            Optional path to load the GUI state from. If None, opens a file dialog.
        """

        def file_callback(_, info):
            self.load(path=info["file_path_name"])

        if path is None:
            # no save path set, open save dialog
            self._get_file(file_callback, message="Select a file to load from.")
            return

        try:
            # try loading the manager
            self.manager.load(filepath=path)
        except FileNotFoundError:
            # file does not exist, open error dialog
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Error", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("The file does not exist.")

                dpg.add_separator()
                dpg.add_text(path)
                dpg.add_separator()

                # add button
                dpg.add_button(label="Ok", callback=lambda _, __: dpg.delete_item(win))
        except RuntimeError:
            # file does not exist, open error dialog
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Error", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("The current pipeline is not empty.")

                # add button
                dpg.add_button(label="Ok", callback=lambda _, __: dpg.delete_item(win))

    def save(
        self,
        path: Optional[str] = None,
        overwrite: bool = True,
        save_as: bool = False,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Save the current state of the GUI.

        ### Parameters
        `path` : Optional[str]
            Optional path to save the GUI state to. If None, uses the current save path.
        `overwrite` : bool
            Whether to overwrite the current save path.
        `save_as` : bool
            Whether to open a file dialog to select a save path.
        `callback` : Optional[Callable]
            Optional callback function to call when the save is complete.
        """

        def file_callback(_, info):
            # save the manager to the selected path
            self.save(path=info["file_path_name"], overwrite=False, callback=callback)

        if save_as or (self.manager.save_path is None and path is None):
            # no save path set, open save dialog
            self._get_file(file_callback, message="Select a file to save to.")
            return

        try:
            # try saving the manager
            self.manager.save(filepath=path, overwrite=overwrite)

            if callback is not None:
                callback()
        except FileExistsError:

            def confirm_callback(_1, _2, data):
                """Callback for the overwrite confirmation dialog."""
                win, confirm = data
                if confirm:
                    self.save(path=path, overwrite=True, callback=callback)
                # close the confirmation dialog
                dpg.delete_item(win)

            # file already exists, open overwrite dialog
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Overwrite?", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("The file already exists. Overwrite?")

                if path is not None:
                    dpg.add_separator()
                    dpg.add_text(path)
                    dpg.add_separator()

                # add buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Yes", callback=confirm_callback, user_data=(win, True))
                    dpg.add_button(label="Cancel", callback=confirm_callback, user_data=(win, False))

    def _get_file(self, callback: Callable, message: Optional[str] = None) -> None:
        """
        Open a file dialog (e.g. for loading or saving a file).

        ### Parameters
        `callback` : Callable
            The callback function to call when the file is selected.
        `message` : Optional[str]
            Optional message to display in the save dialog.
        """

        # open file selection dialog
        width, height = dpg.get_viewport_width() / 2, dpg.get_viewport_height() / 2
        with dpg.file_dialog(
            label=message,
            callback=callback,
            default_path=".",
            default_filename="",
            modal=True,
            width=width,
            height=height,
        ) as self.file_selection_window:
            dpg.add_file_extension(".gfi", label="goofi-pipes")
            dpg.add_file_extension(".*", label="All Files")

    def get_node_state(self, name: str) -> Dict[str, Any]:
        """
        Get the state of a node.

        ### Parameters
        `name` : str
            The name of the node.

        ### Returns
        `state` : Dict[str, Any]
            The state of the node.
        """
        if name not in self.nodes:
            # TODO: add proper logging
            print(f"Can't get state of node {name}. The node does not exist.")
            return None

        node = self.nodes[name]
        return {
            "pos": dpg.get_item_pos(node.item),
        }

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

        if item is None or dpg.get_item_label(item) not in self.nodes:
            self.selected_node = None
            # node deselected, hide parameters window and resize node editor
            dpg.configure_item(self.parameters, show=False)
            self.resize_callback(None, [dpg.get_viewport_width(), dpg.get_viewport_height()])
            return

        # get node reference
        node = self.nodes[dpg.get_item_label(item)].node_ref
        with dpg.tab_bar(parent=self.parameters):
            for group in node.params:
                with dpg.tab(label=format_name(group)) as tab:
                    with dpg.table(header_row=False, parent=tab, policy=dpg.mvTable_SizingStretchProp) as table:
                        dpg.add_table_column()
                        dpg.add_table_column()

                        for name, param in node.params[group].items():
                            add_param(table, group, name, param, node)

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
            dpg.configure_item(self.node_editor, width=0)
        else:
            # node selected, resize node editor to fill viewport minus parameters window
            dpg.configure_item(self.node_editor, width=size[0] - PARAM_WINDOW_WIDTH)

    def update_title(self) -> None:
        """Update the window title."""
        start = time.time()
        while not hasattr(self, "manager"):
            # wait until the manager is initialized
            if time.time() - start > 1:
                raise RuntimeError("Timeout while waiting for manager to initialize.")
            time.sleep(0.01)

        # update window title
        unsaved_changes = "*" if self.manager.unsaved_changes else ""
        if self.manager.save_path is None:
            dpg.set_viewport_title(f"{unsaved_changes}goofi-pipe")
        else:
            dpg.set_viewport_title(f"goofi-pipe | {unsaved_changes}{self.manager.save_path.split(os.sep)[-1]}")

    def exit_callback(self, value):
        """Callback from DearPyGui that the close button was pressed."""
        if self.manager.unsaved_changes and not self.unsaved_changes_dialog_open:

            def callback(_1, _2, data):
                """Callback for the unsaved changes dialog."""
                win, save = data
                # close the confirmation dialog
                dpg.delete_item(win)
                self.unsaved_changes_dialog_open = False

                if save:
                    # save changes first
                    self.save(callback=self.terminate)
                else:
                    # close without saving
                    self.terminate()

            # unsaved changes, open confirm dialog
            self.unsaved_changes_dialog_open = True
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            pos = (w / 2 - w / 8, h / 2 - h / 12)
            with dpg.window(label="Save before closing?", width=w / 4, height=h / 6, pos=pos) as win:
                dpg.add_text("There are unsaved changes.\nDo you want to save before closing?")

                # add buttons
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save", callback=callback, user_data=(win, True))
                    dpg.add_button(label="Close without saving", callback=callback, user_data=(win, False))
            return

        # no unsaved changes, close window
        self.terminate()

    def _to_node_editor_coords(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert a position from window space to node editor space."""
        # get reference node position
        orig_ref_pos = dpg.get_item_user_data("_ref")
        curr_ref_pos = dpg.get_item_state("_ref")["rect_min"]
        # calculate mouse position within node editor
        return pos[0] - curr_ref_pos[0] + orig_ref_pos[0], pos[1] - curr_ref_pos[1] + orig_ref_pos[1]

    def _to_window_coords(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert a position from node editor space to window space."""
        # get reference node position
        orig_ref_pos = dpg.get_item_user_data("_ref")
        curr_ref_pos = dpg.get_item_state("_ref")["rect_min"]
        # calculate mouse position within node editor
        return pos[0] - curr_ref_pos[0] + orig_ref_pos[0], pos[1] - curr_ref_pos[1] + orig_ref_pos[1]

    def _node_editor_mouse_pos(self) -> Tuple[int, int]:
        """Get the mouse position within the node editor."""
        return self._to_node_editor_coords(dpg.get_mouse_pos(local=False))

    def _initialize(self, manager, width=1280, height=720):
        """Initialize the window and launch the event loop (blocking)."""
        dpg.create_context()
        self.manager = manager

        # initialize dicts to map names to dpg items
        self.nodes = {}
        self.links = {}
        self.selected_node = None
        self.create_node_window = None
        self.last_create_node_tab = 0
        self.file_selection_window = None
        self.unsaved_changes_dialog_open = False
        self.node_clipboard = None

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

        # create menu bar
        with dpg.menu_bar(parent=self.window):
            dpg.add_menu_item(label="Load", callback=lambda: self.load())
            dpg.add_menu_item(label="Save As", callback=lambda: self.save(save_as=True))
            dpg.add_menu_item(label="Save", callback=lambda: self.save())

        with dpg.group(horizontal=True, parent=self.window):
            # create node editor
            self.node_editor = dpg.add_node_editor(callback=self.link_callback, delink_callback=self.delink_callback)
            # add parameters window
            self.parameters = dpg.add_child_window(label="Parameters", autosize_x=True)

        # hide the reference node using themes
        with dpg.theme() as ref_theme:
            with dpg.theme_component():
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, [0, 0, 0, 0], category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 0, 0, category=dpg.mvThemeCat_Nodes)

        # create a reference node to calculate the mouse position within the node editor
        # NOTE: this is a workaround as DearPyGui doesn't provide access to the node editor coordinates
        # NOTE: ideally we would set show=False in the ref node, but this causes a Segmentation Fault
        pos = [0, 0]
        dpg.add_node(label=" ", tag="_ref", parent=self.node_editor, pos=pos, user_data=pos, draggable=False)
        dpg.bind_item_theme("_ref", ref_theme)

        # register user interaction handlers
        with dpg.handler_registry():
            dpg.add_key_release_handler(callback=events.key_release_callback, user_data=self)
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
