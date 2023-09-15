import threading
import time
from typing import Tuple

import dearpygui.dearpygui as dpg

from goofi.gui.events import KEY_MAP
from goofi.node_helpers import NodeRef


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
            cls._instance = super(Window, cls).__new__(cls)
            threading.Thread(target=cls._instance._initialize, args=(manager,), daemon=True).start()
        return cls._instance

    @running
    def add_node(self, name: str, node: NodeRef) -> None:
        # add node
        dpg.add_node(
            parent=self.node_editor,
            id=name,
            label=name,
            user_data=node,
        )

        # add input slots
        for slot_name, slot in node.input_slots.items():
            dpg.add_node_attribute(
                parent=name,
                id=f"{name}_{slot_name}_in",
                label=slot_name,
                attribute_type=dpg.mvNode_Attr_Input,
            )
        # add output slots
        for slot_name, slot in node.output_slots.items():
            dpg.add_node_attribute(
                parent=name,
                id=f"{name}_{slot_name}_out",
                label=slot_name,
                attribute_type=dpg.mvNode_Attr_Output,
            )

    @running
    def remove_node(self, item: int) -> None:
        manager = dpg.get_item_user_data(self.window)
        manager.remove_node(dpg.get_item_label(item), call_gui=False)
        dpg.delete_item(item)

    @running
    def add_link(self, sender: int, items: Tuple[int, int]) -> None:
        dpg.add_node_link(items[0], items[1], parent=sender)
        # get slot names
        slot1 = dpg.get_item_label(items[0])
        slot2 = dpg.get_item_label(items[1])
        # get node names
        node1 = dpg.get_item_label(dpg.get_item_parent(items[0]))
        node2 = dpg.get_item_label(dpg.get_item_parent(items[1]))

        # connect nodes
        manager = dpg.get_item_user_data(self.window)
        manager.add_link(node1, node2, slot1, slot2)

    @running
    def remove_link(self, item: int) -> None:
        conf = dpg.get_item_configuration(item)
        # get slot names
        slot1, slot2 = dpg.get_item_label(conf["attr_1"]), dpg.get_item_label(conf["attr_2"])
        # get node names
        node1 = dpg.get_item_label(dpg.get_item_parent(conf["attr_1"]))
        node2 = dpg.get_item_label(dpg.get_item_parent(conf["attr_2"]))

        manager = dpg.get_item_user_data(self.window)
        manager.remove_link(node1, node2, slot1, slot2)
        dpg.delete_item(item)

    def link_callback(self, sender, data):
        self.add_link(sender, data)

    def delink_callback(self, sender, data):
        self.remove_link(sender, data)

    def key_press_callback(self, sender, data):
        if data in KEY_MAP:
            KEY_MAP[data](sender, self)

    def _initialize(self, manager):
        dpg.create_context()

        self.window = dpg.add_window(
            label="goofi-pipe",
            width=800,
            height=600,
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            no_close=True,
            no_collapse=True,
        )
        self.node_editor = dpg.add_node_editor(
            parent=self.window, callback=self.link_callback, delink_callback=self.delink_callback
        )

        dpg.set_item_user_data(self.window, manager)

        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self.key_press_callback)

        dpg.set_viewport_resize_callback(self.resize_callback)

        dpg.create_viewport(title="goofi-pipe", width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

        manager.terminate()

    def close(self):
        dpg.stop_dearpygui()

    def resize_callback(self, sender, data):
        dpg.configure_item(self.window, width=data[0], height=data[1])
