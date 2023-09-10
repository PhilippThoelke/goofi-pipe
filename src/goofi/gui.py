import threading
import time
from typing import Callable

import dearpygui.dearpygui as dpg

from goofi.node import NodeRef


class Window:
    """
    The graphical user interface window, implemented as a threaded singleton. The window is created
    when Window() is first called. Subsequent calls to Window() will return the same instance. The
    window thread is a daemon thread, so it will be terminated when the main thread exits, or when
    Window().close() is called.

    ### Parameters
    `close_callback` : Callable
        A callback function that will be called when the window is closed. This function takes no
        arguments and returns nothing.
    """

    _instance = None

    def __new__(cls, close_callback: Callable = None):
        if cls._instance is None:
            # instantiate the window thread
            cls._instance = super(Window, cls).__new__(cls)
            threading.Thread(target=cls._instance._initialize, args=(close_callback,), daemon=True).start()
        return cls._instance

    def add_node(self, name: str, node: NodeRef) -> None:
        # wait for dearpygui to start
        tries = 5
        while not dpg.is_dearpygui_running() and tries > 0:
            time.sleep(0.05)
            tries -= 1

        if tries == 0:
            raise RuntimeError("DearPyGui is not running.")

        dpg.add_node(
            parent=self.node_editor,
            id=name,
            label=name,
            user_data=node,
        )

    def _initialize(self, close_callback: Callable):
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
        self.node_editor = dpg.add_node_editor(parent=self.window)

        dpg.set_viewport_resize_callback(self.resize_callback)

        dpg.create_viewport(title="goofi-pipe", width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

        close_callback()

    def close(self):
        dpg.stop_dearpygui()

    def resize_callback(self, sender, data):
        dpg.configure_item(self.window, width=data[0], height=data[1])


if __name__ == "__main__":
    window = Window()

    while True:
        pass
