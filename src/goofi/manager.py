import importlib
import logging
from typing import Dict

from goofi.connection import MultiprocessingConnection
from goofi.gui.window import Window
from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef

logging.basicConfig(
    level=logging.INFO,
    style="{",
    format="{asctime} | {levelname:^8} | {name:^20} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class NodeContainer:
    def __init__(self) -> None:
        self._nodes: Dict[str, NodeRef] = {}

    def add_node(self, name: str, node: NodeRef) -> str:
        """Adds a node to the container with a unique name."""
        if not isinstance(name, str):
            raise ValueError(f"Expected string, got {type(name)}")
        if not isinstance(node, NodeRef):
            raise ValueError(f"Expected NodeRef, got {type(node)}")

        idx = 0
        while f"{name}{idx}" in self._nodes:
            idx += 1
        self._nodes[f"{name}{idx}"] = node
        return f"{name}{idx}"

    def remove_node(self, name: str) -> None:
        """Terminates the node and removes it from the container."""
        if name in self._nodes:
            self._nodes[name].terminate()
            del self._nodes[name]
            return True
        raise KeyError(f"Node {name} not in container")

    def __getitem__(self, name: str) -> NodeRef:
        return self._nodes[name]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._nodes


class Manager:
    """
    The manager keeps track of all nodes, and provides methods to add and remove nodes, and links between them.
    It also interfaces with the GUI to display the nodes and links, and to handle user interaction.

    ### Parameters
    `headless` : bool
        Whether to run in headless mode. If `True`, the GUI will not be started.
    """

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._running = True
        self.nodes = NodeContainer()

        logger.info("Manager initialized.")

        if not self.headless:
            Window(self)

    def add_node(self, name: str, category: str, notify_gui: bool = True) -> None:
        """
        Adds a node to the container.

        ### Parameters
        `name` : str
            The name of the node.
        `category` : str
            The category of the node.
        `notify_gui` : bool
            Whether to notify the gui to add the node.
        """
        mod = importlib.import_module(f"goofi.nodes.{category}.{name.lower()}")
        node = getattr(mod, name)

        # instantiate the node and add it to the container
        ref = node.create_local()[0]
        name = self.nodes.add_node(name.lower(), ref)

        # add the node to the gui
        if not self.headless and notify_gui:
            Window().add_node(name, ref)
        return name

    def remove_node(self, name: str, notify_gui: bool = True) -> None:
        """
        Removes a node from the container.

        ### Parameters
        `name` : str
            The name of the node.
        `notify_gui` : bool
            Whether to notify the gui to remove the node.
        """
        self.nodes.remove_node(name)

        if not self.headless and notify_gui:
            Window().remove_node(name)

    def add_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str, notify_gui: bool = True) -> None:
        """
        Adds a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        `notify_gui` : bool
            Whether to notify the gui to add the link.
        """
        # TODO: prevent multiple links to the same input slot
        self.nodes[node_out].connection.send(
            Message(
                MessageType.ADD_OUTPUT_PIPE,
                {"slot_name_out": slot_out, "slot_name_in": slot_in, "node_connection": self.nodes[node_in].connection},
            )
        )

        if not self.headless and notify_gui:
            Window().add_link(node_out, node_in, slot_out, slot_in)

    def remove_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str, notify_gui: bool = True) -> None:
        """
        Removes a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        `notify_gui` : bool
            Whether to notify the gui to remove the link.
        """
        self.nodes[node_out].connection.send(
            Message(
                MessageType.REMOVE_OUTPUT_PIPE,
                {"slot_name_out": slot_out, "slot_name_in": slot_in, "node_connection": self.nodes[node_in].connection},
            )
        )

        if not self.headless and notify_gui:
            Window().remove_link(node_out, node_in, slot_out, slot_in)

    def terminate(self, notify_gui: bool = True) -> None:
        """
        Terminates the manager and all nodes.

        ### Parameters
        `notify_gui` : bool
            Whether to notify the gui to terminate.
        """
        if not self.headless and notify_gui:
            # terminate the gui, which calls manager.terminate() with notify_gui=False once it is closed
            Window().terminate()
        else:
            # terminate the manager
            self._running = False
            for node in self.nodes:
                self.nodes[node].connection.send(Message(MessageType.TERMINATE, {}))
                self.nodes[node].connection.close()
            logger.info("Manager terminated.")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def headless(self) -> bool:
        return self._headless


def main(duration: float = 0, args=None):
    import argparse
    import time

    # parse arguments
    parser = argparse.ArgumentParser(description="goofi-pipe")
    parser.add_argument("--headless", action="store_true", help="run in headless mode")
    args = parser.parse_args(args)

    # create manager
    manager = Manager(headless=args.headless)

    # add some example nodes
    manager.add_node("Constant", "generators")
    manager.add_node("Sine", "generators")
    manager.add_node("Add", "array")
    manager.add_node("Buffer", "array")

    # connect example nodes
    manager.add_link("constant0", "add0", "out", "a")
    manager.add_link("sine0", "add0", "out", "b")

    # create a local pipe to receive data from the node
    my_conn, node_conn = MultiprocessingConnection.create()
    manager.nodes["add0"].connection.send(
        Message(MessageType.ADD_OUTPUT_PIPE, {"slot_name_out": "out", "slot_name_in": "in", "node_connection": my_conn})
    )

    # print data from the node until the manager terminates
    start = last_msg = time.time()
    while manager.running:
        try:
            if duration > 0 and time.time() - start > duration:
                # duration exceeded, terminate the manager
                manager.terminate()
                break

            # check if there is a message from the node
            if not node_conn.poll(0.01):
                continue

            # parse the message and print the data
            msg = node_conn.recv()
            # if msg.type == MessageType.DATA:
            #     print(f"{1 / (time.time() - last_msg):.2f} Hz: Output of 'add0' is {msg.content['data'].data[0]}")
            last_msg = time.time()
        except KeyboardInterrupt:
            manager.terminate()
            break


if __name__ == "__main__":
    main()
