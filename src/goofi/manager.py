import argparse
import importlib
from typing import Dict

from goofi.connection import MultiprocessingConnection
from goofi.gui.window import Window
from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef


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
    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._running = True
        self.nodes = NodeContainer()

        if not self.headless:
            Window(self)

    def add_node(self, node_path: str, call_gui: bool = True) -> None:
        """
        Adds a node to the container.

        ### Parameters
        `node_path` : str
            The path to the node class, e.g. `generators.Constant`.
        `call_gui` : bool
            Whether to call the gui to add the node.
        """

        node_cls = node_path.split(".")[-1]
        mod = importlib.import_module(f"goofi.nodes.{node_path.lower()}")
        node = getattr(mod, node_cls)

        # instantiate the node and add it to the container
        ref = node.create_local()[0]
        name = self.nodes.add_node(node_cls.lower(), ref)

        # add the node to the gui
        if not self.headless and call_gui:
            Window().add_node(name, ref)
        return name

    def remove_node(self, name: str, call_gui: bool = True) -> None:
        """
        Removes a node from the container.

        ### Parameters
        `name` : str
            The name of the node.
        `call_gui` : bool
            Whether to call the gui to remove the node.
        """
        if name in self.nodes:
            self.nodes.remove_node(name)

            if not self.headless and call_gui:
                Window().remove_node(name)

    def add_link(self, node1: str, node2: str, slot1: str, slot2: str) -> None:
        if node1 not in self.nodes:
            raise KeyError(f"Node {node1} not in container")
        if node2 not in self.nodes:
            raise KeyError(f"Node {node2} not in container")

        node1 = self.nodes[node1]
        node2 = self.nodes[node2]
        node1.connection.send(
            Message(
                MessageType.ADD_OUTPUT_PIPE,
                {"slot_name_out": slot1, "slot_name_in": slot2, "node_connection": node2.connection},
            )
        )

    def remove_link(self, node1: str, node2: str, slot1: str, slot2: str) -> None:
        if node1 not in self.nodes:
            raise KeyError(f"Node {node1} not in container")
        if node2 not in self.nodes:
            raise KeyError(f"Node {node2} not in container")

        n1 = self.nodes[node1]
        n2 = self.nodes[node2]
        n1.connection.send(
            Message(
                MessageType.REMOVE_OUTPUT_PIPE,
                {"slot_name_out": slot1, "slot_name_in": slot2, "node_connection": n2.connection},
            )
        )

    def terminate(self) -> None:
        self._running = False
        for node in self.nodes:
            self.nodes[node].connection.send(Message(MessageType.TERMINATE, {}))
            self.nodes[node].connection.close()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def headless(self) -> bool:
        return self._headless


def main(duration: float = 0, args=None):
    parser = argparse.ArgumentParser(description="goofi-pipe")
    parser.add_argument("--headless", action="store_true", help="run in headless mode")
    args = parser.parse_args(args)

    import time

    manager = Manager(headless=args.headless)
    manager.add_node("generators.Constant")
    manager.add_node("generators.Sine")
    manager.add_node("Add")

    # manager.connect("constant0", "add0", "out", "a")
    # manager.connect("sine0", "add0", "out", "b")

    my_conn, node_conn = MultiprocessingConnection.create()
    manager.nodes["add0"].connection.send(
        Message(MessageType.ADD_OUTPUT_PIPE, {"slot_name_out": "out", "slot_name_in": "in", "node_connection": my_conn})
    )

    start = last = time.time()
    while manager.running:
        if duration > 0 and time.time() - start > duration:
            manager.terminate()
            break

        if not node_conn.poll(0.05):
            continue
        msg = node_conn.recv()

        if msg.type == MessageType.DATA:
            print(f"{1/(time.time()-last):.2f}: {msg.content['data'].data[0]}")

        last = time.time()


if __name__ == "__main__":
    main()
