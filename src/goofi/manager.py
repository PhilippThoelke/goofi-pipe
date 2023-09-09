import importlib
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from goofi.message import Message, MessageType


@dataclass
class NodeRef:
    connection: Connection


class NodeContainer:
    def __init__(self) -> None:
        self._nodes = {}

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
            self._nodes[name].connection.send(Message(MessageType.TERMINATE, {}))
            self._nodes[name].connection.close()
            del self._nodes[name]
            return True
        raise KeyError(f"Node {name} not in container")

    def __getitem__(self, name: str) -> NodeRef:
        return self._nodes[name]

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, name: str) -> bool:
        return name in self._nodes


class Manager:
    def __init__(self) -> None:
        self.nodes = NodeContainer()

    def add_node(self, node_path: str) -> None:
        node_cls = node_path.split(".")[-1]
        mod = importlib.import_module(f"goofi.nodes.{node_path.lower()}")
        node = getattr(mod, node_cls)
        conn1, conn2 = Pipe()
        ref = NodeRef(conn1)
        Process(target=node, args=(conn2,), daemon=True).start()
        return self.nodes.add_node(node_cls.lower(), ref)

    def connect(self, node1: str, node2: str, slot1: str, slot2: str) -> None:
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
