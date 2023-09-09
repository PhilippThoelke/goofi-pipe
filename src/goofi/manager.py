from dataclasses import dataclass
from multiprocessing.connection import Connection


@dataclass
class NodeRef:
    connection: Connection


class NodeContainer:
    def __init__(self) -> None:
        self._nodes = {}

    def add_node(self, name: str, node: NodeRef) -> str:
        """Adds a node to the container with a unique name."""
        idx = 0
        while f"{name}{idx}" in self._nodes:
            idx += 1
        self._nodes[f"{name}{idx}"] = node
        return f"{name}{idx}"

    def remove_node(self, name: str) -> bool:
        """Terminates the node and removes it from the container."""
        if name in self._nodes:
            self._nodes[name].connection.close()
            del self._nodes[name]
            return True
        return False

    def __getitem__(self, name: str) -> NodeRef:
        return self._nodes[name]

    def __len__(self) -> int:
        return len(self._nodes)


class Manager:
    def __init__(self) -> None:
        self.nodes = []
