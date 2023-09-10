from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Dict, Tuple

from goofi.data import DataType
from goofi.manager import NodeRef
from goofi.node import Node


class DummyNode(Node):
    """
    A dummy node that does nothing.

    ### Parameters
    `input_pipe` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    `call_super` : bool
        Whether or not to call the super class constructor (for testing only).
    `input_slots` : Dict[str, DataType]
        The input slots of the node.
    `output_slots` : Dict[str, DataType]
        The output slots of the node.
    """

    def __init__(
        self,
        input_pipe: Connection,
        call_super: bool = True,
        input_slots: Dict[str, DataType] = dict(),
        output_slots: Dict[str, DataType] = dict(),
    ) -> None:
        if call_super:
            super().__init__(input_pipe)

            for name, dtype in input_slots.items():
                self.register_input(name, dtype)
            for name, dtype in output_slots.items():
                self.register_output(name, dtype)

    def process(self):
        return {}


class BrokenProcessingNode(Node):
    """
    A node that raises an exception in its process method.

    ### Parameters
    `connection` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    `n_fails` : int
        The number of times the node should fail before succeeding.
    """

    def __init__(self, connection: Connection, n_fails: int = 1) -> None:
        super().__init__(connection)
        self.n_fails = n_fails

    def process(self):
        if self.n_fails > 0:
            self.n_fails -= 1
            raise Exception("BrokenProcessingNode")
        return {}


def create_dummy_node(*args, **kwargs) -> Tuple[NodeRef, DummyNode]:
    manager_pipe, node_pipe = Pipe()
    return NodeRef(manager_pipe), DummyNode(node_pipe, *args, **kwargs)
