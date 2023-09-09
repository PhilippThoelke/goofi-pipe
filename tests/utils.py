from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Tuple

from goofi.node import Node


class DummyNode(Node):
    """
    A dummy node that does nothing.

    ### Parameters
    `input_pipe` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    `call_super` : bool
        Whether or not to call the super class constructor (for testing only).
    """

    def __init__(self, input_pipe: Connection, call_super: bool = True) -> None:
        if call_super:
            super().__init__(input_pipe)

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


def create_dummy_node(call_super: bool = True) -> Tuple[Connection, DummyNode]:
    manager_pipe, node_pipe = Pipe()
    return manager_pipe, DummyNode(node_pipe, call_super)
