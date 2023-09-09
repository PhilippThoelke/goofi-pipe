from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Tuple

from goofi.node import Node


class DummyNode(Node):
    def __init__(self, input_pipe: Connection, call_super: bool = True) -> None:
        if call_super:
            super().__init__(input_pipe)

    def process(self):
        return


def create_dummy_node(call_super: bool = True) -> Tuple[Connection, DummyNode]:
    manager_pipe, node_pipe = Pipe()
    return manager_pipe, DummyNode(node_pipe, call_super)
