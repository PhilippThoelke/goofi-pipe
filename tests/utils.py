import inspect
from typing import List, Type

from goofi import params
from goofi.connection import Connection
from goofi.data import DataType
from goofi.node import Node
from goofi.params import Param


def list_param_types() -> List[Type[Param]]:
    """
    List all available parameter types.
    """
    return [cls for _, cls in inspect.getmembers(params, inspect.isclass) if issubclass(cls, Param) and cls is not Param]


class DummyNode(Node):
    def process(self):
        return None, {}


class FullDummyNode(Node):
    def config_input_slots():
        res = {}
        for dtype in DataType.__members__.values():
            res["in_" + dtype.name.lower()] = dtype
        return res

    def config_output_slots():
        res = {}
        for dtype in DataType.__members__.values():
            res["out_" + dtype.name.lower()] = dtype
        return res

    def config_params():
        res = {}
        for param_type in list_param_types():
            res["param_" + param_type.__name__.lower()] = param_type()
        return {"test": res}

    def process(self, **kwargs):
        return None, {}


class BrokenProcessingNode(Node):
    """
    A node that raises an exception in its process method.

    ### Parameters
    `connection` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    `n_fails` : int
        The number of times the node should fail before succeeding.
    """

    def __init__(self, connection: Connection, *args, n_fails: int = 1, **kwargs) -> None:
        super().__init__(connection, *args, **kwargs)
        self.n_fails = n_fails

    def process(self):
        if self.n_fails > 0:
            self.n_fails -= 1
            raise Exception("BrokenProcessingNode")
        return {}
