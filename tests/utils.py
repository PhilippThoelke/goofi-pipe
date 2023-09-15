import importlib
import inspect
import pkgutil
from typing import Callable, Dict, List, Type

import goofi.nodes
from goofi import params
from goofi.connection import Connection
from goofi.data import DataType
from goofi.node import Node
from goofi.params import Param


def list_param_types() -> List[Type[Param]]:
    """List all available parameter types."""
    return [cls for _, cls in inspect.getmembers(params, inspect.isclass) if issubclass(cls, Param) and cls is not Param]


def list_data_types() -> List[DataType]:
    """List all available data types."""
    return list(DataType.__members__.values())


def list_nodes(nodes=None, parent_module=goofi.nodes) -> List[Type[Node]]:
    """Returns a list of all nodes in goofi.nodes that are subclasses of Node."""
    if nodes is None:
        nodes = []

    for info in pkgutil.walk_packages(parent_module.__path__):
        module = importlib.import_module(f"{parent_module.__name__}.{info.name}")

        if info.ispkg:
            list_nodes(nodes, module)
            continue

        members = inspect.getmembers(module, inspect.isclass)
        nodes.extend([cls for _, cls in members if issubclass(cls, Node) and cls is not Node])
    return nodes


class DummyNode(Node):
    def process(self):
        return None, {}


class FullDummyNode(Node):
    def config_input_slots():
        res = {}
        for dtype in list_data_types():
            res["in_" + dtype.name.lower()] = dtype
        return res

    def config_output_slots():
        res = {}
        for dtype in list_data_types():
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


def make_custom_node(
    input_slots: Dict[str, DataType] = None,
    output_slots: Dict[str, DataType] = None,
    params: Dict[str, Dict[str, Param]] = None,
    process_callback: Callable = None,
) -> Type[Node]:
    class CustomNode(Node):
        @staticmethod
        def config_input_slots():
            return input_slots or {}

        @staticmethod
        def config_output_slots():
            return output_slots or {}

        @staticmethod
        def config_params():
            return params or {}

        def process(self, **kwargs):
            if process_callback:
                return process_callback(**kwargs)

            res = {}
            for name, slot in self.output_slots.items():
                res[name] = slot.dtype.empty(), {}
            return res

    return CustomNode
