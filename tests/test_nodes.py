import importlib
import inspect
import pkgutil

import pytest

import goofi.nodes
from goofi.node import Node


def list_nodes(nodes=[], parent_module=goofi.nodes):
    """Returns a list of all nodes in goofi.nodes that are subclasses of Node."""
    for info in pkgutil.walk_packages(parent_module.__path__):
        module = importlib.import_module(f"{parent_module.__name__}.{info.name}")

        if info.ispkg:
            list_nodes(nodes, module)
            continue

        members = inspect.getmembers(module, inspect.isclass)
        nodes.extend([cls for _, cls in members if issubclass(cls, Node) and cls is not Node])
    return nodes


@pytest.mark.parametrize("node", list_nodes())
def test_implement_init(node):
    # make sure the node uses the base class' __init__ and does not override it
    assert (
        node.__init__.__qualname__ == "Node.__init__"
    ), f"{node.__name__} should not override __init__. Use the setup() function to initialize the node instead."


@pytest.mark.parametrize("node", list_nodes())
def test_create_local(node):
    node.create_local()[0].terminate()


@pytest.mark.parametrize("node", list_nodes())
def test_create(node):
    node.create().terminate()
