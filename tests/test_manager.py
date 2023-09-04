import pytest

from goofi.main import Manager
from goofi.nodes.add import Add
from goofi.nodes.constant import Constant


@pytest.mark.parametrize("node_class", [Add, Constant])  # TODO: automatically discover all nodes
def test_create_node(node_class):
    manager = Manager()
    manager.create_node(node_class)
