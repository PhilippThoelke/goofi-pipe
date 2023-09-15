import pytest

from .utils import list_nodes


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
