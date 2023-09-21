from typing import Type

import pytest

from goofi.node_helpers import list_nodes
from goofi.node import Node


@pytest.mark.parametrize("node", list_nodes())
def test_implement_init(node: Type[Node]):
    # make sure the node uses the base class' __init__ and does not override it
    assert (
        node.__init__.__qualname__ == "Node.__init__"
    ), f"{node.__name__} should not override __init__. Use the setup() function to initialize the node instead."


@pytest.mark.parametrize("node", list_nodes())
def test_create_local(node: Type[Node]):
    if node.__name__ in ["LSLClient", "EEGRecording"]:
        # TODO: try to find a way to clean up these nodes properly, without leaving threads running
        pytest.skip(f"Skipping {node.__name__} because it requires extra time to clean up threads.")

    node.create_local()[0].terminate()


@pytest.mark.parametrize("node", list_nodes())
def test_create(node: Type[Node]):
    node.create().terminate()


@pytest.mark.parametrize("node", list_nodes())
def test_category(node: Type[Node]):
    cat = node.category()
    assert isinstance(cat, str), f"Category of {node.__name__} should be a string."
    assert cat != "nodes", (
        f"Category of {node.__name__} should not be 'nodes'. The node should be placed in "
        f"'goofi/nodes/<category>/{node.__name__.lower()}.py'."
    )
    assert cat != "goofi", (
        f"Category of {node.__name__} should not be 'goofi'. The node should be placed in "
        f"'goofi/nodes/<category>/{node.__name__.lower()}.py'."
    )


@pytest.mark.parametrize("node", list_nodes())
def test_node_filename(node: Type[Node]):
    fname = node.__module__.split(".")[-1]
    # TODO: we might want to loosen this restriction in the future
    assert (
        fname == node.__name__.lower()
    ), f"Filename of node '{node.__name__}' should be '{node.__name__.lower()}.py' (TODO: maybe loosen this restriction)."
