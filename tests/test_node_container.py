import pytest

from goofi.manager import NodeContainer, NodeRef

from .utils import create_dummy_node


def test_creation():
    cont = NodeContainer()
    assert len(cont) == 0


def test_assignment():
    cont = NodeContainer()
    # the container shouldn't allow item assignment
    with pytest.raises(TypeError):
        cont["test"] = 1


def test_contains():
    cont = NodeContainer()
    assert "test" not in cont, "Empty container shouldn't contain anything"

    cont.add_node("test", NodeRef(None))
    assert "test0" in cont, "Added node but container doesn't contain it"


def test_add_node():
    cont = NodeContainer()

    # adding a node should increase the length of the container
    cont.add_node("test", NodeRef(None))
    assert len(cont) == 1, "Added node but length didn't increase"
    assert "test0" in cont, "Wrong name when adding node"

    # when adding a node with the same name, the name should be changed
    cont.add_node("test", NodeRef(None))
    assert len(cont) == 2, "Added node but length didn't increase"

    # make sure the name was changed
    assert "test1" in cont, "Wrong name when adding node"

    # check failure cases
    with pytest.raises(ValueError):
        cont.add_node(None, NodeRef(None))
    with pytest.raises(ValueError):
        cont.add_node(1, NodeRef(None))
    with pytest.raises(ValueError):
        cont.add_node("test", None)


def test_remove_node():
    cont = NodeContainer()
    cont.add_node("test", create_dummy_node()[0])

    # removing a node should decrease the length of the container
    cont.remove_node("test0")
    assert len(cont) == 0, "Removed node but length didn't decrease"

    # check failure cases
    with pytest.raises(KeyError):
        cont.remove_node("test0")
    with pytest.raises(KeyError):
        cont.remove_node(None)
    with pytest.raises(KeyError):
        cont.remove_node(1)
