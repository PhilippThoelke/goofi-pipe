import pytest

from goofi.manager import NodeContainer


def test_creation():
    cont = NodeContainer()
    assert len(cont) == 0


def test_assignment():
    cont = NodeContainer()
    # the container shouldn't allow item assignment
    with pytest.raises(TypeError):
        cont["test"] = 1
