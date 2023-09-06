import time
from multiprocessing import Pipe

import pytest

from goofi.data import Data, DataType
from goofi.node import InputSlot, Node, OutputSlot

from .utils import DummyNode, create_dummy_node


def test_abstract_node():
    # instantiating an abstract node should raise a TypeError
    with pytest.raises(TypeError):
        Node("test", Pipe()[0])


def test_create_node():
    _, n = create_dummy_node("test")

    # some basic checks
    assert n.name == "test", "Node name is not set correctly."
    assert n.messaging_thread.is_alive(), "Node messaging thread is not alive."
    assert n.processing_thread.is_alive(), "Node processing thread is not alive."
    assert len(n.input_slots) == 0, "Node input slots are not empty."
    assert len(n.output_slots) == 0, "Node output slots are not empty."


def test_dead_pipe():
    conn, n = create_dummy_node("test")

    # close the connection
    conn.close()
    time.sleep(0.05)

    # the node should stop its messaging thread and exit
    assert not n.messaging_thread.is_alive(), "Node messaging thread is alive."


def test_create_node_errors():
    # name is empty
    with pytest.raises(ValueError):
        DummyNode("", Pipe()[0])
    # name is None
    with pytest.raises(TypeError):
        DummyNode(None, Pipe()[0])
    # name is not str
    with pytest.raises(TypeError):
        DummyNode(1, Pipe()[0])
    # input_conn is None
    with pytest.raises(TypeError):
        DummyNode("test", None)
    # input_conn is not Connection
    with pytest.raises(TypeError):
        DummyNode("test", 1)


def test_missing_super_call():
    _, n = create_dummy_node("test", call_super=False)

    # check all properties and methods that should raise a RuntimeError if super() is not called
    with pytest.raises(RuntimeError):
        n.name
    with pytest.raises(RuntimeError):
        n.identifier
    with pytest.raises(RuntimeError):
        n.input_slots
    with pytest.raises(RuntimeError):
        n.output_slots
    with pytest.raises(RuntimeError):
        n.register_input("test", DataType.STRING)
    with pytest.raises(RuntimeError):
        n.register_output("test", DataType.STRING)


@pytest.mark.parametrize("is_input", [True, False])
def test_register_slot(is_input):
    _, n = create_dummy_node("test")

    fn = n.register_input if is_input else n.register_output
    slot_dict = n.input_slots if is_input else n.output_slots
    slot_cls = InputSlot if is_input else OutputSlot

    # register a slot, no errors should occur
    fn("slot1", DataType.STRING)
    assert len(slot_dict) == 1, "Slots should have one entry."
    assert "slot1" in slot_dict, "Slots should have a slot named 'slot1'."
    assert isinstance(slot_dict["slot1"], slot_cls), f"Slot 'slot1' should be of type {slot_cls.__name__}."
    assert slot_dict["slot1"].dtype == DataType.STRING, "Slot 'slot1' should have type DataType.STRING."

    # slot already exists
    with pytest.raises(ValueError):
        fn("slot1", DataType.STRING)

    # dtype is None
    with pytest.raises(TypeError):
        fn("slot2", None)

    # dtype is not DataType
    with pytest.raises(TypeError):
        fn("slot2", 1)

    # name is None
    with pytest.raises(TypeError):
        fn(None, DataType.STRING)

    # name is empty
    with pytest.raises(TypeError):
        fn("", DataType.STRING)

    # name is not str
    with pytest.raises(TypeError):
        fn(1, DataType.STRING)

    assert len(slot_dict) == 1, "Slots should still have one entry."


@pytest.mark.parametrize("is_input", [True, False])
@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_register_slot_dtypes(is_input, dtype):
    _, n = create_dummy_node("test")

    if is_input:
        # try to register an input slot with the given dtype
        n.register_input("slot1", dtype)
    else:
        # try to register an output slot with the given dtype
        n.register_output("slot1", dtype)


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_input_slot(dtype):
    slot = InputSlot(dtype)
    assert slot.data is None, "InputSlot data should be None."


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_output_slot(dtype):
    slot = OutputSlot(dtype)
    assert isinstance(slot.connections, dict), "OutputSlot connections should be a dict."
    assert len(slot.connections) == 0, "OutputSlot connections should be empty."
