import time
from multiprocessing import Pipe, Process

import pytest

from goofi.data import DataType
from goofi.message import Message, MessageType
from goofi.node import InputSlot, Node, OutputSlot
from goofi.node_helpers import NodeRef

from .utils import BrokenProcessingNode, DummyNode, create_dummy_node


def test_abstract_node():
    # instantiating an abstract node should raise a TypeError
    with pytest.raises(TypeError):
        Node(Pipe()[0])


def test_create_node():
    ref, n = create_dummy_node()

    # some basic checks
    assert len(n.input_slots) == 0, "Node input slots are not empty."
    assert len(n.output_slots) == 0, "Node output slots are not empty."
    assert n.messaging_thread.is_alive(), "Node messaging thread is not alive."
    assert n.processing_thread.is_alive(), "Node processing thread is not alive."

    # clean up dangling threads
    ref.terminate()


def test_dead_pipe():
    conn, node_pipe = Pipe()
    n = DummyNode(node_pipe)

    # close the connection
    conn.close()
    time.sleep(0.05)

    # the node should stop its messaging thread and exit
    assert not n.messaging_thread.is_alive(), "Node messaging thread is alive."


def test_create_node_errors():
    # connection is None
    with pytest.raises(TypeError):
        DummyNode(None)
    # connection is not Connection
    with pytest.raises(TypeError):
        DummyNode(1)


def test_missing_super_call():
    ref, n = create_dummy_node(call_super=False)

    # check all properties and methods that should raise a RuntimeError if super() is not called
    with pytest.raises(RuntimeError):
        n.input_slots
    with pytest.raises(RuntimeError):
        n.output_slots
    with pytest.raises(RuntimeError):
        n.register_input("test", DataType.STRING)
    with pytest.raises(RuntimeError):
        n.register_output("test", DataType.STRING)

    # clean up dangling threads
    ref.terminate()


@pytest.mark.parametrize("is_input", [True, False])
def test_register_slot(is_input):
    ref, n = create_dummy_node()

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

    # clean up dangling threads
    ref.terminate()


@pytest.mark.parametrize("is_input", [True, False])
@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_register_slot_dtypes(is_input, dtype):
    ref, n = create_dummy_node()

    if is_input:
        # try to register an input slot with the given dtype
        n.register_input("slot1", dtype)
    else:
        # try to register an output slot with the given dtype
        n.register_output("slot1", dtype)

    # clean up dangling threads
    ref.terminate()


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_input_slot(dtype):
    slot = InputSlot(dtype)
    assert slot.data is None, "InputSlot data should be None."


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_output_slot(dtype):
    slot = OutputSlot(dtype)
    assert isinstance(slot.connections, list), "OutputSlot connections should be a list."
    assert len(slot.connections) == 0, "OutputSlot connections should be empty."


def test_ping_pong():
    messages = []

    def callback(_: NodeRef, msg: Message):
        messages.append(msg)

    ref, _ = create_dummy_node()
    ref.set_message_handler(MessageType.PONG, callback)

    ref.connection.send(Message(MessageType.PING, {}))
    time.sleep(0.01)
    assert len(messages) == 1, "Node should have received one message."
    assert messages[0].type == MessageType.PONG, "Node should have received a pong message."

    # clean up dangling threads
    ref.terminate()


def test_terminate():
    ref, n = create_dummy_node()
    ref.connection.send(Message(MessageType.TERMINATE, {}))
    time.sleep(0.01)
    assert not n.alive, "Node should be dead after receiving terminate message."


def test_terminate_func():
    ref, n = create_dummy_node()
    ref.terminate()
    time.sleep(0.01)
    assert not n.alive, "Node should be dead after receiving terminate message."


def test_multiproc():
    conn1, conn2 = Pipe()
    proc = Process(target=DummyNode, args=(conn2,), daemon=True)
    proc.start()

    time.sleep(0.1)  # TODO: 0.1 is quite long but required on Windows, Linux works with 0.01
    # make sure the node is not dead
    assert proc.is_alive(), "Process should be alive."

    # send a terminate message
    conn1.send(Message(MessageType.TERMINATE, {}))
    time.sleep(0.1)  # TODO: 0.1 is quite long but required on Windows, Linux works with 0.01
    # make sure the node is dead
    assert not proc.is_alive(), "Process should be dead."


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_broken_processing():
    conn1, conn2 = Pipe()
    n = BrokenProcessingNode(conn2)

    # manually trigger processing once
    n.process_flag.set()
    time.sleep(0.01)

    # the processing thread should be dead
    assert not n.processing_thread.is_alive(), "Processing thread should be dead."

    # the node and messaging thread should be alive
    assert n.alive, "Node should be alive."
    assert n.messaging_thread.is_alive(), "Messaging thread should be alive."

    # sending a message should restart the processing thread
    conn1.send(Message(MessageType.PING, {}))
    time.sleep(0.01)
    assert n.processing_thread.is_alive(), "Processing thread should be alive."


def test_node_params_request_empty():
    messages = []

    def callback(_: NodeRef, msg: Message):
        messages.append(msg)

    ref, _ = create_dummy_node()
    ref.set_message_handler(MessageType.NODE_PARAMS, callback)

    ref.connection.send(Message(MessageType.NODE_PARAMS_REQUEST, {}))
    time.sleep(0.01)
    assert len(messages) == 1, "Node should have received one message."
    assert messages[0].type == MessageType.NODE_PARAMS, "Node should have received a node params message."

    # the dummy node has no params, no input slots and no output slots
    assert messages[0].content["params"] == dict(), "params should be empty for DummyNode."
    assert messages[0].content["input_slots"] == dict(), "input_slots should be empty for DummyNode."
    assert messages[0].content["output_slots"] == dict(), "output_slots should be empty for DummyNode."

    # clean up dangling threads
    ref.terminate()


def test_node_params_request_multiple():
    messages = []

    def callback(_: NodeRef, msg: Message):
        messages.append(msg)

    in_slots = {"in1": DataType.STRING, "in2": DataType.FLOAT_1D}
    out_slots = {"out1": DataType.FLOAT_1D, "out2": DataType.FLOAT_2D}

    ref, _ = create_dummy_node(input_slots=in_slots, output_slots=out_slots)
    ref.set_message_handler(MessageType.NODE_PARAMS, callback)

    ref.connection.send(Message(MessageType.NODE_PARAMS_REQUEST, {}))
    time.sleep(0.01)
    assert len(messages) == 1, "Node should have received one message."
    assert messages[0].type == MessageType.NODE_PARAMS, "Node should have received a node params message."

    # the dummy node has no params, no input slots and no output slots
    assert messages[0].content["params"] == dict(), "params should be empty for DummyNode."
    assert messages[0].content["input_slots"] == in_slots, "input_slots should be empty for DummyNode."
    assert messages[0].content["output_slots"] == out_slots, "output_slots should be empty for DummyNode."

    # clean up dangling threads
    ref.terminate()


@pytest.mark.parametrize("dtype1", DataType.__members__.values())
@pytest.mark.parametrize("dtype2", DataType.__members__.values())
def test_node_params_request_dtypes(dtype1, dtype2):
    messages = []

    def callback(_: NodeRef, msg: Message):
        messages.append(msg)

    ref, _ = create_dummy_node(input_slots={"in": dtype1}, output_slots={"out": dtype2})
    ref.set_message_handler(MessageType.NODE_PARAMS, callback)

    ref.connection.send(Message(MessageType.NODE_PARAMS_REQUEST, {}))
    time.sleep(0.01)
    assert len(messages) == 1, "Node should have received one message."
    assert messages[0].type == MessageType.NODE_PARAMS, "Node should have received a node params message."

    # the dummy node has no params, no input slots and no output slots
    assert messages[0].content["params"] == dict(), "params should be empty for DummyNode."
    assert messages[0].content["input_slots"] == {"in": dtype1}, "input_slots should have one item."
    assert messages[0].content["output_slots"] == {"out": dtype2}, "output_slots should have one item."

    # clean up dangling threads
    ref.terminate()
