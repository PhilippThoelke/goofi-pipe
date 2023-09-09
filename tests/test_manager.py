import time
from multiprocessing import Pipe

import pytest

from goofi.manager import Manager
from goofi.message import Message, MessageType


def test_creation():
    Manager()


def test_simple():
    manager = Manager()
    manager.add_node("generators.Constant")
    manager.add_node("generators.Sine")
    manager.add_node("Add")

    manager.connect("constant0", "add0", "out", "a")
    manager.connect("sine0", "add0", "out", "b")

    my_conn, node_conn = Pipe()
    manager.nodes["add0"].connection.send(
        Message(MessageType.ADD_OUTPUT_PIPE, {"slot_name_out": "out", "slot_name_in": "in", "node_connection": my_conn})
    )

    last = None
    rates = []
    data = []
    for _ in range(30):
        msg = node_conn.recv()
        data.append(msg.content["data"].data[0])

        if last is not None:
            rates.append(1 / (time.time() - last))
        last = time.time()

    # data should be between 0 and 2
    assert all(0 <= x <= 2 for x in data), "Data should be between 0 and 2."

    # mean rate should be ~30 Hz
    mean_rate = sum(rates) / len(rates)
    assert mean_rate == pytest.approx(30, abs=1), f"Mean rate should be ~30 Hz, got {mean_rate} Hz."

    #######################
    # TODO: node cleanup should happen automatically, none of this should be necessary
    manager.nodes.remove_node("sine0")
    time.sleep(0.05)
    manager.nodes.remove_node("constant0")
    time.sleep(0.05)
    manager.nodes.remove_node("add0")
    #######################
