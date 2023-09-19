import platform
import time

import pytest

import goofi
from goofi.connection import MultiprocessingConnection
from goofi.manager import Manager
from goofi.message import Message, MessageType


def test_creation():
    Manager()


def test_main():
    goofi.manager.main(1, ["--headless"])


@pytest.mark.skipif(platform.system() == "Windows", reason="Multiprocessing is very slow on Windows.")
def test_simple():
    manager = Manager()
    manager.add_node("Constant", "generators")
    manager.add_node("Sine", "generators")
    manager.add_node("Add", "array")

    manager.add_link("constant0", "add0", "out", "a")
    manager.add_link("sine0", "add0", "out", "b")

    my_conn, node_conn = MultiprocessingConnection.create()
    manager.nodes["add0"].connection.send(
        Message(MessageType.ADD_OUTPUT_PIPE, {"slot_name_out": "out", "slot_name_in": "in", "node_connection": my_conn})
    )

    last = None
    rates = []
    data = []
    for _ in range(10):
        if not node_conn.poll(0.5):
            raise TimeoutError("No message received from node.")
        msg = node_conn.recv()
        data.append(msg.content["data"].data[0])

        if last is not None:
            rates.append(1 / (time.time() - last))
        last = time.time()

    # data should be between 0 and 2
    assert all(0 <= x <= 2 for x in data), "Data should be between 0 and 2."

    # mean rate should be ~30 Hz
    mean_rate = sum(rates) / len(rates)
    assert mean_rate == pytest.approx(30, abs=0.5), f"Mean rate should be ~30 Hz, got {mean_rate} Hz."

    # clean up
    manager.terminate()
