import platform
import time
from multiprocessing import Manager as MPManager
from os import path

import pytest

import goofi
from goofi.connection import Connection
from goofi.manager import Manager
from goofi.message import Message, MessageType


def create_simple_manager() -> Manager:
    """
    Creates a simple manager with a constant node, a sine node, and an add node.

    ### Returns
    manager : Manager
        The manager object.
    """
    manager = Manager()
    manager.add_node("ConstantArray", "inputs")
    manager.add_node("Sine", "inputs")
    manager.add_node("Operation", "array")

    manager.add_link("constantarray0", "operation0", "out", "a")
    manager.add_link("sine0", "operation0", "out", "b")
    return manager


def test_creation():
    Manager()


def test_main():
    goofi.manager.main(1, ["--headless"])


@pytest.mark.skipif(platform.system() == "Windows", reason="Multiprocessing is very slow on Windows.")
@pytest.mark.parametrize("comm_backend", Connection.get_backends().keys())
def test_simple(comm_backend):
    if comm_backend.startswith("zmq"):
        # TODO: make sure zmq backend works
        pytest.skip("ZeroMQ backend still has some issues.")

    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    manager = create_simple_manager()
    my_conn, node_conn = Connection.create()
    manager.nodes["operation0"].connection.send(
        Message(MessageType.ADD_OUTPUT_PIPE, {"slot_name_out": "out", "slot_name_in": "in", "node_connection": my_conn})
    )

    last = None
    rates = []
    data = []
    for _ in range(10):
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


def test_save_empty(tmpdir):
    manager = Manager()

    # if path is a file, save to that file
    manager.save(path.join(tmpdir, "test.gfi"))
    assert path.exists(path.join(tmpdir, "test.gfi")), "Expected file test.gfi to exist."

    # clean up
    manager.terminate()


def test_save_extension(tmpdir):
    manager = create_simple_manager()

    # make sure the file gets the correct extension
    manager.save(path.join(tmpdir, "test2"))
    assert path.exists(path.join(tmpdir, "test2.gfi")), "Expected file extension to be set to .gfi"

    # clean up
    manager.terminate()


@pytest.mark.parametrize("overwrite", [True, False])
def test_save_simple(overwrite, tmpdir):
    manager = create_simple_manager()

    tmpdir = str(tmpdir)

    # if path is a file, save to that file
    manager.save(path.join(tmpdir, "test.gfi"), overwrite=overwrite)
    assert path.exists(path.join(tmpdir, "test.gfi")), "Expected file test.gfi to exist."

    time.sleep(0.1)

    # if file already exists, raise FileExistsError
    if overwrite:
        manager.save(path.join(tmpdir, "test.gfi"), overwrite=True)
        assert path.exists(path.join(tmpdir, "test.gfi")), "test.gfi should still be there."
    else:
        with pytest.raises(FileExistsError):
            # overwrite is False by default
            manager.save(path.join(tmpdir, "test.gfi"))

    # clean up
    manager.terminate()


def test_save_simple_dir(tmpdir):
    manager = create_simple_manager()

    with pytest.raises(ValueError):
        # if path is not a string, raise an error (tmpdir is a Path object)
        manager.save(tmpdir)

    tmpdir = str(tmpdir)

    # if path is a directory, save to untitled0.gfi
    manager.save(tmpdir)
    assert path.exists(path.join(tmpdir, "untitled0.gfi")), "Expected file untitled0.gfi to exist."

    time.sleep(0.1)

    # if untitled0.gfi exists in the directory, save to untitled1.gfi
    manager.save(tmpdir)
    assert path.exists(path.join(tmpdir, "untitled1.gfi")), "Expected file untitled1.gfi to exist."

    # clean up
    manager.terminate()
