import inspect
import time
from multiprocessing import Process
from typing import List, Type

import pytest

from goofi import connection
from goofi.message import Message, MessageType


def list_connection_backends() -> List[Type[connection.Connection]]:
    """
    List all available connection backends.
    """
    return [
        cls
        for _, cls in inspect.getmembers(connection, inspect.isclass)
        if issubclass(cls, connection.Connection) and cls is not connection.Connection
    ]


def test_abstract_create():
    with pytest.raises(TypeError):
        connection.Connection.create()


@pytest.mark.parametrize("backend", list_connection_backends())
def test_create(backend):
    conn1, conn2 = backend.create()
    assert conn1 is not None and conn2 is not None, f"{backend.__name__}.create() returned None"


@pytest.mark.parametrize("backend", list_connection_backends())
def test_super_init(backend):
    conn1, conn2 = backend.create()
    assert conn1._id and conn2._id, f"{backend.__name__} does not call super().__init__()."


@pytest.mark.parametrize("obj", [1, "test", None, [1, 2, 3], {"a": 1}, Message(MessageType.PING, {})])
@pytest.mark.parametrize("backend", list_connection_backends())
def test_send_recv(obj, backend):
    conn1, conn2 = backend.create()
    conn1.send(obj)

    if not conn2.poll():
        pytest.fail("Connection.poll() returned False")

    assert conn2.recv() == obj, "Connection.send() and Connection.recv() didn't work"


@pytest.mark.parametrize("backend", list_connection_backends())
def test_close(backend):
    conn1, conn2 = backend.create()
    conn1.close()

    # closing multiple times should not raise an error
    conn1.close()

    with pytest.raises(ConnectionError):
        conn1.send(1)
    with pytest.raises(ConnectionError):
        conn1.recv()
    with pytest.raises(ConnectionError):
        conn1.poll()

    with pytest.raises(ConnectionError):
        conn2.send(1)
    with pytest.raises(ConnectionError):
        conn2.recv()

    assert conn2.poll(), "Connection.poll() should return True on closed connection"

    conn2.close()
    conn2.close()


@pytest.mark.parametrize("backend", list_connection_backends())
def test_try_send(backend):
    conn1, conn2 = backend.create()

    assert conn2.try_send(1), "Connection.try_send() should return True on non-empty connection"

    if not conn1.poll():
        pytest.fail("Connection.poll() returned False")
    assert conn1.recv() == 1, "Connection.send() and Connection.recv() didn't work"

    # closing conn1 should raise an error on conn2.send(), but not on conn2.try_send()
    conn1.close()
    assert not conn2.try_send(1), "Connection.try_send() should return False on closed connection"

    conn2.close()


@pytest.mark.parametrize("backend", list_connection_backends())
def test_poll(backend):
    conn1, conn2 = backend.create()
    assert not conn1.poll(), "Connection.poll() should return False on empty connection"
    conn1.send(1)
    assert conn2.poll(), "Connection.poll() should return True on non-empty connection"
    conn2.recv()
    assert not conn2.poll(), "Connection.poll() should return False on empty connection"
    conn1.close()
    assert conn2.poll(), "Connection.poll() should return True on closed connection"


@pytest.mark.parametrize("timeout", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("backend", list_connection_backends())
def test_poll_timeout(timeout, backend, tol=0.001):
    conn1, _ = backend.create()

    start = time.time()
    assert not conn1.poll(timeout), "Connection.poll() should return False on empty connection"
    end = time.time()
    assert (
        abs(end - start - timeout) < tol
    ), f"Connection.poll() should block for {timeout} seconds, but blocked for {end - start} seconds"


@pytest.mark.parametrize("backend", list_connection_backends())
def test_multiproc(backend):
    def _send(conn):
        conn.send(1)

    conn1, conn2 = backend.create()
    p = Process(target=_send, args=(conn1,))
    p.start()
    p.join()

    assert conn2.poll(), "Connection.poll() should return True on non-empty connection"
    assert conn2.recv() == 1, "Connection.send() and Connection.recv() didn't work"
