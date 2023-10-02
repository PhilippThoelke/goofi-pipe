import pickle
from abc import ABC, abstractmethod
from multiprocessing import Pipe
from multiprocessing.connection import _ConnectionBase
from typing import Tuple

_CONNECTION_IDS = set()


class Connection(ABC):
    def __init__(self) -> None:
        # register a unique id for the connection
        self._id = 0
        while self._id in _CONNECTION_IDS:
            self._id += 1
        _CONNECTION_IDS.add(self._id)

    @staticmethod
    @abstractmethod
    def create(cls) -> Tuple["Connection", "Connection"]:
        """
        Create 2 instances of the connection class and return them as a tuple. Both instances should be connected
        to each other using the underlying connection mechanism of the deriving class.
        """
        pass

    @abstractmethod
    def send(self, obj: object) -> None:
        """
        Send an object through the connection (requires the object to be picklable).

        This method should raise a ConnectionError if either it, or the receiving end of the connection is closed.

        ### Parameters
        `obj` : object
            The object to send through the connection.
        """
        pass

    @abstractmethod
    def recv(self) -> object:
        """
        Receive an object through the connection.

        This method should raise a ConnectionError if either it, or the sending end of the connection is closed.

        ### Returns
        `object`
            The object received through the connection.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the connection and free any resources associated with it.
        """
        pass

    @abstractmethod
    def poll(self, timeout: float = 0.0) -> bool:
        """
        Check if there is data available on the connection. If `timeout` is 0, the method should return immediately.
        If `timeout > 0`, the method should block for `timeout` seconds, and return True if data is available, or
        False otherwise.
        """
        pass

    def try_send(self, obj: object) -> bool:
        """
        Try to send an object through the connection. If the connection is closed, return False.

        ### Parameters
        `obj` : object
            The object to send through the connection.

        ### Returns
        `bool`
            True if the object was sent, False otherwise.
        """
        try:
            self.send(obj)
            return True
        except ConnectionError:
            return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            return False
        return self._id == other._id


class MultiprocessingConnection(Connection):
    def __init__(self, conn: _ConnectionBase) -> None:
        super().__init__()
        self.conn = conn

    @staticmethod
    def create() -> Tuple[Connection, Connection]:
        conn1, conn2 = Pipe()
        return MultiprocessingConnection(conn1), MultiprocessingConnection(conn2)

    def send(self, obj: object) -> None:
        try:
            self.conn.send(obj)
        except (OSError, BrokenPipeError):
            raise ConnectionError("Connection closed")

    def recv(self) -> object:
        try:
            return self.conn.recv()
        except (OSError, EOFError, TypeError, pickle.UnpicklingError, ValueError):
            raise ConnectionError("Connection closed")

    def close(self) -> None:
        try:
            self.conn.close()
        except OSError:
            pass

    def poll(self, timeout: float = 0.0) -> bool:
        try:
            return self.conn.poll(timeout)
        except OSError:
            raise ConnectionError("Connection closed")
