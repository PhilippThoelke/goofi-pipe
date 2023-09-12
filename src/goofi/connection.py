import inspect
from abc import ABC, abstractmethod
from multiprocessing import Pipe
from multiprocessing.connection import _ConnectionBase
from typing import List, Tuple, Type

import goofi


class Connection(ABC):
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


class MultiprocessingConnection(Connection):
    def __init__(self, conn: _ConnectionBase) -> None:
        self.conn = conn

    @staticmethod
    def create() -> Tuple[Connection, Connection]:
        conn1, conn2 = Pipe()
        return MultiprocessingConnection(conn1), MultiprocessingConnection(conn2)

    def send(self, obj: object) -> None:
        try:
            self.conn.send(obj)
        except OSError:
            raise ConnectionError("Connection closed")

    def recv(self) -> object:
        try:
            return self.conn.recv()
        except (OSError, EOFError):
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


def list_backends() -> List[Type[Connection]]:
    """
    List all available connection backends.
    """
    return [
        cls
        for _, cls in inspect.getmembers(goofi.connection, inspect.isclass)
        if issubclass(cls, Connection) and cls is not Connection
    ]
