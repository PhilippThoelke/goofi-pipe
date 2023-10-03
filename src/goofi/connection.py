import pickle
import tempfile
from abc import ABC, abstractmethod
from multiprocessing import Pipe
from multiprocessing.connection import _ConnectionBase
from typing import Callable, Dict, Tuple, Type

import zmq


class Connection(ABC):
    _CONNECTION_IDS = set()
    _BACKEND = None

    def __init__(self) -> None:
        # register a unique id for the connection
        self._id = 0
        while self._id in Connection._CONNECTION_IDS:
            self._id += 1
        Connection._CONNECTION_IDS.add(self._id)

    @staticmethod
    def get_backends() -> Dict[str, Type["Connection"]]:
        """
        List all available connection backends.
        """
        return {"zmq-tcp": TCPZeroMQConnection, "zmq-ipc": IPCZeroMQConnection, "mp": MultiprocessingConnection}

    @staticmethod
    def set_backend(backend: str) -> None:
        assert (
            backend in Connection.get_backends().keys()
        ), f"Invalid backend: {backend}. Choose from {list(Connection.get_backends().keys())}"
        Connection._BACKEND = backend

    @staticmethod
    def create() -> Tuple["Connection", "Connection"]:
        """
        Create two instances of the connection class and return them as a tuple. Both instances should be
        connected to each other using the underlying connection mechanism of the deriving class.
        """
        assert Connection._BACKEND is not None, "No backend set. Call Connection.set_backend() first."
        return Connection.get_backends()[Connection._BACKEND]._create()

    @staticmethod
    @abstractmethod
    def _create() -> Tuple["Connection", "Connection"]:
        """
        Create two instances of the connection class and return them as a tuple. Both instances should be
        connected to each other using the underlying connection mechanism of the deriving class.
        """
        raise TypeError("Abstract method")

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

    def try_send(self, obj: object) -> bool:
        """
        Try to send an object through the connection (i.e. catch any ConnectionError that might be raised).

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
    def _create() -> Tuple[Connection, Connection]:
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
        except (OSError, EOFError, TypeError):
            raise ConnectionError("Connection closed")

    def close(self) -> None:
        try:
            self.conn.close()
        except OSError:
            pass

    def __del__(self) -> None:
        """Destructor to close the connection and free any resources associated with it."""
        self.close()
        self.conn._handle = None

        # remove the connection id from the set of connection ids
        try:
            Connection._CONNECTION_IDS.remove(self._id)
        except KeyError:
            pass


class ZeroMQConnection(Connection, ABC):
    def __init__(self, push_endpoint: str, pull_endpoint: str = None) -> None:
        super().__init__()
        self.context = zmq.Context()
        self.push_socket = None
        self.pull_socket = None
        self.push_endpoint = push_endpoint
        self.pull_endpoint = pull_endpoint

    @classmethod
    def _create(cls) -> Tuple[Connection, Connection]:
        if cls.protocol() == "tcp":
            endpoint1 = TCPZeroMQConnection.unique_tcp_endpoint()
            endpoint2 = TCPZeroMQConnection.unique_tcp_endpoint()
        elif cls.protocol() == "ipc":
            endpoint1 = IPCZeroMQConnection.unique_ipc_endpoint()
            endpoint2 = IPCZeroMQConnection.unique_ipc_endpoint()
        else:
            raise ValueError(f"Invalid protocol: {cls.protocol}")
        return cls(endpoint1, endpoint2), cls(endpoint2, endpoint1)

    def init_sockets(func: Callable) -> Callable:
        """Decorator to initialize the sockets before calling the decorated function"""

        def wrapper(self, *args, **kwargs):
            if self.push_endpoint is not None and self.push_socket is None:
                # push sockets always use connect, i.e. take the role of the client
                self.push_socket = self.context.socket(zmq.PUSH)
                self.push_socket.connect(self.push_endpoint)

            if self.pull_endpoint is not None and self.pull_socket is None:
                # pull sockets always use bind, i.e. take the role of the server
                self.pull_socket = self.context.socket(zmq.PULL)
                self.pull_socket.bind(self.pull_endpoint)
            return func(self, *args, **kwargs)

        return wrapper

    @init_sockets
    def send(self, obj: object) -> None:
        try:
            self.push_socket.send(pickle.dumps(obj))
        except Exception as e:
            raise ConnectionError("Connection closed")

    @init_sockets
    def recv(self) -> object:
        try:
            return pickle.loads(self.pull_socket.recv())
        except Exception:
            raise ConnectionError("Connection closed")

    def close(self) -> None:
        try:
            if self.push_socket is not None:
                self.push_socket.close()
        except Exception:
            raise ConnectionError("Push socket closed.")

        try:
            if self.pull_socket is not None:
                self.pull_socket.close()
        except Exception:
            raise ConnectionError("Pull socket closed.")

    @abstractmethod
    def __reduce__(self):
        pass

    def __getstate__(self):
        return {"_id": self._id}

    def __setstate__(self, state):
        self._id = state["_id"]

    def __del__(self) -> None:
        """Destructor to close the connection and free any resources associated with it."""
        self.close()

        # remove the connection id from the set of connection ids
        try:
            Connection._CONNECTION_IDS.remove(self._id)
        except KeyError:
            pass

    @staticmethod
    @abstractmethod
    def protocol() -> str:
        """Return the protocol used by the connection"""
        pass

    @staticmethod
    def unique_ipc_endpoint() -> str:
        """Generate a unique temporary file path. The file is deleted immediately after creation to avoid clashes"""
        temp_file = tempfile.NamedTemporaryFile(delete=True)
        endpoint = f"ipc://{temp_file.name}"
        # delete the file
        temp_file.close()
        return endpoint

    @staticmethod
    def unique_tcp_endpoint() -> str:
        """Generate a unique TCP endpoint. The endpoint is guaranteed to be unique for the current machine"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://127.0.0.1:0")
        port = socket.getsockopt(zmq.LAST_ENDPOINT).decode("ascii").split(":")[-1].rstrip("]")
        socket.close()
        context.term()
        return f"tcp://127.0.0.1:{port}"


class TCPZeroMQConnection(ZeroMQConnection):
    @staticmethod
    def protocol() -> str:
        return "tcp"

    def __reduce__(self):
        return (TCPZeroMQConnection, (self.push_endpoint, None), self.__getstate__())


class IPCZeroMQConnection(ZeroMQConnection):
    @staticmethod
    def protocol() -> str:
        return "ipc"

    def __reduce__(self):
        return (IPCZeroMQConnection, (self.push_endpoint, None), self.__getstate__())
