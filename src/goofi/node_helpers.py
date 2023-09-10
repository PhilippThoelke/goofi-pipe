from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple

from goofi.data import Data, DataType
from goofi.message import Message, MessageType


@dataclass
class InputSlot:
    """
    An input slot is used to receive data from an output slot. It contains the data type (`dtype`) and a data
    object (`data`). The data object is updated when data is received from an output slot. The data type is used
    to check that the data received from the output slot is of the correct type.

    ### Parameters
    `dtype` : DataType
        The data type of the input slot.
    `data` : Optional[Data]
        The data object of the input slot. Defaults to None.
    `trigger_update` : bool
        If True, the node will automatically trigger processing when the data object is updated.
    """

    dtype: DataType
    data: Optional[Data] = None
    trigger_update: bool = True


@dataclass
class OutputSlot:
    """
    An output slot is used to send data to an input slot. It contains the data type (`dtype`) and a list of
    connections (`connections`). The data type is used to check that the data sent out from the output slot
    is of the correct type. The connections list maps names of target input slots to the connection
    objects that are used to send data to other nodes.

    ### Parameters
    `dtype` : DataType
        The data type of the output slot.
    `connections` : List[Tuple[str, Connection]]
        A list of tuples containing the name of the target input slot and the connection object of the target.
    """

    dtype: DataType
    connections: List[Tuple[str, Connection]] = field(default_factory=list)


@dataclass
class NodeRef:
    connection: Connection
    input_slots: Dict[str, DataType] = field(default_factory=dict)
    output_slots: Dict[str, DataType] = field(default_factory=dict)

    callbacks: Dict[object, Callable] = field(default_factory=dict)

    def register_callback(self, obj: object, callback: Callable) -> None:
        """
        Registers a callback function that is called when the node receives data.

        ### Parameters
        `obj` : object
            The object that the callback function belongs to.
        `callback` : Callable
            The callback function.
        """
        if not callable(callback):
            raise TypeError(f"Expected callable, got {type(callback)}")
        self.callbacks[obj] = callback

    def unregister_callback(self, obj: object) -> None:
        """
        Unregisters a callback function.

        ### Parameters
        `obj` : object
            The object that the callback function belongs to. The callback takes a single argument,
            the message.
        """
        del self.callbacks[obj]

    def terminate(self) -> None:
        """Terminates the node (both reference and actual node)."""
        try:
            self.connection.send(Message(MessageType.TERMINATE, {}))
        except BrokenPipeError:
            pass
        self.connection.close()

    def _messaging_loop(self) -> None:
        """This method runs in a separate thread and handles incoming messages the node."""
        while self._alive:
            try:
                msg = self.connection.recv()
            except (EOFError, ConnectionResetError):
                # the connection was closed, consider the node dead
                self._alive = False
                continue

            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message, got {type(msg)}")

            for callback in self.callbacks.values():
                callback(msg)

            if msg.type == MessageType.PING:
                self.connection.send(Message(MessageType.PONG, {}))
            elif msg.type == MessageType.TERMINATE:
                self._alive = False

    def __post_init__(self):
        if self.connection is None:
            raise ValueError("Expected Connection, got None")

        self._alive = True
        self._messaging_thread = Thread(target=self._messaging_loop, daemon=True)
        self._messaging_thread.start()
