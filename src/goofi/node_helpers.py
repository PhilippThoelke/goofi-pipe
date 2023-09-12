from dataclasses import dataclass, field
from multiprocessing import Process
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple

from goofi.connection import Connection
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
    """
    A reference to a node, implementing the node's counterpart in the manager. The reference contains the
    connection object to the node, and the input and output slots of the node. NodeRef instances should
    be created along with a node.

    NodeRef instances will handle the following message types with default behavior, which can be overridden
    by registering a message handler function with the `set_message_handler()` method:

    - `PING`: Responds with a `PONG` message.
    - `TERMINATE`: Terminates the node by closing the connection to it.
    - `NODE_PARAMS_REQUEST`: Raises `ValueError`. NodeRef instances should not receive this message type.

    ### Parameters
    `connection` : Connection
        The connection object to the node.
    `process` : Optional[Process]
        If the node is running in a separate process, this should be the process object. Defaults to None.
    """

    connection: Connection
    process: Optional[Process] = None

    input_slots: Dict[str, DataType] = field(default_factory=dict)
    output_slots: Dict[str, DataType] = field(default_factory=dict)
    callbacks: Dict[MessageType, Callable] = field(default_factory=dict)

    def set_message_handler(self, msg_type: MessageType, callback: Optional[Callable]) -> None:
        """
        Registers a message handler function that is called when the node sends this type of message
        to the manager.

        ### Parameters
        `msg_type` : MessageType
            The type of message to register the handler for.
        `callback` : Optional[Callable]
            Callback function: `callback(node: NodeRef, msg: Message) -> None`
        """
        self.callbacks[msg_type] = callback

    def terminate(self) -> None:
        """Terminates the node by closing the connection to it."""
        try:
            self.connection.send(Message(MessageType.TERMINATE, {}))
        except ConnectionError:
            pass
        self.connection.close()

    def _messaging_loop(self) -> None:
        """This method runs in a separate thread and handles incoming messages the node."""
        while self._alive:
            try:
                msg = self.connection.recv()
            except ConnectionError:
                # the connection was closed, consider the node dead
                self._alive = False
                self.connection.close()
                continue

            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message, got {type(msg)}")

            # if the message type has a registered callback, call it and skip built-in message handling
            if msg.type in self.callbacks:
                self.callbacks[msg.type](self, msg)
                continue

            # built-in message handling
            if msg.type == MessageType.PING:
                self.connection.send(Message(MessageType.PONG, {}))
            elif msg.type == MessageType.TERMINATE:
                self._alive = False
                self.connection.close()
            elif msg.type == MessageType.NODE_PARAMS_REQUEST:
                raise ValueError("Nodes should not send NODE_PARAMS_REQUEST messages.")

    def __post_init__(self):
        if self.connection is None:
            raise ValueError("Expected Connection, got None.")

        self._alive = True
        self._messaging_thread = Thread(target=self._messaging_loop, daemon=True)
        self._messaging_thread.start()
