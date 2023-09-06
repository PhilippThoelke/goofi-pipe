import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Connection
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """
    The type of a message. The message type determines the content of the message.

    - `ADD_OUTPUT_PIPE`: Sent by the manager to a node to add an output pipe to an output slot.
        - `slot_name` (str): The name of the output slot.
        - `connection_id` (str): The identifier of the node to connect to.
        - `node_connection` (Connection): The connection object to the node.
    - `REMOVE_OUTPUT_PIPE`: Sent by the manager to a node to remove an output pipe from an output slot.
        - `slot_name` (str): The name of the output slot.
        - `connection_id` (str): The identifier of the node to disconnect from.
    - `DATA`: Sent by one node to another and contains data sent from an output slot to an input slot.
        - `slot_name` (str): The name of the target input slot.
        - `data` (Data): The data object. See the `Data` class for more information.
    """

    ADD_OUTPUT_PIPE = 1
    REMOVE_OUTPUT_PIPE = 2
    DATA = 3


@dataclass
class Message:
    """
    A message is used to send information between nodes. Each message has a type, an origin id, and content. The
    content field is a dict that contains the message content. The content field must contain the correct fields for
    the message type.

    ### Parameters
    `type` : MessageType
        The type of the message.
    `origin_id` : str
        The identifier of the node that sent the message.
    `content` : Dict[str, Any]
        The content of the message with required fields for the message type.
    """

    type: MessageType
    origin_id: str
    content: Dict[str, Any]

    def check_content(self):
        """
        Check that the message content is valid. The content field must be a dict, and it must contain the correct
        fields for the message type.
        """
        if not isinstance(self.content, dict):
            raise ValueError(f"Expected dict, got {type(self.content)}")
        if "slot_name" not in self.content or not isinstance(self.content["slot_name"], str):
            raise ValueError("Message content must contain slot_name with type str")

        if self.type == MessageType.ADD_OUTPUT_PIPE:
            if "connection_id" not in self.content or not isinstance(self.content["connection_id"], str):
                raise ValueError("Message content must contain connection_id with type str")
            if "node_connection" not in self.content or not isinstance(self.content["node_connection"], Connection):
                raise ValueError("Message content must contain node_connection with type Connection")
        elif self.type == MessageType.REMOVE_OUTPUT_PIPE:
            if "connection_id" not in self.content or not isinstance(self.content["connection_id"], str):
                raise ValueError("Message content must contain connection_id with type str")
        elif self.type == MessageType.DATA:
            if "data" not in self.content or not isinstance(self.content["data"], Data):
                raise ValueError("Message content must contain data with type Data")


class DataType(Enum):
    """
    The type of data contained in a data object. The data type determines the type of the data field in the data
    object.

    - `FLOAT_1D`: A 1D numpy array of floats.
    - `FLOAT_2D`: A 2D numpy array of floats.
    - `FLOAT_3D`: A 3D numpy array of floats.
    - `STRING`: A string.
    """

    FLOAT_1D = 1
    FLOAT_2D = 2
    FLOAT_3D = 3
    STRING = 4

    def empty(self) -> Any:
        """
        Get an empty numpy array of the correct type for this data type.
        """
        if self == DataType.FLOAT_1D:
            return np.empty(0)
        elif self == DataType.FLOAT_2D:
            return np.empty((0, 0))
        elif self == DataType.FLOAT_3D:
            return np.empty((0, 0, 0))
        elif self == DataType.STRING:
            return ""
        else:
            raise ValueError(f"Unknown data type {self}")


@dataclass
class Data:
    """
    Data objects are used to send data between nodes. They contain the data object (`data`) itself, the data type
    (`dtype`), and some metadata (`meta`). The data types are limited to the types defined in the `DataType` enum.
    The data field must be of the correct type for the data type. The metadata field is a dict that can contain
    any metadata about the data. The metadata field can be used to store information about the data, such as
    channel names, sampling frequencies, etc.

    ### Parameters
    `dtype` : DataType
        The data type of the data object.
    `data` : Any
        A data object matching the data type.
    `meta` : Dict[str, Any]
        The metadata dictionary.
    """

    dtype: DataType
    data: Any
    meta: Dict[str, Any]

    def check_data(self):
        """
        Check that the data type is valid. The data field must be of the correct type for the data type.
        """
        if self.dtype is None or not isinstance(self.dtype, DataType):
            raise ValueError(f"Expected DataType, got {type(self.dtype)}")
        if self.data is None:
            raise ValueError("Expected data object, got None")
        if self.meta is None or not isinstance(self.meta, dict):
            raise ValueError(f"Expected metadata dict, got {type(self.meta)}")

        if self.dtype == DataType.STRING:
            # make sure it's a string
            if not isinstance(self.data, str):
                raise ValueError(f"Expected string, got {type(self.data)}")
        elif isinstance(self.data, np.ndarray):
            # all other types are numpy arrays, so make sure it's a numpy array
            if self.dtype == DataType.FLOAT_1D:
                # data should be a 1D array
                if self.data.ndim != 1:
                    raise ValueError(f"Expected 1D array, got {self.data.ndim}D array")
            elif self.dtype == DataType.FLOAT_2D:
                # data should be a 2D array
                if self.data.ndim != 2:
                    raise ValueError(f"Expected 2D array, got {self.data.ndim}D array")
            elif self.dtype == DataType.FLOAT_3D:
                # data should be a 3D array
                if self.data.ndim != 3:
                    raise ValueError(f"Expected 3D array, got {self.data.ndim}D array")
            else:
                raise ValueError(f"Unknown data type {self.dtype}")
        else:
            raise ValueError(f"Expected numpy array, got {type(self.data)}")

        # TODO: add better metadata checks


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
    """

    dtype: DataType
    data: Optional[Data] = None


@dataclass
class OutputSlot:
    dtype: DataType
    connections: Dict[str, Connection] = field(default_factory=dict)


def require_init(func: Callable) -> Callable:
    """
    Decorator that checks if `super().__init__()` has been called in the `__init__()` method of the class
    that the decorated method belongs to. This is used to make sure that the base class is initialized before
    accessing any of its attributes.

    ### Parameters
    `func` : Callable
        The method to decorate.

    ### Returns
    `Callable`
        The decorated method.
    """

    def wrapper(self, *args, **kwargs):
        # check if the base class is initialized
        if not hasattr(self, "_base_initialized"):
            raise RuntimeError("Make sure to call super().__init__() in your node's __init__ method.")
        return func(self, *args, **kwargs)

    return wrapper


class Node(ABC):
    """
    The base class for all nodes. A node is a processing unit that can receive data from other nodes, process the
    data, and send the processed data to other nodes. A node can have any number of input and output slots. Each
    input slot can receive data from one output slot, and each output slot can send data to any number of input
    slots.

    ### Parameters
    `name` : str
        The name of the node, which should be unique in the session.
    `input_conn` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    """

    def __init__(self, name: str, input_conn: Connection) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Expected str, got {type(name)}.")
        if not name:
            raise ValueError("Name cannot be empty.")
        if not isinstance(input_conn, Connection):
            raise TypeError(f"Expected Connection, got {type(input_conn)}.")

        # base class is initialized
        self._base_initialized = True

        self._name = name
        self.input_conn = input_conn

        # initialize input and output slots
        self._input_slots = dict()
        self._output_slots = dict()

        # initialize node update flag
        self.process_flag = Event()

        # initialize message handling thread
        self.messaging_thread = Thread(target=self.messaging_loop)
        self.messaging_thread.start()

        # initialize node update thread
        self.processing_thread = Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def messaging_loop(self):
        """
        This method runs in a separate thread and handles incoming messages from the manager, or other nodes.
        """
        while True:
            try:
                msg = self.input_conn.recv()
            except EOFError:
                # the connection was closed
                # TODO: handle this properly
                break

            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message, got {type(msg)}")

            # TODO: handle the incoming message

    def processing_loop(self):
        """
        This method runs in a separate thread and handles the processing of input data and sending of
        output data to other nodes.
        """
        while True:
            # wait for a trigger
            self.process_flag.wait()
            self.process_flag.clear()

            # gather input data
            input_data = {name: slot.data for name, slot in self.input_slots.items()}

            try:
                # process data
                output_data = self.process(**input_data)
            except Exception as e:
                # the process method raised an exception
                raise RuntimeError("Error while processing data: " + str(e)) from e

            if not isinstance(output_data, dict):
                # process didn't return a dict
                raise TypeError(f"The process method didn't return a dict. Got {type(output_data)}.")

            if set(output_data.keys()) != set(self.output_slots.keys()):
                # process returned a dict with the wrong keys
                missing_fields = set(self.output_slots.keys()) - set(output_data.keys())
                extra_fields = set(output_data.keys()) - set(self.output_slots.keys())
                raise ValueError(
                    "Mismatch between expected and received output fields. "
                    f"Missing fields: {missing_fields}. Extra fields: {extra_fields}"
                )

            # send output data
            for name, data in output_data.items():
                # make sure the data is of the correct type
                if self.output_slots[name].dtype != data.dtype:
                    raise RuntimeError(f"Data type mismatch for output slot {name}")

                # send the data to all connected nodes
                for conn in self.output_slots[name].connections:
                    conn.send(Message(MessageType.DATA, self.identifier, data))

    @require_init
    def _register_slot(self, name: str, dtype: DataType, is_input: bool):
        if not isinstance(name, str) or not name:
            raise TypeError(f"Expected non-empty str, got {type(name)}.")
        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected DataType, got {type(dtype)}.")

        # use input or output slot dict depending on the is_input flag
        slot_dict = self._input_slots if is_input else self._output_slots

        if name in slot_dict:
            raise ValueError(f"Input slot {name} already exists")

        # create the new slot
        slot = InputSlot(dtype) if is_input else OutputSlot(dtype)
        slot_dict[name] = slot

    def register_input(self, name: str, dtype: DataType):
        """
        Register an input slot with the given name and data type.

        ### Parameters
        `name` : str
            The name of the input slot.
        `dtype` : DataType
            The data type of the input slot.
        """
        self._register_slot(name, dtype, True)

    def register_output(self, name: str, dtype: DataType):
        """
        Register an output slot with the given name and data type.

        ### Parameters
        `name` : str
            The name of the output slot.
        `dtype` : DataType
            The data type of the output slot.
        """
        self._register_slot(name, dtype, False)

    @property
    @require_init
    def name(self) -> str:
        return self._name

    @property
    @require_init
    def input_slots(self) -> Dict[str, InputSlot]:
        return self._input_slots

    @property
    @require_init
    def output_slots(self) -> Dict[str, OutputSlot]:
        return self._output_slots

    @property
    @require_init
    def identifier(self) -> str:
        return str(os.getpid())

    @abstractmethod
    def process(self):  # TODO: define process method signature
        pass
