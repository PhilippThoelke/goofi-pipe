import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from threading import Event, Thread
from typing import Callable, Dict, Optional

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
    """

    dtype: DataType
    data: Optional[Data] = None


@dataclass
class OutputSlot:
    """
    An output slot is used to send data to an input slot. It contains the data type (`dtype`) and a dictionary
    of connections (`connections`). The data type is used to check that the data sent out from the output slot
    is of the correct type. The connections dictionary maps names of target input slots to the connection
    objects that are used to send data to other nodes.

    ### Parameters
    `dtype` : DataType
        The data type of the output slot.
    `connections` : Dict[str, Connection]
        The dictionary of connections. Defaults to an empty dictionary.
    """

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


@dataclass
class NodeRef:
    connection: Connection


class Node(ABC):
    """
    The base class for all nodes. A node is a processing unit that can receive data from other nodes, process the
    data, and send the processed data to other nodes. A node can have any number of input and output slots. Each
    input slot can receive data from one output slot, and each output slot can send data to any number of input
    slots.

    ### Parameters
    `connection` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    """

    def __init__(self, connection: Connection) -> None:
        if not isinstance(connection, Connection):
            raise TypeError(f"Expected Connection, got {type(connection)}.")

        # base class is initialized
        self._base_initialized = True

        self.connection = connection

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
                msg = self.connection.recv()
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
                    conn.send(Message(MessageType.DATA, data))

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
    def input_slots(self) -> Dict[str, InputSlot]:
        return self._input_slots

    @property
    @require_init
    def output_slots(self) -> Dict[str, OutputSlot]:
        return self._output_slots

    @abstractmethod
    def process(self):  # TODO: define process method signature
        pass
