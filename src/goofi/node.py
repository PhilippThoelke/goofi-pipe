import time
from abc import ABC, abstractmethod
from goofi.connection import Connection
from threading import Event, Thread
from typing import Callable, Dict

from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.node_helpers import InputSlot, OutputSlot


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
        if not hasattr(self, "_alive"):
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
    `connection` : Connection
        The input connection to the node. This is used to receive messages from the manager, or other nodes.
    `autotrigger` : bool
        If True, the node will automatically trigger processing regardless of whether or not it received data.
    """

    def __init__(self, connection: Connection, autotrigger: bool = False) -> None:
        print(connection, Connection, isinstance(connection, Connection))
        if not isinstance(connection, Connection):
            raise TypeError(f"Expected Connection, got {type(connection)}.")
        self._alive = True

        # store arguments
        self.connection = connection
        self.autotrigger = autotrigger

        # initialize input and output slots
        self._input_slots = dict()
        self._output_slots = dict()

        # initialize node flags
        self.process_flag = Event()
        if self.autotrigger:
            self.process_flag.set()

        # initialize message handling thread
        self.messaging_thread = Thread(target=self._messaging_loop)
        self.messaging_thread.start()

        # initialize data processing thread
        self.processing_thread = Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _messaging_loop(self):
        """
        This method runs in a separate thread and handles incoming messages from the manager, or other nodes.
        """
        while self.alive:
            try:
                msg = self.connection.recv()
            except (EOFError, ConnectionResetError):
                # the connection was closed, consider the node dead
                self._alive = False
                self.connection.close()
                continue

            # potentially restart the processing thread
            if not self.processing_thread.is_alive():
                self.processing_thread = Thread(target=self._processing_loop, daemon=True)
                self.processing_thread.start()

            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message, got {type(msg)}")

            if msg.type == MessageType.PING:
                self.connection.send(Message(MessageType.PONG, {}))
            elif msg.type == MessageType.TERMINATE:
                self._alive = False
            elif msg.type == MessageType.ADD_OUTPUT_PIPE:
                slot = self.output_slots[msg.content["slot_name_out"]]
                slot.connections.append((msg.content["slot_name_in"], msg.content["node_connection"]))
            elif msg.type == MessageType.DATA:
                slot = self.input_slots[msg.content["slot_name"]]
                slot.data = msg.content["data"]
                if slot.trigger_update:
                    self.process_flag.set()
            elif msg.type == MessageType.NODE_PARAMS_REQUEST:
                self.connection.send(
                    Message(
                        MessageType.NODE_PARAMS,
                        {
                            "params": dict(),
                            "input_slots": {name: slot.dtype for name, slot in self.input_slots.items()},
                            "output_slots": {name: slot.dtype for name, slot in self.output_slots.items()},
                        },
                    )
                )
            else:
                # TODO: handle the incoming message
                raise NotImplementedError(f"Message type {msg.type} not implemented.")

    def _processing_loop(self):
        """
        This method runs in a separate thread and handles the processing of input data and sending of
        output data to other nodes.
        """
        last_update = 0
        while self.alive:
            # wait for a trigger
            self.process_flag.wait()
            # clear the trigger if autotrigger is False
            if not self.autotrigger:
                self.process_flag.clear()

            # limit the update rate to 30 Hz
            if time.time() - last_update < 1 / 30:
                sleep_time = 1 / 30 - (time.time() - last_update)
                time.sleep(sleep_time)
            last_update = time.time()

            # gather input data
            input_data = {name: slot.data for name, slot in self.input_slots.items()}

            # process data
            output_data = self.process(**input_data)

            # if process returns None, skip sending output data
            if output_data is None:
                # TODO: make sure this is the correct behavior
                continue

            # check that the process method returned a dict
            if not isinstance(output_data, dict):
                raise TypeError(f"The process method didn't return a dict. Got {type(output_data)}.")

            # check that the output data contains the correct fields
            if missing := set(self.output_slots.keys()) - set(output_data.keys()):
                raise ValueError(f"Missing output fields: {missing}")

            # TODO: handle extra fields in output data
            # extra_fields = list(set(output_data.keys()) - set(self.output_slots.keys()))

            # send output data
            for name in self.output_slots.keys():
                data = output_data[name]

                # make sure the data is of the correct type
                if self.output_slots[name].dtype != data.dtype:
                    raise RuntimeError(f"Data type mismatch for output slot {name}")

                # send the data to all connected nodes
                for target_slot, conn in self.output_slots[name].connections:
                    msg = Message(MessageType.DATA, {"data": data, "slot_name": target_slot})
                    try:
                        conn.send(msg)
                    except BrokenPipeError:
                        # TODO: broken pipe indicates that the target node is dead, handle this
                        raise

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
    def alive(self) -> bool:
        return self._alive

    @property
    @require_init
    def input_slots(self) -> Dict[str, InputSlot]:
        return self._input_slots

    @property
    @require_init
    def output_slots(self) -> Dict[str, OutputSlot]:
        return self._output_slots

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Data]:
        pass
