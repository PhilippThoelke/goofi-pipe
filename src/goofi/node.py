import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import Process
from threading import Event, Thread
from typing import Any, Callable, Dict, Tuple, Union

from goofi.connection import Connection, MultiprocessingConnection
from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.node_helpers import InputSlot, NodeRef, OutputSlot
from goofi.params import NodeParams

logger = logging.getLogger(__name__)


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
    `input_slots` : Dict[str, InputSlot]
        A dict containing the input slots of the node. The keys are the names of the input slots, and the values
        are the input slots themselves.
    `output_slots` : Dict[str, OutputSlot]
        A dict containing the output slots of the node. The keys are the names of the output slots, and the values
        are the output slots themselves.
    `params` : NodeParams
        An instance of the NodeParams class containing the parameters of the node.
    """

    def __init__(
        self,
        connection: Connection,
        input_slots: Dict[str, InputSlot],
        output_slots: Dict[str, OutputSlot],
        params: NodeParams,
    ) -> None:
        # initialize the base class
        self._alive = True

        # store the arguments and validate them
        self.connection = connection
        self._input_slots = input_slots
        self._output_slots = output_slots
        self._params = params

        self._validate_attrs()

        # initialize node flags
        self.process_flag = Event()
        if self.params.common.autotrigger.value:
            self.process_flag.set()

        # initialize message handling thread
        self.messaging_thread = Thread(target=self._messaging_loop)
        self.messaging_thread.start()

        # initialize data processing thread
        self.processing_thread = Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    @require_init
    def _validate_attrs(self):
        """
        Check that all attributes are present and of the correct type.
        """
        # check connection type
        if not isinstance(self.connection, Connection):
            raise TypeError(f"Expected Connection, got {type(self.connection)}")
        # check input slots type
        for name, slot in self._input_slots.items():
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(f"Expected input slot name '{name}' to be a non-empty string.")
            if not isinstance(slot, InputSlot):
                raise TypeError(f"Expected InputSlot for input slot '{name}', got {type(slot)}")
        # check output slots type
        for name, slot in self._output_slots.items():
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(f"Expected output slot name '{name}' to be a non-empty string.")
            if not isinstance(slot, OutputSlot):
                raise TypeError(f"Expected OutputSlot for output slot '{name}', got {type(slot)}")
        # check params type
        if not isinstance(self._params, NodeParams):
            raise TypeError(f"Expected NodeParams, got {type(self._params)}")

    def _messaging_loop(self):
        """
        This method runs in a separate thread and handles incoming messages from the manager, or other nodes.
        """
        # run the node's setup method
        try:
            self.setup()
        except Exception as e:
            self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": str(e)}))

        # run the messaging loop
        while self.alive:
            # receive the message
            try:
                msg = self.connection.recv()
            except (EOFError, ConnectionResetError, OSError):
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
                # respond to a ping message by sending a pong message
                self.connection.try_send(Message(MessageType.PONG, {}))
            elif msg.type == MessageType.TERMINATE:
                # terminate the node
                self._alive = False
                # clear data in connected downstream nodes
                for slot in self.output_slots.values():
                    for slot_name, conn in slot.connections:
                        conn.try_send(Message(MessageType.CLEAR_DATA, {"slot_name": slot_name}))
            elif msg.type == MessageType.ADD_OUTPUT_PIPE:
                # add a connection to the output slot
                slot = self.output_slots[msg.content["slot_name_out"]]
                conn = msg.content["node_connection"]

                if conn is None:
                    # if no connection is specified, connect to the node's own node reference
                    conn = self.connection
                slot.connections.append((msg.content["slot_name_in"], conn))
            elif msg.type == MessageType.REMOVE_OUTPUT_PIPE:
                # clear the data in the input slot
                msg.content["node_connection"].try_send(
                    Message(MessageType.CLEAR_DATA, {"slot_name": msg.content["slot_name_in"]})
                )
                # remove the connection
                slot = self.output_slots[msg.content["slot_name_out"]]
                try:
                    slot.connections.remove((msg.content["slot_name_in"], msg.content["node_connection"]))
                except ValueError:
                    # connection doesn't exist
                    # TODO: send error message to manager
                    pass
            elif msg.type == MessageType.DATA:
                # received data from another node
                if msg.content["slot_name"] not in self.input_slots:
                    raise ValueError(f"Received DATA message but input slot '{msg.content['slot_name']}' doesn't exist.")
                slot = self.input_slots[msg.content["slot_name"]]
                slot.data = msg.content["data"]
                if slot.trigger_process:
                    self.process_flag.set()
            elif msg.type == MessageType.CLEAR_DATA:
                # clear the data in the input slot (usually triggered by a REMOVE_OUTPUT_PIPE message)
                slot = self.input_slots[msg.content["slot_name"]]
                slot.data = None
            elif msg.type == MessageType.PARAMETER_UPDATE:
                # update a parameter
                group = msg.content["group"]
                param_name = msg.content["param_name"]
                param_value = msg.content["param_value"]
                if group not in self.params:
                    raise ValueError(f"Parameter group '{group}' doesn't exist.")
                if param_name not in self.params[group]:
                    raise ValueError(f"Parameter '{param_name}' doesn't exist in group '{group}'.")
                self.params[group][param_name].value = param_value

                # call the callback if it exists
                if hasattr(self, f"{group}_{param_name}_changed"):
                    try:
                        getattr(self, f"{group}_{param_name}_changed")(param_value)
                    except Exception as e:
                        # parameter callback raised an exception, send out an error message
                        self.connection.try_send(
                            Message(
                                MessageType.PROCESSING_ERROR,
                                {"error": f"Parameter callback for {group}.{param_name} failed: {e}"},
                            )
                        )
            else:
                # TODO: handle the incoming message
                raise NotImplementedError(f"Message type {msg.type} not implemented.")

        # run the node's terminate method
        try:
            self.terminate()
        except Exception as e:
            self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": str(e)}))

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
            if not self.params.common.autotrigger.value:
                self.process_flag.clear()

            # limit the update rate
            if self.params.common.max_frequency.value > 0:
                sleep_time = 1 / self.params.common.max_frequency.value - (time.time() - last_update)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_update = time.time()

            # gather input data
            input_data = {name: slot.data for name, slot in self.input_slots.items()}

            try:
                # process data
                output_data = self.process(**input_data)
            except Exception as e:
                self.connection.try_send(Message(MessageType.PROCESSING_ERROR, {"error": str(e)}))
                continue

            if not self.alive:
                # the node was terminated during processing
                break

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
                if not isinstance(data, tuple) or len(data) != 2:
                    raise ValueError(
                        f"Expected {self.__class__.__name__}.process() to return a tuple of data and metadata but got {data}."
                    )
                data = Data(self.output_slots[name].dtype, data[0], data[1])

                # send the data to all connected nodes
                for target_slot, conn in self.output_slots[name].connections:
                    msg = Message(MessageType.DATA, {"data": data, "slot_name": target_slot})
                    try:
                        conn.send(msg)
                    except ConnectionError:
                        # the target node is dead, remove the connection
                        self.output_slots[name].connections.remove((target_slot, conn))

    @staticmethod
    def _configure(cls) -> Tuple[Dict[str, InputSlot], Dict[str, OutputSlot], NodeParams]:
        """Retrieves the node's configuration of input slots, output slots, and parameters."""
        in_slots = cls.config_input_slots()
        out_slots = cls.config_output_slots()
        params = cls.config_params()

        return (
            {name: slot if isinstance(slot, InputSlot) else InputSlot(slot) for name, slot in in_slots.items()},
            {name: slot if isinstance(slot, OutputSlot) else OutputSlot(slot) for name, slot in out_slots.items()},
            NodeParams(params),
        )

    def common_autotrigger_changed(self, value):
        """If the new value of the parameter common.autotrigger is True, trigger the processing loop."""
        if value:
            self.process_flag.set()

    @classmethod
    def create(cls) -> NodeRef:
        """
        Create a new node instance in a separate process and return a reference to the node.

        ### Returns
        `NodeRef`
            A reference to the node.
        """
        # generate arguments for the node
        in_slots, out_slots, params = cls._configure(cls)
        conn1, conn2 = MultiprocessingConnection.create()
        # instantiate the node in a separate process
        proc = Process(target=cls, args=(conn2, in_slots, out_slots, params), daemon=True)
        proc.start()
        # create the node reference
        return NodeRef(
            conn1,
            {name: slot.dtype for name, slot in in_slots.items()},
            {name: slot.dtype for name, slot in out_slots.items()},
            params,
            process=proc,
        )

    @classmethod
    def create_local(cls) -> Tuple[NodeRef, "Node"]:
        """
        Create a new node instance in the current process and return a reference to the node,
        as well as the node itself.

        ### Returns
        `Tuple[NodeRef, Node]`
            A tuple containing the node reference and the node itself.
        """
        # generate arguments for the node
        in_slots, out_slots, params = cls._configure(cls)
        conn1, conn2 = MultiprocessingConnection.create()
        # instantiate the node in the current process
        node = cls(conn2, in_slots, out_slots, params)
        # create the node reference
        return (
            NodeRef(
                conn1,
                {name: slot.dtype for name, slot in in_slots.items()},
                {name: slot.dtype for name, slot in out_slots.items()},
                deepcopy(params),
            ),
            node,
        )

    @classmethod
    def category(cls) -> str:
        """
        Returns the category of the node, i.e. the name of the node's module.

        ### Returns
        `str`
            The category of the node.
        """
        return cls.__module__.split(".")[-2]

    @property
    @require_init
    def alive(self) -> bool:
        return self._alive

    @property
    @require_init
    def params(self) -> NodeParams:
        return self._params

    @property
    @require_init
    def input_slots(self) -> Dict[str, InputSlot]:
        return self._input_slots

    @property
    @require_init
    def output_slots(self) -> Dict[str, OutputSlot]:
        return self._output_slots

    @staticmethod
    def config_input_slots() -> Dict[str, Union[InputSlot, DataType]]:
        """
        This method is called when the node is instantiated. It should return a dict containing the input slots
        of the node. The keys are the names of the input slots, and the values are either InputSlot instances or
        DataType instances.

        ### Returns
        `Dict[str, Union[InputSlot, DataType]]`
            A dict containing the input slots of the node.
        """
        return {}

    @staticmethod
    def config_output_slots() -> Dict[str, Union[OutputSlot, DataType]]:
        """
        This method is called when the node is instantiated. It should return a dict containing the output slots
        of the node. The keys are the names of the output slots, and the values are either OutputSlot instances or
        DataType instances.

        ### Returns
        `Dict[str, Union[OutputSlot, DataType]]`
            A dict containing the output slots of the node.
        """
        return {}

    @staticmethod
    def config_params() -> Dict[str, Dict[str, Any]]:
        """
        This method is called when the node is instantiated. It should return a dict containing the parameters
        of the node. The keys are the names of the parameter groups, and the values are dicts containing the
        parameters of each group.

        ### Returns
        `Dict[str, Dict[str, Any]]`
            A dict containing the parameters of the node. The values may be any type that is supported by the
            `Param` class.
        """
        return {}

    @require_init
    def setup(self) -> None:
        """
        This method is called after the node is instantiated. It can be used to set up the node.
        """
        pass

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        This method is called when the node is triggered. It should process the input data and return a dict
        containing the output data.

        ### Parameters
        `**kwargs` : Any
            The input data.

        ### Returns
        `Dict[str, Tuple[Any, Dict[str, Any]]]`
            A dict containing the output data. The keys are the names of the output slots, and the values are
            tuples containing the data and metadata of the output data.
        """
        pass

    def terminate(self) -> None:
        """
        This method is called when the node is terminated. It can be used to clean up any resources used by the
        node.
        """
        pass
