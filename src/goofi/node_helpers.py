import functools
import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from multiprocessing import Process
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple, Type

from goofi import nodes as goofi_nodes
from goofi.connection import Connection
from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.params import NodeParams

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def list_nodes() -> List[Type]:
    """
    Gather a list of all available nodes in the goofi.nodes module.

    ### Returns
    List[Type[None]]
        A list containing the classes of all available nodes.
    """

    def _list_nodes_recursive(nodes=None, parent_module=goofi_nodes):
        from goofi.node import Node

        if nodes is None:
            # first call, initialize the list
            nodes = []

        # iterate over all modules in the parent module
        for info in pkgutil.walk_packages(parent_module.__path__):
            module = importlib.import_module(f"{parent_module.__name__}.{info.name}")

            if info.ispkg:
                # recursively list nodes in submodules
                _list_nodes_recursive(nodes, module)
                continue

            # current module is a node, add it to the list
            members = inspect.getmembers(module, inspect.isclass)
            nodes.extend([cls for _, cls in members if issubclass(cls, Node) and cls is not Node])
        return nodes

    return _list_nodes_recursive()


# call list_nodes once to initialize the cache
list_nodes()


@dataclass
class InputSlot:
    """
    Nodes have a number of input slots that can be connected to output slots of other nodes. Each input slot
    has a specific data type (`dtype`) and a `trigger_process` flag that indicates if the node's process
    function should be triggered when data is received on this input slot. The input slot also internally
    handles the data (`data`) received on the input slot.

    ### Parameters
    `dtype` : DataType
        The data type of the input slot.
    `trigger_process` : Optional[bool]
        Optional flag to indicate if the node's process function should be triggered when data is received on
        this input slot. Defaults to True.
    """

    dtype: DataType
    trigger_process: bool = True
    data: Optional[Data] = None


@dataclass
class OutputSlot:
    """
    Nodes have a number of output slots that can be connected to input slots of other nodes. Each output slot
    has a specific data type (`dtype`) and internally handles a list of connections (`connections`).

    ### Parameters
    `dtype` : DataType
        The data type of the output slot.
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

    ### Parameters
    `connection` : Connection
        The connection object to the node.
    `input_slots` : Dict[str, DataType]
        A dictionary of input slots and their data types.
    `output_slots` : Dict[str, DataType]
        A dictionary of output slots and their data types.
    `params` : NodeParams
        The parameters of the node.
    `process` : Optional[Process]
        If the node is running in a separate process, this should be the process object. Defaults to None.
    """

    connection: Connection
    input_slots: Dict[str, DataType]
    output_slots: Dict[str, DataType]
    params: NodeParams

    process: Optional[Process] = None
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

    def update_param(self, group, param_name, param_value):
        """
        Updates the value of a parameter in the local node reference, and sends a message to the node to
        update the parameter value.

        ### Parameters
        `group` : str
            The name of the parameter group.
        `param_name` : str
            The name of the parameter.
        `param_value` : Any
            The new value of the parameter.
        """
        if group not in self.params:
            raise ValueError(f"Parameter group '{group}' doesn't exist.")
        if param_name not in self.params[group]:
            raise ValueError(f"Parameter '{param_name}' doesn't exist in group '{group}'.")
        self.params[group][param_name].value = param_value
        self.connection.send(
            Message(
                MessageType.PARAMETER_UPDATE,
                {
                    "group": group,
                    "param_name": param_name,
                    "param_value": param_value,
                },
            )
        )

    def terminate(self) -> None:
        """Terminates the node by closing the connection to it."""
        self.connection.try_send(Message(MessageType.TERMINATE, {}))
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
                try:
                    self.callbacks[msg.type](self, msg)
                except Exception as e:
                    logger.error(f"Message callback for {msg.type} failed: {e}")
                continue

            # built-in message handling
            if msg.type == MessageType.PING:
                self.connection.send(Message(MessageType.PONG, {}))
            elif msg.type == MessageType.TERMINATE:
                self._alive = False
                self.connection.close()

    def __post_init__(self):
        if self.connection is None:
            raise ValueError("Expected Connection, got None.")

        self._alive = True
        self._messaging_thread = Thread(target=self._messaging_loop, daemon=True)
        self._messaging_thread.start()

        # register the node reference as an output pipe for each output slot
        for name in self.output_slots.keys():
            self.connection.send(
                Message(
                    MessageType.ADD_OUTPUT_PIPE,
                    {"slot_name_out": name, "slot_name_in": name, "node_connection": None},
                )
            )
