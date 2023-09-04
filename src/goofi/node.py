import logging
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from threading import Event, Thread
from typing import Any, Dict, List, Tuple

from goofi.message import Message, MessageType
from goofi.pipes import Data, InputPipe, OutputPipe

logger = logging.getLogger(__name__)


class NodeRef:
    def __init__(self, process: Process, node_pipe: Pipe) -> None:
        self.node_process = process
        self.node_pipe = node_pipe

    @property
    def pid(self) -> int:
        return self.node_process.pid

    def send(self, msg: Message) -> None:
        self.node_pipe.send(msg)


class BaseNode(ABC):
    def __init__(self, manager_pipe: Pipe) -> None:
        self.manager_pipe = manager_pipe

        self._input_pipes = None
        self._output_pipes = None

        self.output_connections = None

        # initialize node update flag
        self.update_flag = Event()

        # initialize message handling thread
        self.message_thread = Thread(target=self.message_loop, daemon=True)
        self.message_thread.start()

        # initialize node update thread
        self.update_thread = Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

    def message_loop(self):
        """
        This method runs in a separate thread and handles incoming messages from the manager, or other nodes.
        """
        while True:
            try:
                # wait for a new message
                msg = self.manager_pipe.recv()
                # make sure the message is a Message object
                if not isinstance(msg, Message):
                    raise TypeError(f"Expected Message, got {type(msg)}")
                # parse the message
                self.handle_message(msg)
            except Exception as e:
                # error while handling message
                logger.error(e)
                # TODO: send error message to manager

    def handle_message(self, msg: Message) -> None:
        """
        This method is called from the message loop when a message is received.
        """
        # make sure the message is a Message object
        if msg.payload is None:
            raise ValueError("Received empty data message")

        # handle message
        if msg.type == MessageType.ADD_PIPE:
            # make sure the message has the correct fields
            if not "target_name" in msg.payload:
                raise ValueError("Missing target name")
            if not "node_ref" in msg.payload:
                raise ValueError("Missing node reference")
            if not isinstance(msg.payload["node_ref"], NodeRef):
                raise TypeError(f"Expected NodeRef, got {type(msg.payload['node_ref'])}")

            # add the node reference to the output connections
            self.output_connections[msg.payload["target_name"]].append(msg.payload["node_ref"])
        elif msg.type == MessageType.REMOVE_PIPE:
            # make sure the message has the correct fields
            if not "target_name" in msg.payload:
                raise ValueError("Missing target name")
            if not "node_ref" in msg.payload:
                raise ValueError("Missing node reference")
            if not isinstance(msg.payload["node_ref"], NodeRef):
                raise TypeError(f"Expected NodeRef, got {type(msg.payload['node_ref'])}")

            # remove the node reference from the output connections
            self.output_connections[msg.payload["target_name"]].remove(msg.payload["node_ref"])
        elif msg.type == MessageType.DATA:
            # make sure the message has the correct fields
            if not "target_name" in msg.payload:
                raise ValueError("Missing target name")
            if not "data" in msg.payload:
                raise ValueError("Missing data")
            if not isinstance(msg.payload["data"], Data):
                raise TypeError(f"Expected Data, got {type(msg.payload['data'])}")

            # find the input pipe and update its data reference
            input_pipe = self.input_pipes[msg.payload["target_name"]]
            input_pipe.data = msg.payload["data"].data
            input_pipe.metadata = msg.payload["data"].metadata
            # trigger an update if necessary
            if input_pipe.trigger_update:
                self.update_flag.set()
        else:
            raise ValueError(f"Unknown message type {msg.type}")

    @property
    def input_pipes(self) -> Dict[str, InputPipe]:
        return self._input_pipes

    @input_pipes.setter
    def input_pipes(self, value: Dict[str, InputPipe]) -> None:
        if self._input_pipes is not None:
            # TODO: handle updated inputs
            raise NotImplementedError("TODO: handle updated inputs")
        self._input_pipes = value

    @property
    def output_pipes(self) -> List[OutputPipe]:
        return self._output_pipes

    @output_pipes.setter
    def output_pipes(self, value: List[OutputPipe]) -> None:
        if self._output_pipes is not None:
            # TODO: handle updated outputs
            raise NotImplementedError("TODO: handle updated outputs")
        self._output_pipes = value
        self.output_connections = {output.name: [] for output in self._output_pipes}

    def update_loop(self):
        """
        This method runs in a separate thread and calls the node's update method. It waits for an update signal from
        the message thread before calling the update method. It also sends outgoing messages to other nodes.
        """
        while True:
            # wait until the update flag is set
            self.update_flag.wait()
            self.update_flag.clear()

            try:
                self.node.wrapped_update()
            except Exception as e:
                logger.error(e)

            # send outgoing messages
            for output in self.output_pipes:
                for node_ref in output.connected_nodes:
                    node_ref.send(Message(MessageType.DATA, output.data))

    def wrapped_setup(self):
        self.input_pipes, self.output_pipes = self.setup()

    @abstractmethod
    def setup(self) -> Tuple[List[InputPipe], List[OutputPipe]]:
        pass

    def wrapped_update(self) -> None:
        # generate inputs from input pipes
        inputs = {name: Data(pipe.dtype, pipe.data, pipe.metadata) for name, pipe in self.input_pipes.items()}
        # call update method
        outputs = self.update(**inputs)

        # make sure all expected outputs are present and have the correct type
        for out_pipe in self.output_pipes:
            if out_pipe.name not in outputs:
                raise ValueError(f"Missing output {out_pipe.name}")
            out_pipe.dtype == outputs[out_pipe.name].dtype

        # make sure there aren't any extra outputs
        if len(outputs) != len(self.output_pipes):
            expected = [p.name for p in self.output_pipes]
            raise ValueError(f"Received unexpected outputs: {set(outputs.keys()) - set(expected)}")

        # send outputs to output connections
        for name, conn in self.output_connections.items():
            conn.send(Message(MessageType.DATA, {"target_name": name, "data": outputs[name]}))

    @abstractmethod
    def update(self, **inputs: Dict[str, Data]) -> Dict[str, Data]:
        pass
