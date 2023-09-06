from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Dict

from goofi.data import Data


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

    def __post_init__(self):
        """
        Check that the message content is valid. The content field must be a dict, and it must contain the correct
        fields for the message type.
        """
        # general checks
        if self.type is None:
            raise ValueError("Expected message type, got None")
        if self.origin_id is None:
            raise ValueError("Expected origin id, got None")
        if not isinstance(self.content, dict):
            raise ValueError(f"Expected dict, got {type(self.content)}")
        
        # currently, all messages must contain a slot_name field
        if "slot_name" not in self.content or not isinstance(self.content["slot_name"], str):
            raise ValueError("Message content must contain slot_name with type str")

        # check the content for the specific message type
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
