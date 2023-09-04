from dataclasses import dataclass
from typing import Any

from goofi.constants import MessageType


@dataclass
class Message:
    type: MessageType
    payload: Any
