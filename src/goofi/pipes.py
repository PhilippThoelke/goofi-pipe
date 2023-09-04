from dataclasses import dataclass
from typing import Any, Dict

from goofi.constants import DataType


class Data:
    def __init__(self, dtype: DataType, data: Any, metadata: Dict[str, Any] = None) -> None:
        self.dtype = dtype
        self.dtype.check_type(data)
        self.data = data
        self.metadata = metadata


@dataclass
class InputPipe:
    name: str
    dtype: DataType
    trigger_update: bool = True
    data: Any = None
    metadata: Dict[str, Any] = None


@dataclass
class OutputPipe:
    name: str
    dtype: DataType
