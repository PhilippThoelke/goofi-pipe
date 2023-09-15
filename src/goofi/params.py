from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Param(ABC):
    value: Any = None

    def __post_init__(self):
        if self.value is None:
            self.value = self.default()
        if not isinstance(self.value, type(self.default())):
            raise TypeError(f"Expected {type(self.default())}, got {type(self.value)}")

    @staticmethod
    @abstractmethod
    def default() -> Any:
        pass


@dataclass
class BoolParam(Param):
    @staticmethod
    def default() -> bool:
        return False


@dataclass
class FloatParam(Param):
    min: float = 0.0
    max: float = 1.0

    @staticmethod
    def default() -> float:
        return 0.0


DEFAULT_PARAMS = {
    "common": {
        "autotrigger": BoolParam(False),
        "max_frequency": FloatParam(30.0, 0.0, 60.0),
    },
}

TYPE_PARAM_MAP = {
    bool: BoolParam,
    float: FloatParam,
}


class NodeParams:
    """
    A class for storing node parameters, which store the configuration state of the node, and are exposed
    to the user through the GUI. The parameters are stored in named groups with each group containing a
    number of parameters. The parameters are stored as named tuples, and can be accessed as attributes.

    When initializing a `NodeParams` object, a set of default parameters is inserted if they are not provided.

    ### Parameters
    `data` : Dict[str, Dict[str, Any]]
        A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
    """

    def __init__(self, data: Dict[str, Dict[str, Any]]):
        # insert default parameters if they are not present
        for name, group in deepcopy(DEFAULT_PARAMS).items():
            if name not in data:
                # group is not present, insert it
                data[name] = group
            else:
                # group is present, insert missing parameters
                for param_name, param in group.items():
                    if param_name not in data[name]:
                        data[name][param_name] = param

        # convert values to Param objects
        for group, params in data.items():
            for param_name, param in params.items():
                if not isinstance(param, Param):
                    if type(param) not in TYPE_PARAM_MAP:
                        raise TypeError(f"Invalid parameter type {type(param)}. Must be one of {list(TYPE_PARAM_MAP.keys())}")
                    data[group][param_name] = TYPE_PARAM_MAP[type(param)](param)

        # convert to named tuples
        self._data = {}
        for group, params in data.items():
            # create the named tuple class for the current group
            NamedTupleClass = namedtuple(group.capitalize(), params.keys())

            # implement __contains__ for the named tuple class
            class CustomNamedTuple(NamedTupleClass):
                def __contains__(self, item):
                    return hasattr(self, item)

            self._data[group] = CustomNamedTuple(**params)

    def __getattr__(self, group: str):
        # don't allow access to the _data attribute
        if group == "_data":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{group}'")

        # return the group if it exists
        if group in self._data:
            return self._data[group]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{group}'")

    def __contains__(self, group: str) -> bool:
        return group in self._data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeParams):
            return False
        return self._data == other._data

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr,self._data.values()))})"
