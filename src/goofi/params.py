from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Union


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
    toggle: bool = True

    @staticmethod
    def default() -> bool:
        return False


@dataclass
class FloatParam(Param):
    vmin: float = 0.0
    vmax: float = 1.0

    @staticmethod
    def default() -> float:
        return 0.0


@dataclass
class IntParam(Param):
    vmin: int = -1
    vmax: int = 3

    @staticmethod
    def default() -> int:
        return 0


@dataclass
class StringParam(Param):
    @staticmethod
    def default() -> str:
        return ""


DEFAULT_PARAMS = {
    "common": {
        "autotrigger": BoolParam(False),
        "max_frequency": FloatParam(30.0, 0.0, 60.0),
    },
}

TYPE_PARAM_MAP = {
    bool: BoolParam,
    float: FloatParam,
    int: IntParam,
    str: StringParam,
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
            if not isinstance(params, dict):
                raise TypeError(f"Expected dict, got {type(params)}.")
            for param_name, param in params.items():
                if not isinstance(param, Param):
                    if isinstance(param, dict):
                        # reconstruct serialized param object
                        param_type = TYPE_PARAM_MAP[type(param["value"])]
                        data[group][param_name] = param_type(**param)
                        continue

                    # convert to Param object
                    if type(param) not in TYPE_PARAM_MAP:
                        raise TypeError(
                            f"Invalid parameter type {type(param).__name__}. Must be one of "
                            f"{list(map(lambda x: x.__name__, TYPE_PARAM_MAP.keys()))}."
                        )
                    data[group][param_name] = TYPE_PARAM_MAP[type(param)](param)

        # convert to named tuples
        self._data = {}
        for group, params in data.items():
            # create the named tuple class for the current group
            NamedTupleClass = namedtuple(group.capitalize(), params.keys())

            # implement __contains__ for the named tuple class
            NamedTupleClass = type(
                NamedTupleClass.__name__,
                (NamedTupleClass,),
                {
                    "__contains__": lambda self, item: hasattr(self, item),
                    "__getitem__": lambda self, item: getattr(self, item),
                    "keys": lambda self: self._asdict().keys(),
                    "values": lambda self: self._asdict().values(),
                    "items": lambda self: self._asdict().items(),
                },
            )

            self._data[group] = NamedTupleClass(**params)

    def update(self, params: Dict[str, Dict[str, Any]]):
        """
        Update the parameters with new values.

        ### Parameters
        `params` : Dict[str, Dict[str, Any]]
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
        """
        for group, params in params.items():
            for name, param in params.items():
                if group not in self._data:
                    raise ValueError(f"Parameter group '{group}' does not exist.")
                if name not in self._data[group]._fields:
                    raise ValueError(f"Parameter '{name}' does not exist in group '{group}'.")
                if not isinstance(param, Param):
                    if isinstance(param, dict):
                        # reconstruct serialized param object
                        param_type = TYPE_PARAM_MAP[type(param["value"])]
                        self._data[group] = self._data[group]._replace(**{name: param_type(**param)})
                        continue

                    # convert to Param object
                    if type(param) not in TYPE_PARAM_MAP:
                        raise TypeError(
                            f"Invalid parameter type {type(param).__name__}. Must be one of "
                            f"{list(map(lambda x: x.__name__, TYPE_PARAM_MAP.keys()))}."
                        )
                    self._data[group] = self._data[group]._replace(**{name: TYPE_PARAM_MAP[type(param)](param)})
                else:
                    self._data[group] = self._data[group]._replace(**{name: param})

    def serialize(self) -> Dict[str, Dict[str, Any]]:
        """
        Serialize the parameters to a dictionary.

        ### Returns
        `Dict[str, Dict[str, Any]]`
            A dictionary of parameter groups, where each group is a dictionary of parameter names and values.
        """
        serialized_data = {}
        for group, params in self._data.items():
            serialized_params = {}
            for name, param in params.items():
                param = asdict(param)
                serialized_params[name] = param
            serialized_data[group] = serialized_params
        return serialized_data

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

    def __getitem__(self, group: Union[int, str]):
        if isinstance(group, int):
            return list(self._data.keys())[group]
        return self._data[group]

    def __len__(self) -> int:
        return len(self._data)
