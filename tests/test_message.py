from multiprocessing import Pipe

import pytest

from goofi.data import Data, DataType
from goofi.message import Message, MessageType

EXAMPLE_CONTENT = {
    MessageType.ADD_OUTPUT_PIPE: {"slot_name_out": "out", "slot_name_in": "in", "node_connection": Pipe()[0]},
    MessageType.REMOVE_OUTPUT_PIPE: {"slot_name": "test"},
    MessageType.DATA: {"slot_name": "test", "data": Data(DataType.STRING, "", {})},
    MessageType.PING: {},
    MessageType.PONG: {},
    MessageType.TERMINATE: {},
    MessageType.NODE_PARAMS: {"params": dict(), "input_slots": dict(), "output_slots": dict()},
    MessageType.NODE_PARAMS_REQUEST: {},
}


@pytest.mark.parametrize("type", MessageType.__members__.values())
def test_create_message(type):
    if type not in EXAMPLE_CONTENT:
        # EXAMPLE_CONTENT is missing a test for this type
        raise NotImplementedError(f"Missing test for {type}.")

    # all type checks should pass
    Message(type, EXAMPLE_CONTENT[type])


@pytest.mark.parametrize("type", MessageType.__members__.values())
def test_message_content(type):
    if type not in EXAMPLE_CONTENT:
        # EXAMPLE_CONTENT is missing a test for this type
        raise NotImplementedError(f"Missing test for {type}.")

    for key in EXAMPLE_CONTENT[type]:
        # remove the key from the content
        content = EXAMPLE_CONTENT[type].copy()
        del content[key]

        # should raise ValueError
        with pytest.raises(ValueError):
            Message(type, content)


@pytest.mark.parametrize("type", MessageType.__members__.values())
def test_message_errors(type):
    if type not in EXAMPLE_CONTENT:
        # EXAMPLE_CONTENT is missing a test for this type
        raise NotImplementedError(f"Missing test for {type}.")

    # type is None
    with pytest.raises(ValueError):
        Message(None, EXAMPLE_CONTENT[type])
    # content is None
    with pytest.raises(ValueError):
        Message(type, None)
