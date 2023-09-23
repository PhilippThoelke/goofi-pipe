import pytest

from goofi.data import DataType
from goofi.node_helpers import InputSlot, OutputSlot


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_input_slot(dtype):
    slot = InputSlot(dtype)
    assert slot.trigger_process is True, "InputSlot trigger_process should be False."
    assert slot.data is None, "InputSlot data should be None."


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_output_slot(dtype):
    slot = OutputSlot(dtype)
    assert isinstance(slot.connections, list), "OutputSlot connections should be a list."
    assert len(slot.connections) == 0, "OutputSlot connections should be empty."
