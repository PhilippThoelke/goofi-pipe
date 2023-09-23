import pytest
import yaml

from goofi.connection import MultiprocessingConnection
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


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_serialize_input_slot(dtype):
    slot = InputSlot(dtype)

    # serialize the slot
    serialized = slot.serialize()
    serialized_str = yaml.dump(serialized)

    print(serialized_str)

    # reconstruct the slot from the serialized data
    reconstructed = yaml.load(serialized_str, Loader=yaml.FullLoader)
    slot2 = InputSlot(**reconstructed)

    # check if the reconstructed slot is equal to the original slot
    assert slot == slot2, "Reconstructed InputSlot should be equal to the original InputSlot."


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_serialize_output_slot(dtype):
    slot = OutputSlot(dtype)

    # serialize the slot
    serialized = slot.serialize()
    serialized_str = yaml.dump(serialized)

    # reconstruct the slot from the serialized data
    reconstructed = yaml.load(serialized_str, Loader=yaml.FullLoader)
    slot2 = OutputSlot(**reconstructed)

    # check if the reconstructed slot is equal to the original slot
    assert slot == slot2, "Reconstructed OutputSlot should be equal to the original OutputSlot."


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_serialize_output_slot_conn(dtype):
    slot = OutputSlot(dtype)
    conn = MultiprocessingConnection.create()[0]
    slot.connections.append(conn)

    # serialize the slot
    serialized = slot.serialize()

    # we want the connections to remain connection objects, as these are needed to determine the
    # linked nodes when saving a pipeline. The connection objects are serialized outside of the
    # node class.
    assert serialized["connections"] == [conn], "Serialized OutputSlot should contain the connection object."
