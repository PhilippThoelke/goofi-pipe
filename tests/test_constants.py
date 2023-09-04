import numpy as np
import pytest

from goofi.constants import DataType


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_check_float(dim):
    dtype = getattr(DataType, f"FLOAT_{dim}D")
    for d in [1, 2, 3]:
        if d == dim:
            # make sure the correct number of dimensions passes
            value = np.zeros([1] * d)
            dtype.check_type(value)

            # a list, even if it's the right number of dimensions, should fail
            with pytest.raises(ValueError):
                dtype.check_type(value.tolist())
        else:
            # make sure the wrong number of dimensions fails
            value = np.zeros([1] * d)
            with pytest.raises(ValueError):
                dtype.check_type(value)

    # make sure a string fails
    with pytest.raises(ValueError):
        dtype.check_type("test")

    # None should fail
    with pytest.raises(ValueError):
        dtype.check_type(None)


def test_check_string():
    dtype = DataType.STRING
    # make sure a string passes
    dtype.check_type("test")

    # make sure a numpy array fails
    with pytest.raises(ValueError):
        dtype.check_type(np.zeros([1, 1]))

    # None should fail
    with pytest.raises(ValueError):
        dtype.check_type(None)
