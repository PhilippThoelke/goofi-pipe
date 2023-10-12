import pytest

from goofi.data import DTYPE_TO_TYPE, Data

from .utils import list_data_types


@pytest.mark.parametrize("dtype", list_data_types())
def test_create_data(dtype):
    # all dtype checks should pass
    Data(dtype, dtype.empty(), {})

    # data is None, should raise ValueError
    with pytest.raises(ValueError):
        Data(dtype, None, {})

    # metadata is None, should raise ValueError
    with pytest.raises(ValueError):
        Data(dtype, dtype.empty(), None)

    # dtype is None, should raise ValueError
    with pytest.raises(ValueError):
        Data(None, dtype.empty(), {})

    # make sure all other dtypes raise a ValueError
    for other_dtype in list_data_types():
        if other_dtype == dtype:
            continue
        with pytest.raises(ValueError):
            Data(dtype, other_dtype.empty(), {})


@pytest.mark.parametrize("dtype", list_data_types())
def test_dtype_map(dtype):
    assert dtype in DTYPE_TO_TYPE, f"Missing entry in DTYPE_TO_TYPE for dtype {dtype}."
