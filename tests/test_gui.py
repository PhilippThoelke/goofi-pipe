import pytest

from goofi.gui.window import DTYPE_SHAPE_MAP

from .utils import list_data_types


@pytest.mark.parametrize("dtype", list_data_types())
def test_dtype_shape_map(dtype):
    assert dtype in DTYPE_SHAPE_MAP, f"{dtype} not in DTYPE_SHAPE_MAP."
