import pytest

from goofi import params
from goofi.params import DEFAULT_PARAMS, NodeParams

from .utils import list_param_types


def test_default_members():
    assert "common" in DEFAULT_PARAMS, "Missing default group 'common'."
    # autotrigger
    assert "autotrigger" in DEFAULT_PARAMS["common"], "Missing default param 'autotrigger' in group 'common'."
    assert isinstance(
        DEFAULT_PARAMS["common"]["autotrigger"], params.BoolParam
    ), "Wrong type for default param 'autotrigger' in group 'common'"
    # max_frequency
    assert "max_frequency" in DEFAULT_PARAMS["common"], "Missing default param 'max_frequency' in group 'common'."
    assert isinstance(
        DEFAULT_PARAMS["common"]["max_frequency"], params.FloatParam
    ), "Wrong type for default param 'max_frequency' in group 'common'."


@pytest.mark.parametrize("param_type", list_param_types())
def test_param_types(param_type):
    p = param_type()
    assert p.value == param_type.default(), f"Wrong default value for param type {param_type.__name__}."

    for type2 in list_param_types():
        if type2 is param_type:
            continue

        if isinstance(type2.default(), type(param_type.default())):
            # some types may be instances of other types, e.g. bool is an instance of int
            continue

        with pytest.raises(TypeError):
            param_type(type2.default())


def test_node_params():
    params = NodeParams(DEFAULT_PARAMS)

    try:
        params.common
    except AttributeError:
        pytest.fail("NodeParams should support attribute access (e.g. params.common).")
    try:
        params.common.autotrigger
    except AttributeError:
        pytest.fail("Parameter groups should support attribute access (e.g. params.common.autotrigger).")

    with pytest.raises(AttributeError):
        params.nonexistent

    for name, group in DEFAULT_PARAMS.items():
        assert getattr(params, name)._asdict() == group, f"Wrong parameter group {name} in NodeParams."


def test_node_params_repr():
    repr(NodeParams(DEFAULT_PARAMS))


@pytest.mark.parametrize("param_type", list_param_types())
def test_type_param_map(param_type):
    assert any(
        param_type == t for t in params.TYPE_PARAM_MAP.values()
    ), f"Param type '{param_type.__name__}' is not in TYPE_PARAM_MAP."


@pytest.mark.parametrize("type_cls", params.TYPE_PARAM_MAP.keys())
def test_raw_value(type_cls):
    data = {"test": {"param": type_cls()}}
    NodeParams(data)


def test_node_params_iter():
    params = DEFAULT_PARAMS.copy()
    params["test"] = {"param": True}

    p = NodeParams(params)
    assert len(p) == 2, "NodeParams should have 2 groups."

    it = iter(p)
    assert next(it) == "common", "First group should be 'common'."
    assert next(it) == "test", "Second group should be 'test'."
    with pytest.raises(StopIteration):
        next(it)

    expected = ["common", "test"]
    groups = list(p)
    assert groups == expected, "NodeParams should have groups 'common' and 'test'."

    for i, group in enumerate(p):
        assert group == expected[i], f"NodeParams group {i} should be '{expected[i]}'."


def test_node_params_contains():
    params = DEFAULT_PARAMS.copy()
    params["test"] = {"param": True}

    p = NodeParams(params)
    assert "common" in p, "NodeParams should contain group 'common'."
    assert "test" in p, "NodeParams should contain group 'test'."
    assert "nonexistent" not in p, "NodeParams should not contain group 'nonexistent'."


def test_node_params_getitem():
    p = NodeParams(DEFAULT_PARAMS)
    assert p[0] == "common", "NodeParams group 0 should be 'common'."
    assert p["common"] == p.common, "NodeParams group 'common' should be accessible by name."


def test_param_group_getitem():
    p = NodeParams(DEFAULT_PARAMS)
    assert p.common["autotrigger"].value is False, "NodeParams group 'common' should have the correct value for 'autotrigger'."


def test_param_group_keys():
    p = NodeParams(DEFAULT_PARAMS)
    assert p.common.keys() == DEFAULT_PARAMS["common"].keys(), "NodeParams group 'common' should have the correct keys."


def test_param_group_values():
    p = NodeParams(DEFAULT_PARAMS)
    assert isinstance(
        p.common.values(), type(DEFAULT_PARAMS["common"].values())
    ), "NodeParams group 'common' should have a dict_values object."
    assert list(p.common.values()) == list(
        DEFAULT_PARAMS["common"].values()
    ), "NodeParams group 'common' should have the correct values."


def test_param_group_items():
    p = NodeParams(DEFAULT_PARAMS)
    assert p.common.items() == DEFAULT_PARAMS["common"].items(), "NodeParams group 'common' should have the correct items."
