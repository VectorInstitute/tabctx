"""tabctx-mcp's model store (mcp_tools/store.py)."""

import pytest

from tabctx.mcp_tools.store import HOME_ENV, ModelInfo, ModelStore


def info(name="m"):
    return ModelInfo(
        name=name,
        model="tabpfn-3.5",
        task="classification",
        target="y",
        features=["a", "b"],
        train_file="t.csv",
        train_rows=10,
        classes=["no", "yes"],
    )


def test_save_load_and_list(tmp_path):
    store = ModelStore(tmp_path / "s")
    assert store.list() == []
    store.save(info("b"), {"fitted": 1})
    store.save(info("a"), {"fitted": 2})
    loaded_info, estimator = store.load("a")
    assert loaded_info == info("a") and estimator == {"fitted": 2}
    assert [i.name for i in store.list()] == ["a", "b"]
    assert not list((tmp_path / "s").glob("*.tmp"))


def test_saving_again_replaces(tmp_path):
    store = ModelStore(tmp_path)
    store.save(info(), 1)
    store.save(info(), 2)
    assert store.load("m")[1] == 2 and len(store.list()) == 1


def test_unknown_model_names_the_known_ones(tmp_path):
    store = ModelStore(tmp_path)
    store.save(info("churn"), 1)
    with pytest.raises(ValueError, match=r"no model named 'x'.*\['churn'\]"):
        store.load("x")


@pytest.mark.parametrize("name", ["", "../etc", "a/b", ".hidden", "x" * 101])
def test_bad_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError, match="model name"):
        ModelStore(tmp_path).save(info(name), 1)


def test_root_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "env"))
    assert ModelStore().root == tmp_path / "env"
    monkeypatch.delenv(HOME_ENV)
    assert str(ModelStore().root) == ".tabctx-mcp"
