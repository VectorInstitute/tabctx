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


def test_a_planted_model_is_neither_listed_nor_loaded(tmp_path):
    """A repository could ship .tabctx-mcp/x.pkl: unpickling it would run
    its code, so the store only loads pickles signed with the user's key."""
    import json
    import pickle

    store = ModelStore(tmp_path)
    store.save(info("mine"), 1)
    (tmp_path / "x.pkl").write_bytes(pickle.dumps({"planted": True}))
    (tmp_path / "x.json").write_text(
        json.dumps({**info("x").as_dict(), "signature": "0" * 64})
    )
    assert [i.name for i in store.list()] == ["mine"]
    with pytest.raises(ValueError, match="not saved by your tabctx-mcp"):
        store.load("x")


def test_a_changed_pickle_is_refused(tmp_path):
    store = ModelStore(tmp_path)
    store.save(info(), {"a": 1})
    path = tmp_path / "m.pkl"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="signature"):
        store.load("m")


def test_another_users_key_cannot_load_it(tmp_path):
    ModelStore(tmp_path / "s", key_file=tmp_path / "k1").save(info(), 1)
    other = ModelStore(tmp_path / "s", key_file=tmp_path / "k2")
    assert other.list() == []
    with pytest.raises(ValueError, match="signature"):
        other.load("m")


def test_the_key_is_private_and_reused(tmp_path):
    key = tmp_path / "cfg" / "signing.key"
    store = ModelStore(tmp_path / "s", key_file=key)
    store.save(info(), 1)
    assert oct(key.stat().st_mode & 0o777) == "0o600"
    assert ModelStore(tmp_path / "s", key_file=key).load("m")[1] == 1


def test_records_from_a_newer_version_still_load(tmp_path):
    import json

    store = ModelStore(tmp_path)
    store.save(info(), 1)
    record = json.loads((tmp_path / "m.json").read_text())
    record["added_later"] = True
    (tmp_path / "m.json").write_text(json.dumps(record))
    assert store.load("m")[0] == info()
