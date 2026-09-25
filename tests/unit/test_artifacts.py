import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from coveo_recsys.artifacts import ArtifactError, LocalArtifactStore, sha256_bytes, sha256_file

META = dict(code_sha256="a" * 64, run_id="test", scope="synthetic", config_id="base")


def test_publish_read_reuse(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    desc = store.publish_bytes("data/item", b"hello", **META)
    assert desc.sha256 == sha256_file(tmp_path / "data/item")
    assert store.read_bytes("data/item") == store.read_by_sha256(desc.sha256) == b"hello"
    assert store.publish_bytes("data/item", b"hello", reuse=True, **META) == desc
    with pytest.raises(FileExistsError):
        store.publish_bytes("data/item", b"hello", **META)
    for kwargs in (
        {"data": b"changed"},
        {"inputs": {"source": "b" * 64}},
        {"configuration": {"x": 1}},
        {"code_sha256": "c" * 64},
    ):
        options = {**META, **kwargs}
        data = options.pop("data", b"hello")
        with pytest.raises(ArtifactError):
            store.publish_bytes("data/item", data, reuse=True, **options)


@pytest.mark.parametrize(
    "path", ["../x", "/x", "C:/x", "a\\b", "a/../b", "a//b", "a.", "NUL", "a:stream", ".objects/x"]
)
def test_reject_path(tmp_path: Path, path: str) -> None:
    with pytest.raises(ArtifactError):
        LocalArtifactStore(tmp_path).publish_bytes(path, b"x", **META)


def test_corruption_and_incomplete_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    store.publish_bytes("x", b"content", **META)
    (tmp_path / "x").write_bytes(b"changed")
    with pytest.raises(ArtifactError):
        store.read_bytes("x")
    original = store._atomic_write

    def fail_descriptor(path: str, data: bytes) -> None:
        if path.endswith("artifact.json"):
            raise OSError("simulated interrupted publication")
        original(path, data)

    monkeypatch.setattr(store, "_atomic_write", fail_descriptor)
    with pytest.raises(OSError):
        store.publish_bytes("interrupted", b"bytes", **META)
    with pytest.raises(ArtifactError):
        store.read_bytes("interrupted")
    assert not list(tmp_path.rglob(".pending-*"))


def test_tables_and_matrices(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    table = pa.table({"id": ["z", "a"], "value": [1.0, None]})
    store.publish_table("datasets/B/prefixes/part.parquet", table, data_role="prefix", **META)
    assert store.read_table("datasets/B/prefixes/part.parquet").equals(table)
    with pytest.raises(ArtifactError):
        store.publish_table(
            "datasets/B/prefixes/targets.parquet", table, data_role="targets", **META
        )
    matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
    store.publish_matrix("matrix.npy", matrix, row_ids=["z", "a"], **META)
    restored, ids = store.read_matrix("matrix.npy")
    assert restored.dtype == matrix.dtype and np.array_equal(restored, matrix)
    assert ids == ("z", "a")
    with pytest.raises(ArtifactError):
        store.publish_matrix("bad.npy", matrix, row_ids=["z", "z"], **META)


def test_descriptor_and_json(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    desc = store.publish_json("manifest.json", {"значение": 2}, **META)
    assert json.loads(store.read_bytes(desc.path)) == {"значение": 2}
    assert desc.sha256 == sha256_bytes(store.read_bytes(desc.path))
    with pytest.raises(ValueError):
        store.publish_json("bad.json", {"value": float("nan")}, **META)


def test_explicit_dependency_verification(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.publish_bytes("x", b"data", inputs={"source": "b" * 64}, **META)
    assert (
        store.verify_compatible(
            "x",
            expected_inputs={"source": "b" * 64},
            expected_code_sha256="a" * 64,
            expected_schema_version="1",
            expected_configuration={},
        ).path
        == "x"
    )
    with pytest.raises(ArtifactError):
        store.verify_compatible(
            "x",
            expected_inputs={"source": "c" * 64},
            expected_code_sha256="a" * 64,
            expected_schema_version="1",
            expected_configuration={},
        )


def test_matrix_metadata_cannot_hide_invalid_ids(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.publish_matrix("matrix", np.ones((2, 1)), row_ids=["a", "b"], **META)
    path = tmp_path / "matrix.artifact.json"
    data = json.loads(path.read_bytes())
    data["metadata"]["row_ids"] = ["a", "a"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactError):
        store.read_matrix("matrix")


def test_local_dependency_graph_rejects_cycles_forward_refs_and_changed_metadata(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    parent = store.publish_bytes("parent", b"input", **META)
    child = store.publish_bytes(
        "child", b"output", artifact_dependencies={"parent": parent.descriptor_sha256}, **META
    )
    assert child.artifact_dependencies == {"parent": parent.descriptor_sha256}
    with pytest.raises(ArtifactError):
        store.publish_bytes("cycle", b"x", artifact_dependencies={"cycle": "a" * 64}, **META)
    with pytest.raises(ArtifactError):
        store.publish_bytes(
            "forward", b"x", artifact_dependencies={"unpublished": "a" * 64}, **META
        )
    with pytest.raises(ArtifactError):
        store.publish_bytes("wrong", b"x", artifact_dependencies={"parent": "a" * 64}, **META)
    # Reusing a content hash does not create a dependency cycle: copying raw input
    # bytes is legitimate, whereas local artifact identity includes its descriptor.
    store.publish_bytes("copy", b"input", inputs={"external.csv": parent.sha256}, **META)


@pytest.mark.parametrize(
    "payload", [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":1,"x":2}']
)
def test_strict_json_rejects_ambiguous_or_nonfinite_input(payload: bytes) -> None:
    from coveo_recsys.artifacts import strict_json_loads

    with pytest.raises(ValueError):
        strict_json_loads(payload)


def _verification_scope(store):
    # Старый код должен падать по нарушению правила, а не отсутствию нового API.
    return getattr(store, "verification_scope", nullcontext)()


def _shared_graph(store, count):
    leaves = [
        store.publish_bytes(
            f"canonical/{index}.bin", index.to_bytes(8, "big") + b"x" * (65_536 - 8), **META
        )
        for index in range(count)
    ]
    index = store.publish_json(
        "canonical/index.json",
        [leaf.path for leaf in leaves],
        artifact_dependencies={leaf.path: leaf.descriptor_sha256 for leaf in leaves},
        **META,
    )
    sessions = store.publish_json(
        "preparation/sessions.json",
        {"sessions": count},
        artifact_dependencies={index.path: index.descriptor_sha256},
        **META,
    )
    selection = store.publish_json(
        "preparation/selection.json",
        {"selected": count},
        artifact_dependencies={sessions.path: sessions.descriptor_sha256},
        **META,
    )
    return leaves, index, selection


@pytest.mark.parametrize("count", [4, 8, 16])
def test_verification_scope_shared_graph_has_linear_physical_reads(tmp_path, monkeypatch, count):
    store = LocalArtifactStore(tmp_path)
    leaves, index, selection = _shared_graph(store, count)
    leaf_paths = {store.root / leaf.path for leaf in leaves}
    index_paths = {
        store.root / index.path: "index_payload",
        store.root / (index.path + ".artifact.json"): "index_descriptor",
    }
    measured = dict(leaf_reads=0, leaf_bytes=0, index_payload=0, index_descriptor=0, total_bytes=0)
    read_bytes = Path.read_bytes

    def observed_read(path):
        data = read_bytes(path)
        measured["total_bytes"] += len(data)
        if path in leaf_paths:
            measured["leaf_reads"] += 1
            measured["leaf_bytes"] += len(data)
        if path in index_paths:
            measured[index_paths[path]] += 1
        return data

    monkeypatch.setattr(Path, "read_bytes", observed_read)
    with _verification_scope(store):
        # Прямой лист и selection -> sessions -> index образуют общий ромб.
        for number, leaf in enumerate(leaves):
            for kind in ("prefix", "targets", "cohort"):
                store.publish_json(
                    f"outputs/{kind}/{number}.json",
                    {"session": number, "kind": kind},
                    artifact_dependencies={
                        leaf.path: leaf.descriptor_sha256,
                        selection.path: selection.descriptor_sha256,
                    },
                    **META,
                )
    assert (
        count <= measured["leaf_reads"] <= 3 * count
        and measured["leaf_bytes"] == 65_536 * measured["leaf_reads"]
        and measured["index_payload"] <= 3
        and measured["index_descriptor"] <= 3
        and measured["total_bytes"] <= count * (3 * 65_536 + 32_768) + 32_768
    ), f"N={count}, publications={3 * count}, physical_reads={measured}"


def test_verification_scope_rejects_conflicting_expected_digest_on_cached_path(tmp_path):
    store = LocalArtifactStore(tmp_path)
    leaves, _, selection = _shared_graph(store, 1)
    wrong = "0" * 64 if leaves[0].descriptor_sha256 != "0" * 64 else "1" * 64
    with pytest.raises(ArtifactError):
        with _verification_scope(store):
            store.publish_bytes(
                "good",
                b"good",
                artifact_dependencies={selection.path: selection.descriptor_sha256},
                **META,
            )
            store.publish_bytes(
                "bad", b"bad", artifact_dependencies={leaves[0].path: wrong}, **META
            )
    assert not (store.root / "bad").exists()


def _damage_artifact(store, descriptor, component):
    if component == "descriptor":
        path = store.root / (descriptor.path + ".artifact.json")
        data = json.loads(path.read_bytes())
        data["metadata"]["changed_after_verification"] = True
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        path.write_bytes(encoded)
        (store.root / (descriptor.path + ".artifact.sha256")).write_text(
            sha256_bytes(encoded) + "\n", encoding="ascii"
        )
    else:
        path = store.root / (
            f".objects/{descriptor.sha256}" if component == "object" else descriptor.path
        )
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])


@pytest.mark.parametrize("component", ["payload", "descriptor", "object"])
@pytest.mark.parametrize("node", ["shared", "own"])
def test_verification_scope_fresh_exit_rejects_late_corruption(tmp_path, component, node):
    store = LocalArtifactStore(tmp_path)
    leaves, _, selection = _shared_graph(store, 1)
    with pytest.raises(ArtifactError):
        with _verification_scope(store):
            own = store.publish_bytes(
                "own",
                b"own",
                artifact_dependencies={selection.path: selection.descriptor_sha256},
                **META,
            )
            store.publish_bytes(
                "sibling",
                b"sibling",
                artifact_dependencies={selection.path: selection.descriptor_sha256},
                **META,
            )
            _damage_artifact(store, leaves[0] if node == "shared" else own, component)


@pytest.mark.parametrize("interrupted", [False, True])
def test_verification_scope_does_not_survive_operation_or_exception(tmp_path, interrupted):
    store = LocalArtifactStore(tmp_path)
    parent = store.publish_bytes("parent", b"parent", **META)
    expected_exit = (
        pytest.raises(RuntimeError, match="injected interruption") if interrupted else nullcontext()
    )
    with expected_exit:
        with _verification_scope(store):
            store.publish_bytes(
                "first",
                b"first",
                artifact_dependencies={parent.path: parent.descriptor_sha256},
                **META,
            )
            if interrupted:
                raise RuntimeError("injected interruption")
    _damage_artifact(store, parent, "payload")
    # Вне операции чтение остаётся строгим; следующая операция на том же store не доверяет прошлой.
    with pytest.raises(ArtifactError):
        store.read_bytes(parent.path)
    with pytest.raises(ArtifactError):
        with _verification_scope(store):
            store.publish_bytes(
                "second",
                b"second",
                artifact_dependencies={parent.path: parent.descriptor_sha256},
                **META,
            )
    assert not (store.root / "second").exists()
