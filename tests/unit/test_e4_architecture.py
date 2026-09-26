"""Загрузка и применение ранжировщика не зависят от обучающих целей."""

from pathlib import Path

from tests.unit.test_architecture import imports


def test_prediction_and_loading_do_not_import_training_or_future_data():
    directory = Path(__file__).resolve().parents[2] / "src/coveo_recsys/ranking"
    forbidden = {"train", "table", "objective", "targets", "training", "data", "cli"}
    for filename in ("__init__.py", "load.py", "predict.py", "matrix.py"):
        path = directory / filename
        assert path.exists(), path
        for name in imports(path):
            assert not (set(name.split(".")) & forbidden), (path, name)
