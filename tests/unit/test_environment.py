import importlib.metadata
import subprocess
import sys
from pathlib import Path

import jsonschema
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.stats import beta


def test_environment_smoke(tmp_path):
    python_version = (Path(__file__).resolve().parents[2] / ".python-version").read_text().strip()
    assert sys.version_info[:3] == tuple(int(part) for part in python_version.split("."))
    assert np.__version__ == "2.1.3"
    assert lgb.__version__ == "4.6.0"
    assert np.dot([1.0, 2.0], [3.0, 4.0]) == 11.0
    assert 0 < beta.ppf(0.025, 2, 4) < 1
    jsonschema.validate({"n": 1}, {"type": "object", "properties": {"n": {"type": "integer"}}})
    table = pa.table({"id": ["я", "e\u0301"], "time_ms": [2**63 - 1, -(2**63)]})
    path = tmp_path / "test.parquet"
    pq.write_table(table, path, compression="zstd")
    assert pq.read_table(path).equals(table)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/health").json() == {"ok": True}
    assert importlib.metadata.version("uvicorn") == "0.34.2"


def test_lightgbm_custom_objective_fp32():
    # Только совместимость API. Это не нормативный callback ранжировщика Э4.
    x = np.arange(16, dtype=np.float64).reshape(8, 2)
    y = np.arange(8, dtype=np.float64)
    calls = []

    def objective(preds, dataset):
        gradient = np.ascontiguousarray(preds - dataset.get_label(), dtype=np.float32)
        hessian = np.ones_like(gradient)
        calls.append((gradient.dtype, gradient.flags.c_contiguous))
        assert dataset.get_weight() is None
        return gradient, hessian

    params = {
        "objective": objective,
        "metric": "None",
        "boosting": "gbdt",
        "data_sample_strategy": "bagging",
        "device_type": "cpu",
        "tree_learner": "serial",
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "force_row_wise": False,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
        "lambda_l1": 0,
        "lambda_l2": 1,
        "min_gain_to_split": 0,
        # Только для малого испытания совместимости ослаблены размеры групп/листьев.
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.001,
        "bagging_fraction": 1,
        "bagging_freq": 0,
        "bagging_by_query": False,
        "feature_fraction": 1,
        "feature_fraction_bynode": 1,
        "extra_trees": False,
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 200000,
        "feature_pre_filter": False,
        "enable_bundle": False,
        "use_missing": True,
        "zero_as_missing": False,
        "use_quantized_grad": False,
        "boost_from_average": False,
        "linear_tree": False,
        "path_smooth": 0,
        "max_delta_step": 0,
        "cat_l2": 0,
        "cat_smooth": 1,
        "max_cat_threshold": 32,
        "max_cat_to_onehot": 4,
        "min_data_per_group": 1,
        "seed": 17,
        "data_random_seed": 17,
        "feature_fraction_seed": 17,
        "bagging_seed": 17,
        "extra_seed": 17,
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        lgb.Dataset(x, label=y, group=[4, 4], weight=None, init_score=None, free_raw_data=False),
        num_boost_round=2,
        valid_sets=None,
        feval=None,
        init_model=None,
        callbacks=None,
    )
    assert calls == [(np.dtype("float32"), True)] * 2
    assert np.isfinite(model.predict(x, raw_score=True, num_iteration=2)).all()


def test_base_import_without_torch():
    script = "import coveo_recsys, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)
