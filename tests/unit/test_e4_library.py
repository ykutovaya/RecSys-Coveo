"""B05: реальный LightGBM 4.6.0, послабления только для малых эталонов."""

import lightgbm as lgb
import numpy as np

from tests.reference.e4_cases import recipe


def train_one(data, gradients, hessians, *, categorical=False):
    params = recipe()["model"]
    params.update(
        num_leaves=2,
        min_data_in_leaf=1,
        min_sum_hessian_in_leaf=0,
        min_data_per_group=1,
        max_depth=1,
        verbosity=-1,
        objective=lambda scores, dataset: (gradients, hessians),
    )
    ds = lgb.Dataset(
        data,
        label=np.zeros(len(data)),
        group=[len(data)],
        categorical_feature=[0] if categorical else [],
        params={"min_data_in_bin": 1, "feature_pre_filter": False},
    )
    assert ds.weight is None and ds.init_score is None
    return lgb.train(params, ds, num_boost_round=1)


def test_numeric_newton_leaf_and_double_gain():
    x = np.repeat([[0.0], [1.0]], 4, axis=0)
    g = np.array([-1.0] * 4 + [1.0] * 4, dtype=np.float32)
    h = np.ones(8, dtype=np.float32)
    booster = train_one(x, g, h)
    tree = booster.dump_model()["tree_info"][0]["tree_structure"]
    expected_gain = 0.5 * (16 / 5 + 16 / 5)
    assert np.isclose(tree["split_gain"], 2 * expected_gain, atol=1e-6)
    np.testing.assert_allclose(
        booster.predict(x, raw_score=True, num_threads=1),
        [0.05 * 4 / 5] * 4 + [-0.05 * 4 / 5] * 4,
        atol=1e-10,
        rtol=1e-8,
    )


def test_grouped_categories_keep_zero_curvature_category_finite():
    x = np.repeat(np.arange(5), 20).astype(float).reshape(-1, 1)
    g = np.repeat([0, -1, -0.8, 0.8, 1], 20).astype(np.float32)
    h = np.repeat([0, 1, 1, 1, 1], 20).astype(np.float32)
    booster = train_one(x, g, h, categorical=True)
    tree = booster.dump_model()["tree_info"][0]["tree_structure"]
    assert tree["decision_type"] == "=="
    assert len(tree["threshold"].split("||")) > 1  # Не one-hot.
    assert np.isfinite(booster.predict(x, raw_score=True, num_threads=1)).all()
    keys = [float(g[x[:, 0] == c].sum()) / (float(h[x[:, 0] == c].sum()) + 1) for c in range(5)]
    assert np.isfinite(keys).all() and keys[0] == 0
