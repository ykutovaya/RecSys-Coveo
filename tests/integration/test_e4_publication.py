"""Повреждение модели до публикации не создаёт завершённого результата."""

import lightgbm as lgb
import pytest

from tests.integration.test_e3_pipeline import e3_settings  # noqa: F401
from tests.integration.test_e4_pipeline import ROOT, e4_settings  # noqa: F401
from tests.unit.test_e2_input import prepared_source  # noqa: F401


@pytest.mark.parametrize("replacement", ["corrupt", "different_model"])
def test_model_is_checked_after_last_hook_before_completed_report(
    e4_settings,  # noqa: F811
    tmp_path,
    replacement,
):
    from coveo_recsys.cli.training import run_training

    configuration, _, _ = e4_settings
    destination = tmp_path / "ranker"

    def replace_model():
        path = destination / "models/ranker.txt"
        if replacement == "corrupt":
            path.write_bytes(b"corrupt\n")
        else:
            altered = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            altered.set_leaf_output(0, 0, altered.get_leaf_output(0, 0) + 1.0)
            path.write_text(altered.model_to_string(), encoding="utf-8")

    with pytest.raises(ValueError):
        run_training(ROOT, configuration, destination, "e4", before_publish=replace_model)
    assert not (destination / "reports/ranker.json").exists()
