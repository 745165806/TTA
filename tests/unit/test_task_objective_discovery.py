"""Fixed objective-discovery budget shared by per-sample and local arms."""
import json
from pathlib import Path


def test_preregistered_budget_is_unchanged():
    root = Path(__file__).parents[2]
    config = json.loads((root / "experiments/task_objective_discovery/objective_config.json").read_text())
    assert (config["steps"], config["lr"], config["rho"], config["lambda_keep"]) == (5, .03, .1, 0.)
    assert config["parameter_scope"] == "production_R_8x8"
