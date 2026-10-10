import json
import statistics
from pathlib import Path

path = Path(__file__).parent / "runs/mechanism_cpu_20261009_retry1/per_sample.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines() if line]
for variant in ("joint", "cross", "same"):
    print("\n", variant)
    for condition in ("original", "deterministic_fir"):
        group = [row for row in rows if row["variant"] == variant and row["condition"] == condition]
        print(condition, "n", len(group),
              "cos mean", statistics.mean(row["gradient_cosine"] for row in group),
              "cos median", statistics.median(row["gradient_cosine"] for row in group),
              "dot mean", statistics.mean(row["gradient_dot"] for row in group),
              "norms BYOL/CE", statistics.mean(row["byol_gradient_l2"] for row in group),
              statistics.mean(row["ce_gradient_l2"] for row in group),
              "mean delta CE BYOL/oracle",
              statistics.mean(row["byol_ce_loss_after"] - row["k0_ce_loss"] for row in group),
              statistics.mean(row["oracle_ce_loss_after"] - row["k0_ce_loss"] for row in group))
        for difficulty in ("hard", "easy"):
            part = [row for row in group if row["difficulty"] == difficulty]
            print(" ", difficulty, "n", len(part),
                  "cos", statistics.mean(row["gradient_cosine"] for row in part),
                  "dCE BYOL/oracle",
                  statistics.mean(row["byol_ce_loss_after"] - row["k0_ce_loss"] for row in part),
                  statistics.mean(row["oracle_ce_loss_after"] - row["k0_ce_loss"] for row in part))
for row in rows:
    if row["condition"] == "original":
        print("ITEM", row["variant"], row["sample_id"][:12], row["canonical_label"], row["difficulty"],
              row["gradient_cosine"], row["byol_ce_loss_after"] - row["k0_ce_loss"] if row["byol_ce_loss_after"] is not None else None,
              row["oracle_ce_loss_after"] - row["k0_ce_loss"])
