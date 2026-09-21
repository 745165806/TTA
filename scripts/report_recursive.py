import json
from pathlib import Path
import csv


roots=[
    "outputs_v2/aasist/target-runs",
    "outputs_v2/ssl_aasist/target-runs"
]


rows=[]


for root in roots:

    for p in Path(root).rglob("metrics.json"):

        data=json.loads(
            p.read_text()
        )

        m=data["metrics"]

        rows.append(
            {
            "path":str(p),
            "eer":m.get("eer"),
            "auroc":m.get("auroc"),
            "fpr":m.get("fpr"),
            "fnr":m.get("fnr"),
            "count":m.get("count")
            }
        )


with open(
    "outputs_v2/all_results_recursive.csv",
    "w",
    newline=""
) as f:

    writer=csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)


print(
    "saved:",
    len(rows),
    "results"
)
