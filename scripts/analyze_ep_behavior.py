#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np


targets = {
    "aasist_itw":
    "outputs_v2/aasist/target-runs/in_the_wild/ep_tta/diagnostics.jsonl",

    "ssl_control":
    "outputs_v2/ssl_aasist/target-runs/asv2019_control_test/ep_tta/diagnostics.jsonl"
}


for name, file in targets.items():

    path = Path(file)

    if not path.exists():
        print("missing:", path)
        continue


    delta_score=[]
    delta_z=[]
    r_norm=[]
    margin_active=[]


    with open(path) as f:

        for line in f:

            x=json.loads(line)

            delta_score.append(
                abs(x.get("delta_score",0))
            )

            delta_z.append(
                x.get("delta_z_norm",0)
            )

            r_norm.append(
                x.get("final_R_norm",0)
            )

            margin_active.append(
                x.get("regularizer_active_steps",0)
            )


    delta_score=np.array(delta_score)
    delta_z=np.array(delta_z)
    r_norm=np.array(r_norm)


    print("\n================")
    print(name)
    print("================")

    print("samples:",len(delta_score))

    print(
        "mean |delta score|:",
        delta_score.mean()
    )

    print(
        "median |delta score|:",
        np.median(delta_score)
    )

    print(
        "p95 |delta score|:",
        np.percentile(delta_score,95)
    )

    print(
        "max |delta score|:",
        delta_score.max()
    )

    print(
        "mean delta z:",
        delta_z.mean()
    )

    print(
        "mean R:",
        r_norm.mean()
    )

    print(
        "max R:",
        r_norm.max()
    )

    print(
        "margin active:",
        sum(np.array(margin_active)>0)
    )
