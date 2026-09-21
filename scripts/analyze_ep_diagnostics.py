#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np


paths = [
    "outputs_v2/aasist/target-runs/in_the_wild/ep_tta/diagnostics.jsonl",
    "outputs_v2/ssl_aasist/target-runs/asv2019_control_test/ep_tta/diagnostics.jsonl"
]


for path in paths:

    p=Path(path)

    if not p.exists():
        print("skip:",path)
        continue


    delta_score=[]
    delta_z=[]
    r_norm=[]
    margin_loss=[]
    active=[]


    with open(p) as f:

        for line in f:

            x=json.loads(line)

            delta_score.append(
                x.get("delta_score",0)
            )

            delta_z.append(
                x.get("delta_z_norm",0)
            )

            r_norm.append(
                x.get("final_R_norm",0)
            )

            margin_loss.append(
                x.get("final_margin_loss",0)
            )

            active.append(
                x.get("regularizer_active_steps",0)
            )


    delta_score=np.array(delta_score)
    delta_z=np.array(delta_z)
    r_norm=np.array(r_norm)


    print("\n====================")
    print(path)
    print("====================")


    print("samples:",len(delta_score))


    print(
        "mean |delta_score|:",
        np.mean(np.abs(delta_score))
    )

    print(
        "max |delta_score|:",
        np.max(np.abs(delta_score))
    )


    print(
        "mean delta_z_norm:",
        np.mean(delta_z)
    )


    print(
        "mean R norm:",
        np.mean(r_norm)
    )


    print(
        "max R norm:",
        np.max(r_norm)
    )


    print(
        "margin active samples:",
        np.sum(np.array(active)>0)
    )

