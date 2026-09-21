import json
import numpy as np
from pathlib import Path


cases = {
    "aasist_itw":
    {
        "diag":
        "outputs_v2/aasist/target-runs/in_the_wild/ep_tta/diagnostics.jsonl",

        "score":
        "outputs_v2/aasist/target-runs/in_the_wild/ep_tta/scores.csv",

        "threshold":
        -6.133878707885742
    },

    "ssl_control":
    {
        "diag":
        "outputs_v2/ssl_aasist/target-runs/asv2019_control_test/ep_tta/diagnostics.jsonl",

        "score":
        "outputs_v2/ssl_aasist/target-runs/asv2019_control_test/ep_tta/scores.csv",

        "threshold":
        -4.770049095153809
    }
}


for name,c in cases.items():

    delta=[]
    distance=[]


    with open(c["diag"]) as f:

        for line in f:

            x=json.loads(line)

            d=abs(
                x["delta_score"]
            )

            delta.append(d)


    import csv

    with open(c["score"]) as f:

        reader=csv.DictReader(f)

        for row in reader:

            s=float(row["score_before"])

            distance.append(
                abs(s-c["threshold"])
            )


    delta=np.array(delta)
    distance=np.array(distance)


    print("\n======",name,"======")

    print(
        "mean delta:",
        delta.mean()
    )

    print(
        "large change >1e-3:",
        np.sum(delta>1e-3)
    )

    print(
        "near threshold (<0.5):",
        np.sum(distance<0.5)
    )

    print(
        "correlation:",
        np.corrcoef(
            delta,
            distance
        )[0,1]
    )
