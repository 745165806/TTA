#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np


root = Path(
"outputs_v2/ssl_aasist/target-runs/in_the_wild/ep_tta"
)


diag = root / "diagnostics.jsonl"


before=[]
after=[]
r=[]


if not diag.exists():
    print("missing",diag)
    exit()


with open(diag) as f:
    for line in f:
        x=json.loads(line)

        if "score_before" in x:
            before.append(x["score_before"])

        if "score_after" in x:
            after.append(x["score_after"])

        if "r_fro" in x:
            r.append(x["r_fro"])


before=np.array(before)
after=np.array(after)


print("======================")
print("ITW EP analysis")
print("======================")


print("samples:",len(before))


if len(before):

    delta=after-before

    print(
        "mean delta:",
        delta.mean()
    )

    print(
        "std delta:",
        delta.std()
    )

    print(
        "max change:",
        abs(delta).max()
    )


if r:
    print(
        "mean R:",
        np.mean(r)
    )

    print(
        "max R:",
        np.max(r)
    )
