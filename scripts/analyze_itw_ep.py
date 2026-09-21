import json
from pathlib import Path
import numpy as np


root = Path(
"outputs_v2/aasist/target-runs/in_the_wild/ep_tta"
)

diag = root / "diagnostics.jsonl"


before=[]
after=[]
r=[]


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


delta=after-before


print("samples:",len(delta))
print("mean delta:",delta.mean())
print("std delta:",delta.std())
print("max delta:",abs(delta).max())

if r:
    print("mean R:",np.mean(r))
    print("max R:",np.max(r))


np.save(
"outputs_v2/r7_itw_score_delta.npy",
delta
)
