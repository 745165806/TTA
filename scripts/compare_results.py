import json
from pathlib import Path


roots=[
"outputs_v2/aasist/target-runs",
"outputs_v2/ssl_aasist/target-runs"
]


for root in roots:

    print("\n====",root,"====")

    for m in Path(root).glob("**/metrics.json"):

        data=json.loads(
            m.read_text()
        )

        print(
            m.parent,
            "EER=",
            data["metrics"]["eer"]
        )
