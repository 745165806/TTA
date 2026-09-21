import json
import csv


def analyze(
    score_file,
    label_file,
    threshold,
    name
):

    scores={}

    with open(score_file) as f:
        reader=csv.DictReader(f)

        for r in reader:
            scores[r["sample_id"]] = (
                float(r["score_before"]),
                float(r["score"])
            )


    labels={}

    with open(label_file) as f:
        for line in f:
            r=json.loads(line)

            labels[r["sample_id"]] = int(
                r.get("label",
                      r.get("canonical_label"))
            )


    helpful=0
    harmful=0


    for sid,(before,after) in scores.items():

        if sid not in labels:
            continue


        y=labels[sid]


        pred_before = int(before>threshold)

        pred_after = int(after>threshold)


        if pred_before != y and pred_after==y:
            helpful+=1


        if pred_before == y and pred_after!=y:
            harmful+=1


    print(name)

    print("samples:",len(scores))

    print("helpful:",helpful)

    print("harmful:",harmful)



analyze(
"outputs_v2/aasist/target-runs/in_the_wild/ep_tta/scores.csv",
"data/manifests_v2/in_the_wild/labels/target_test.jsonl",
-6.133878707885742,
"AASIST ITW"
)


analyze(
"outputs_v2/ssl_aasist/target-runs/asv2019_control_test/ep_tta/scores.csv",
"data/manifests_v2/asv2019_la/labels/control_test.jsonl",
-4.770049095153809,
"SSL control"
)
