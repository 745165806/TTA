"""CPU inference-only matched-epoch probe on a fixed source_val subset."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]
import torch
from stage1_source import _dataset
from experiments.meta_audio_tta.meta_helpers import load_checkpoint as _load_checkpoint
from eptta.evaluation.metrics import binary_metrics


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    config = json.loads(args.source_config.read_text())
    source = _dataset(config, "source_val")
    selected = args.selected_scores.read_text().splitlines()
    selected_rows = [json.loads(line) for line in selected if line]
    by_class = {c: [r for r in selected_rows if r["canonical_label"] == c] for c in (0,1)}
    fixed = []
    for c in (0,1):
        ranked = sorted(by_class[c], key=lambda r:(abs(r["k0"]),r["sample_id"]))
        fixed.extend(("hard",r) for r in ranked[:2])
        fixed.extend(("easy",r) for r in ranked[-2:])
    row_map = {r["sample_id"]:i for i,r in enumerate(source.rows)}
    indices = [row_map[r["sample_id"]] for _,r in fixed]
    epochs=(1,2,4,6,8)
    records=[]; summary={}
    with (args.output/"scores.jsonl").open("x") as stream:
        for epoch in epochs:
            for variant in ("ce","joint"):
                checkpoint=args.checkpoint_root/variant/"checkpoints"/f"epoch_{epoch:04d}.pt"
                model,system=_load_checkpoint(config,checkpoint,torch.device("cpu"))
                model.eval()
                scores=[]; labels=[]
                for (difficulty,_),idx in zip(fixed,indices):
                    waveform,label,sample_id=source[idx]
                    with torch.inference_mode():
                        logits=model(waveform.reshape(1,-1))
                        if logits.shape!=(1,2) or not torch.isfinite(logits).all():
                            raise FloatingPointError("invalid matched-epoch logits")
                        margin=float(logits[0,0]-logits[0,1])
                    row={"variant":variant,"epoch":epoch,"sample_id":sample_id,
                         "canonical_label":int(label),"difficulty":difficulty,"score":margin}
                    records.append(row); scores.append(margin); labels.append(int(label))
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
                summary.setdefault(variant,{})[str(epoch)]={"checkpoint":str(checkpoint),
                    "n":len(scores),"eer":binary_metrics(scores,labels,0.0)["eer"],
                    "auroc":binary_metrics(scores,labels,0.0)["auroc"],
                    "hard_eer":binary_metrics([r["score"] for r in records if r["variant"]==variant and r["epoch"]==epoch and r["difficulty"]=="hard"],
                                               [r["canonical_label"] for r in records if r["variant"]==variant and r["epoch"]==epoch and r["difficulty"]=="hard"],0.0)["eer"],
                    "easy_eer":binary_metrics([r["score"] for r in records if r["variant"]==variant and r["epoch"]==epoch and r["difficulty"]=="easy"],
                                               [r["canonical_label"] for r in records if r["variant"]==variant and r["epoch"]==epoch and r["difficulty"]=="easy"],0.0)["eer"]}
                del system,model
    out={"status":"PASS","device":"cpu","epochs":epochs,"fixed_source_val_ids":[r["sample_id"] for _,r in fixed],
         "difficulty_source":"selected CE epoch6 Frozen margins within pre-existing 32-per-class source_val diagnostic set",
         "matched_epoch_summary":summary,"record_count":len(records),"command":sys.argv}
    (args.output/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"PASS","records":len(records),"output":str(args.output)},sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-config",type=Path,required=True)
    p.add_argument("--checkpoint-root",type=Path,required=True)
    p.add_argument("--selected-scores",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=8)
    run(p.parse_args())
