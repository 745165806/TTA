"""Summarize source-only v1 and CPU mechanism records; emit a diagnostic plot."""
import json
from pathlib import Path

import numpy as np
from eptta.evaluation.metrics import binary_metrics

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/mechanism_cpu_20261009_normmatched_retry1"
V1 = Path("/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta") / "experiments/meta_audio_tta"
SOURCE_RUN = V1 / "runs/source_gpu_diagnostics_20261008/run_fixed_repair1"
CE_RUN = V1 / "runs/source_gpu_diagnostics_20261008/run_fixed_retry1/ce"

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def pair_flips(before, after, labels):
    useful = harmful = 0
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if label_i != 1 or label_j != 0:
                continue
            a = before[i] - before[j] > 0
            b = after[i] - after[j] > 0
            useful += int(not a and b)
            harmful += int(a and not b)
    return {"useful": useful, "harmful": harmful}

full = {}
for variant in ("ce", "joint", "cross", "same"):
    base = CE_RUN if variant == "ce" else SOURCE_RUN / variant
    rows = read_jsonl(base / "source_val.scores.jsonl")
    labels = [r["canonical_label"] for r in rows]
    k0 = [r["k0"] for r in rows]
    k1 = [r["k1"] for r in rows]
    y = np.asarray(labels)
    def bce(score, binary_labels=labels):
        z = np.asarray(score, dtype=np.float64)
        # score = spoof-logit minus bonafide-logit
        t = np.asarray(binary_labels, dtype=float)
        return np.logaddexp(0.0, z) - t * z
    full[variant] = {"n": len(rows), "k0": binary_metrics(k0, labels, 0.0),
                     "k1": None if all(x is None for x in k1) else binary_metrics(k1, labels, 0.0, k0),
                     "mean_bce_k0": float(bce(k0).mean()),
                     "mean_bce_k1": None if all(x is None for x in k1) else float(bce(k1).mean()),
                     "mean_bce_k0_by_label": {str(c): float(bce([r["k0"] for r in rows if r["canonical_label"] == c], [c]*sum(r["canonical_label"]==c for r in rows)).mean()) for c in (0,1)},
                     "mean_bce_k1_by_label": None if all(x is None for x in k1) else {str(c): float(bce([r["k1"] for r in rows if r["canonical_label"] == c], [c]*sum(r["canonical_label"]==c for r in rows)).mean()) for c in (0,1)},
                     "margin_k0": {"mean": float(np.mean(k0)), "min": float(np.min(k0)), "max": float(np.max(k0))},
                     "margin_k1": None if all(x is None for x in k1) else {"mean": float(np.mean(k1)), "min": float(np.min(k1)), "max": float(np.max(k1))},
                     "pair_flips": None if all(x is None for x in k1) else pair_flips(k0,k1,labels),
                     "score_delta": None if all(x is None for x in k1) else {"mean": float(np.mean(np.asarray(k1)-np.asarray(k0))), "mean_abs": float(np.mean(np.abs(np.asarray(k1)-np.asarray(k0))))}}

fit_acoustic = {}
fit_acoustic_rows = {}
for variant in ("joint", "cross", "same"):
    summary = json.loads((SOURCE_RUN / variant / "summary.json").read_text())
    fit_acoustic[variant] = summary["fit_acoustic"]
    fit_acoustic_rows[variant] = read_jsonl(SOURCE_RUN / variant / "fit_acoustic.scores.jsonl")

fit_acoustic_by_class = {}
for variant, group in fit_acoustic_rows.items():
    fit_acoustic_by_class[variant] = {}
    for condition in ("original", "deterministic_fir"):
        for label in (0, 1):
            part=[r for r in group if r["condition"]==condition and r["canonical_label"]==label]
            delta=np.asarray([r["k1"]-r["k0"] for r in part],dtype=np.float64)
            fit_acoustic_by_class[variant][f"{condition}_label_{label}"]={
                "n":len(part),"mean_k0":float(np.mean([r["k0"] for r in part])),
                "mean_k1":float(np.mean([r["k1"] for r in part])),
                "mean_delta":float(delta.mean()),"mean_abs_delta":float(np.abs(delta).mean()),
                "positive_delta":int(np.count_nonzero(delta>1e-8)),
                "negative_delta":int(np.count_nonzero(delta<-1e-8))}

rows = read_jsonl(RUN / "per_sample.jsonl")
mechanism = {}
for variant in ("joint", "cross", "same"):
    mechanism[variant] = {}
    for condition in ("original", "deterministic_fir"):
        group = [r for r in rows if r["variant"] == variant and r["condition"] == condition]
        mechanism[variant][condition] = {}
        for difficulty in ("hard", "easy", "all"):
            part = group if difficulty == "all" else [r for r in group if r["difficulty"] == difficulty]
            mechanism[variant][condition][difficulty] = {
                "n": len(part), "cosine_mean": float(np.mean([r["gradient_cosine"] for r in part])),
                "cosine_median": float(np.median([r["gradient_cosine"] for r in part])),
                "cosine_min": float(np.min([r["gradient_cosine"] for r in part])),
                "cosine_max": float(np.max([r["gradient_cosine"] for r in part])),
                "byol_mean_ce_delta": float(np.mean([r["byol_ce_loss_after"]-r["k0_ce_loss"] for r in part])),
                "oracle_mean_ce_delta": float(np.mean([r["oracle_ce_loss_after"]-r["k0_ce_loss"] for r in part])),
                "oracle_matched_mean_ce_delta": float(np.mean([r["oracle_matched_ce_loss_after"]-r["k0_ce_loss"] for r in part])),
                "byol_update_l2_mean": float(np.mean([r["byol_update_l2"] for r in part])),
                "oracle_matched_update_l2_mean": float(np.mean([r["oracle_matched_update_l2"] for r in part])),
                "byol_margin_delta_mean": float(np.mean([r["byol_margin_delta"] for r in part])),
                "oracle_margin_delta_mean": float(np.mean([r["oracle_margin_delta"] for r in part]))}
        for label in (0,1):
            part=[r for r in group if r["canonical_label"]==label]
            mechanism[variant][condition][f"label_{label}"]={
                "n":len(part),"cosine_mean":float(np.mean([r["gradient_cosine"] for r in part])),
                "cosine_min":float(np.min([r["gradient_cosine"] for r in part])),
                "cosine_max":float(np.max([r["gradient_cosine"] for r in part])),
                "byol_mean_ce_delta":float(np.mean([r["byol_ce_loss_after"]-r["k0_ce_loss"] for r in part])),
                "oracle_matched_mean_ce_delta":float(np.mean([r["oracle_matched_ce_loss_after"]-r["k0_ce_loss"] for r in part]))}

result = {"source_val_full_fixed_64": full, "fit_fir_existing_diagnostic": fit_acoustic,
          "fit_fir_class_strata": fit_acoustic_by_class,
          "tiny_real_model_gradient_oracle_diagnostic": mechanism,
          "limits": ["Eight source_val examples per checkpoint, four per class; repeated originals and deterministic FIR variants are paired, not independent.",
                     "Difficulty is defined post hoc from the already fixed CE Frozen source_val margin solely for stratified diagnosis; it is not a new data assignment or epoch-selection rule.",
                     "Oracle uses the true source label for a single gradient step and is not a deployable TTA method.",
                     "CPU execution due to absent GPU devices; this preserves model/math but timing is not a GPU resource estimate."]}
out = RUN / "analysis_complete.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

figdir = RUN / "figures_complete"
figdir.mkdir(exist_ok=False)
colors = {"joint": "#4267AC", "cross": "#D17A22", "same": "#39845A"}
labels = {"joint": "Joint", "cross": "Cross", "same": "Same"}
def esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="470" viewBox="0 0 1200 470">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.title{font-size:17px;font-weight:bold}</style>',
       '<text x="600" y="27" text-anchor="middle" class="title">Finite source-only SSL-AASIST mechanism diagnostic (n=8 per checkpoint)</text>']
def line(x1,y1,x2,y2,color="#444",width=1):
    svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>')
def label(x,y,text,anchor="middle",cls="small",rotate=None):
    transform=f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}"{transform}>{esc(text)}</text>')
variants=("joint","cross","same")
# Left panel: cosine, bounded [-1,1].
left=(70,65,500,320); x0,y0,pw,ph=left
label(x0+pw/2,52,"BYOL vs supervised CE gradient cosine, shared BN affine",cls="title")
for tick in (-1,-.5,0,.5,1):
    yy=y0+(1-tick)/2*ph; line(x0,yy,x0+pw,yy,"#ddd"); label(x0-12,yy+4,f"{tick:.1f}","end")
line(x0,y0,x0,y0+ph); line(x0,y0+ph,x0+pw,y0+ph)
for i,v in enumerate(variants):
    cx=x0+pw*(i+.5)/3
    vals=[r["gradient_cosine"] for r in rows if r["variant"]==v and r["condition"]=="original"]
    for j,val in enumerate(vals):
        xx=cx+(j-(len(vals)-1)/2)*9; yy=y0+(1-val)/2*ph
        svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{colors[v]}" opacity=".8"/>')
    mean=float(np.mean(vals)); yy=y0+(1-mean)/2*ph; line(cx-24,yy,cx+24,yy,colors[v],3)
    label(cx,y0+ph+22,labels[v])
label(x0-42,y0+ph/2,"Cosine",rotate=-90)
# Right panel: loss delta, sign-preserving symmetric range.
rx,ry,rw,rh=650,65,480,320
label(rx+rw/2,52,"Same-sample CE loss change after one step",cls="title")
losses=[r[f"{method}_ce_loss_after"]-r["k0_ce_loss"] for r in rows if r["condition"]=="original" for method in ("byol","oracle_matched") if r[f"{method}_ce_loss_after"] is not None]
limit=max(abs(min(losses)),abs(max(losses)))*1.15
for tick in (-limit,-limit/2,0,limit/2,limit):
    yy=ry+(limit-tick)/(2*limit)*rh; line(rx,yy,rx+rw,yy,"#ddd"); label(rx-12,yy+4,f"{tick:.3f}","end")
line(rx,ry,rx,ry+rh); line(rx,ry+rh,rx+rw,ry+rh)
for i,v in enumerate(variants):
    cx=rx+rw*(i+.5)/3
    for offset,method,color in ((-.12,"byol","#9C3D50"),(.12,"oracle_matched","#306B9C")):
        vals=[r[f"{method}_ce_loss_after"]-r["k0_ce_loss"] for r in rows if r["variant"]==v and r["condition"]=="original"]
        mean=float(np.mean(vals)); xx=cx+offset*rw/3; yy=ry+(limit-mean)/(2*limit)*rh
        for j,val in enumerate(vals):
            px=xx+(j-(len(vals)-1)/2)*5; py=ry+(limit-val)/(2*limit)*rh
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}" opacity=".72"/>')
        line(xx-16,yy,xx+16,yy,color,3)
    label(cx,ry+rh+22,labels[v])
label(rx-43,ry+rh/2,"CE loss Δ (negative = lower)",rotate=-90)
line(840,430,860,430,"#9C3D50",3); label(866,434,"BYOL",anchor="start")
line(925,430,945,430,"#306B9C",3); label(951,434,"norm-matched CE oracle",anchor="start")
label(600,458,"Hard/easy ranks use CE Frozen source margin; repeated FIR views are not independent.")
svg.append('</svg>')
figpath=figdir/"gradient_alignment_and_loss.svg"
if figpath.exists(): raise FileExistsError(figpath)
figpath.write_text("\n".join(svg)+"\n")
print(json.dumps({"status":"PASS","analysis":str(out),"figure":str(figpath),"source_val_counts":{k:v["n"] for k,v in full.items()},"gradient_n":{k:len([r for r in rows if r["variant"]==k and r["condition"]=="original"]) for k in ("joint","cross","same")}},sort_keys=True))
