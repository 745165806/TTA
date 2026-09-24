"""Small, dependency-light oracle design and statistics; no target labels in workers."""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
MANIFEST_DIR = ROOT / "experiments/target10_selection/manifests"
RAW = MANIFEST_DIR / "inwild_target10.json"
SELECT = MANIFEST_DIR / "inwild_target10_select.json"
RESULTS = HERE / "results"
SIGNALS = ("selection_score", "view_reduction", "probability_consistency",
           "source_margin_retention", "mean_abs_delta_score", "mean_R_norm")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def config():
    cfg = read_json(HERE / "config.json")
    expected = dict(method="ep_tta_guarded", protocol="episodic", dataset="target10",
                    sample_count=3178, K=[0, 1, 3, 5, 10],
                    lr=[.0003, .001, .003, .01, .03, .1, .3], rho=[.05, .1, .2, .4],
                    gamma=.1, lambda_keep=1., folds=5, seed=2026)
    if cfg != expected:
        raise ValueError("config differs from the first-stage oracle contract")
    return cfg


def candidates():
    cfg = config()
    rows = [dict(candidate_id="k0_frozen", method="Frozen", K=0, steps=0, lr=None, rho=None)]
    for k in cfg["K"][1:]:
        for lr in cfg["lr"]:
            for rho in cfg["rho"]:
                rows.append(dict(candidate_id=f"k{k}_lr{lr:g}_rho{rho:g}",
                                 method=cfg["method"], K=k, steps=k, lr=lr, rho=rho))
    if len(rows) != 113 or len({r["candidate_id"] for r in rows}) != 113:
        raise ValueError("expected exactly 113 unique candidates")
    return rows


def partition(rows, group, groups):
    if not 1 <= groups <= len(rows) or not 0 <= group < groups:
        raise ValueError("invalid candidate group")
    return rows[group::groups]


def oracle_key(row):
    return (row["EER"], -row["AUC"], row["K"], row["lr"] or 0., row["rho"] or 0.)


def unsupervised_key(row):
    return (-row["selection_score"], row["K"], row["lr"] or 0., row["rho"] or 0.)


def selection_regret(selected, best):
    return selected["EER"] - best["EER"]


def ranks(values):
    if any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite rank input")
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for index in order[i:j]:
            result[index] = (i + 1 + j) / 2.
        i = j
    return result


def spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation requires aligned observations")
    a, b = ranks(x), ranks(y)
    a = [v - sum(a) / len(a) for v in a]
    b = [v - sum(b) / len(b) for v in b]
    denom = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
    return sum(v*w for v, w in zip(a, b)) / denom if denom else None


def stratified_folds(ids, labels, seed=2026):
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("fold IDs must be aligned and unique")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("canonical labels required")
    rng = random.Random(seed)  # independent integer-seeded generator, never hash()
    folds = [[] for _ in range(5)]
    for label in (0, 1):
        indices = sorted((i for i, y in enumerate(labels) if y == label), key=lambda i: ids[i])
        if len(indices) < 5:
            raise ValueError("each class requires at least five observations")
        rng.shuffle(indices)
        for fold in range(5):
            folds[fold].extend(indices[fold::5])
    return [sorted(fold) for fold in folds]


def sample_ids():
    from experiments.target10_selection.scripts._common import load_select_sample_ids
    return load_select_sample_ids()


def runtime(asset_root, ids):
    from eptta.models.frozen import verify_frozen_export
    from eptta.offline.artifacts import load_frozen_resources
    from eptta.cache.reader import FeatureCache
    base = Path(asset_root) / "outputs_v2/ssl_aasist"
    bundle, *_ = verify_frozen_export(base / "frozen/bundle.json")
    resource_dir = base / "resources"
    for item in read_json(resource_dir / "resources.json")["files"].values():
        contained_file(resource_dir, item["ref"])
    resources, _, meta = load_frozen_resources(resource_dir, bundle)
    cache = FeatureCache(base / "cache-target-in_the_wild")
    identity, compatibility = cache_provenance(cache.index, bundle)
    if cache.index.get("num_views") != 3 or cache.index.get("feature_dim") != bundle["embedding_dim"]:
        raise ValueError("cache view/embedding shape mismatch")
    for chunk in cache.index["chunks"]:
        for key in ("array_ref", "ids_ref"):
            contained_file(cache.root, chunk[key])
    # Reuse the shared cache, retain only target10 features; no other labels/statistics.
    wanted, features = set(ids), {}
    for chunk_ids, array in cache.iter_chunks():
        for index, sid in enumerate(chunk_ids):
            if sid in wanted:
                features[sid] = array[index].copy()
    if set(features) != wanted:
        raise ValueError("target10 cache coverage mismatch")
    if resources.U.device.type != "cpu":
        raise ValueError("unexpected production resource device")
    return resources, features, cache.cache_id, {
        "asset_root": str(Path(asset_root).resolve()), "baseline_id": bundle["baseline_id"],
        "source_run_id": bundle["source_run_id"], "checkpoint_ref": bundle["checkpoint_ref"],
        "cache_id": cache.cache_id, "cache_identity": identity,
        "cache_compatibility": compatibility,
        "tau0": float(meta["scalars"]["tau0"]), "device": "cpu"}


def contained_file(root, ref):
    root = Path(root).resolve()
    path = (root / ref).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing file or path escapes artifact directory")
    return path


def cache_provenance(index, bundle):
    identity = index["identity"]
    expected = {"source_run_id": bundle["source_run_id"], "checkpoint_ref": bundle["checkpoint_ref"],
                "dataset_id": "in_the_wild", "split_role": "target_test", "preprocess": bundle["preprocess"]}
    if index["format"] == "sharded_npy_v2":
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("cache/bundle scientific provenance mismatch")
        return identity, "explicit_v2_provenance"
    # Existing production readers support this historical cache. Bind it by its
    # ordinary baseline_id; never compare or recalculate old digest fields.
    if index["format"] != "sharded_npy_v1" or identity.get("baseline_id") != bundle["baseline_id"]:
        raise ValueError("historical cache baseline mismatch")
    if any(k in identity and identity[k] != v for k, v in expected.items()):
        raise ValueError("historical cache ordinary field mismatch")
    ordinary = {k: identity[k] for k in ("baseline_id", "wrapper_numerical_version", "seed", "dtype", "numerical_mode", *expected)
                if k in identity}
    return ordinary, "legacy_readonly_baseline_binding_no_preprocess_content_verification"


def validate_output(path):
    path = Path(path).resolve()
    if path.parent != RESULTS.resolve():
        raise ValueError("run directory must be a direct child of oracle_diagnosis/results")
    if path.exists():
        raise FileExistsError(path)
    return path
