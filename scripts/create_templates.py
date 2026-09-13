"""Initial v0.1.0 template creation. Exclusive writes protect existing files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"


def write(path, value):
    dest = ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "additionalProperties": False,
            "required": list(properties) if required is None else required}


def enum(*values):
    return {"enum": list(values), "type": "string"}


S = {"type": "string", "minLength": 1}
NS = {"type": ["string", "null"], "minLength": 1}
B = {"type": "boolean"}
I = {"type": "integer", "minimum": 0}
N = {"type": "number"}
V = enum(VERSION)
def arr(item, **kwargs):
    return {"type": "array", "items": item, **kwargs}
def nullable(spec):
    result = dict(spec, type=[spec["type"], "null"])
    if "enum" in spec:
        result["enum"] = [*spec["enum"], None]
    return result

DATASETS = ["asvspoof2019_la", "asvspoof2021_df", "asvspoof2021_la", "wavefake",
            "codecfake_xie", "in_the_wild"]
ROLES = ["fit", "source_val", "select", "cal0", "audit", "control_test", "target_test", "cal1"]
approval = nullable(obj({"reviewer": S, "approved_at": S, "report_ref": S,
                         "sample_evidence_ref": S, "content_sha256": S}))
payloads = {
    "raw": obj({"format": nullable(enum("delimited", "json", "jsonl", "sidecar")),
                "encoding": NS, "delimiter": NS, "header": nullable(B),
                "columns": nullable({"type": "object", "additionalProperties": {"type": ["string", "integer"]}}),
                "json_paths": nullable({"type": "object", "additionalProperties": S}),
                "record_id": NS, "audio_path_rule": NS, "label_field": NS,
                "allowed_values": nullable(arr({"type": ["string", "integer"]})),
                "missing_policy": NS}),
    "label": obj({"policy_id": NS, "raw_to_canonical": nullable({"type": "object", "additionalProperties": {"type": "integer", "enum": [0, 1]}}),
                  "unknown_policy": enum("quarantine", "error")}),
    "group": obj({"resolver": NS, "source_mapping_ref": NS, "group_quality": NS}),
    "preprocess": obj({"decode": NS, "train_unit": NS, "eval_unit": NS,
                       "source_probe": NS, "target_probe": NS, "quality_policy": NS}),
    "split": obj({"assignments_ref": NS, "group_policy_hash": NS,
                  "ratios": nullable({"type": "object", "additionalProperties": N}),
                  "counts": nullable({"type": "object", "additionalProperties": I}),
                  "preserve_existing_assignments": {"type": "boolean", "enum": [True]},
                  "new_group_default_role": enum("unassigned", "quarantine"),
                  "automatic_resplit": {"type": "boolean", "enum": [False]},
                  "automatic_retrain": {"type": "boolean", "enum": [False]}}),
    "architecture": obj({"repo_commit": NS, "patch_sha256": NS, "config_ref": NS,
                          "class_index_map": nullable(obj({"bonafide": {"type": "integer", "enum": [0, 1]},
                                                          "spoof": {"type": "integer", "enum": [0, 1]}}))}),
    "recipe": obj({"optimizer": NS, "lr": nullable(N), "max_epochs": nullable(I),
                   "scheduler": NS, "loss": NS, "class_weights_by_name": nullable(obj({"bonafide": N, "spoof": N})),
                   "sampler_policy": NS, "augmentation_recipe_ref": NS,
                   "trainable_scope": NS, "selection_metric": enum("source_val_eer"),
                   "tie_break": enum("earliest_epoch")}),
    "snapshot": obj({"snapshot_id": NS, "manifest_ref": NS, "manifest_sha256": NS,
                     "fixture_only": {"type": "boolean", "enum": [False]}}),
    "initialization": obj({"scope": enum("generic_ssl_frontend_only"), "artifact_ref": NS,
                           "sha256": NS, "pretraining_provenance": NS}),
    "frozen_bundle": obj({"bundle_ref": NS, "selected_checkpoint_sha256": NS,
                          "training_run_id": NS, "training_status": nullable(enum("FINALIZED")),
                          "task_weight_origin": enum("trained_in_project"),
                          "parity_report_ref": NS}),
    "source_resources": obj({"bundle_id": NS, "U_ref": NS, "anchors_ref": NS,
                             "tau0_ref": NS, "frozen_bundle_sha256": NS}),
    "score_seal": obj({"run_id": NS, "scores_sha256": NS, "coverage_report_ref": NS}),
}
contract_schemas = {}
templates = {}
for kind, payload in payloads.items():
    contract_schemas[kind] = obj({"schema_version": V, "status": enum("UNRESOLVED", "PROPOSED", "APPROVED", "LOCKED"),
                                  "approval": approval, "payload": payload})
    vals = {key: None for key in payload["properties"]}
    if kind == "label": vals["unknown_policy"] = "quarantine"
    if kind == "split": vals.update(preserve_existing_assignments=True, new_group_default_role="unassigned", automatic_resplit=False, automatic_retrain=False)
    if kind == "recipe": vals.update(selection_metric="source_val_eer", tie_break="earliest_epoch")
    if kind == "snapshot": vals["fixture_only"] = False
    if kind == "initialization": vals["scope"] = "generic_ssl_frontend_only"
    if kind == "frozen_bundle": vals["task_weight_origin"] = "trained_in_project"
    templates[kind] = {"schema_version": VERSION, "status": "UNRESOLVED", "approval": None, "payload": vals}
    write(f"src/eptta/schemas/{kind}.json", dict(contract_schemas[kind], **{"$schema": "https://json-schema.org/draft/2020-12/schema", "title": f"EP-TTA {kind} 0.1.0"}))

permissions = obj({k: B for k in ["real_data_io", "model_weight_io", "network_download", "remote_submission", "real_source_training"]})
runtime = obj({"profile_id": enum("local_dev", "remote_a6000"), "environment": enum("local", "remote"),
               "permissions": permissions, "device": enum("cpu", "cuda"),
               "physical_gpu_ids": arr(I, uniqueItems=True), "episode_batch_size": {"type": "integer", "minimum": 1},
               "synchronize_episode_gradients": {"type": "boolean", "enum": [False]}})
paths = obj({"work_root": NS, "dataset_roots": obj({d: NS for d in DATASETS}),
             "protocol_files": obj({d: NS for d in DATASETS}),
             "source_repos": obj({"aasist": NS, "ssl_aasist": NS}),
             "generic_ssl_initialization": NS})
selection = obj({"dataset_ids": arr(enum(*DATASETS), minItems=1, uniqueItems=True),
                 "model_id": enum("aasist_source", "ssl_aasist_source"), "method_id": S})
defaults = obj({"rank": {"type": "integer", "minimum": 1}, "steps": I,
                "lr": {"type": "number", "exclusiveMinimum": 0},
                "rho": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "gamma": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
                "regularizer_weight": {"type": "number", "minimum": 0},
                "momentum": {"type": "number", "enum": [0.0]}, "weight_decay": {"type": "number", "enum": [0.0]}})
dataset_contracts = obj({k: contract_schemas[k] for k in ("raw", "label", "group", "snapshot")})
config_schema = obj({"schema_version": V, "project_version": V, "design_version": V,
                     "method_contract_version": V, "project": enum("ep_tta"),
                     "protocol": obj({"canonical_labels": obj({"bonafide": {"type": "integer", "enum": [0]}, "spoof": {"type": "integer", "enum": [1]}}),
                                      "score_direction": enum("larger_is_spoof"), "decision_rule": enum("strict_greater"),
                                      "target_labels_visible": {"type": "boolean", "enum": [False]},
                                      "target_history_visible": {"type": "boolean", "enum": [False]},
                                      "target_population_statistics_visible": {"type": "boolean", "enum": [False]}}),
                     "selection": selection, "runtime": runtime, "paths": paths,
                     "model_policy": obj({"task_weights": enum("train_in_project"), "allow_external_task_checkpoint": {"type": "boolean", "enum": [False]}, "allow_random_weight_fallback": {"type": "boolean", "enum": [False]}}),
                     "probe": obj({"num_views": {"type": "integer", "enum": [3]}, "view0": enum("identity"),
                                   "regenerate_per_step": {"type": "boolean", "enum": [False]}, "seed": I}),
                     "defaults": defaults,
                     "contracts": obj({"datasets": obj({d: dataset_contracts for d in DATASETS}),
                                        **{k: contract_schemas[k] for k in payloads if k not in ("raw", "label", "group", "snapshot")}}),
                     "safety": obj({k: {"type": "boolean", "enum": [False]} for k in ["automatic_download", "automatic_ssh", "unresolved_contract_execution", "destructive_sync", "silent_algorithm_fallback"]})})
write("src/eptta/schemas/config.json", dict(config_schema, **{"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "EP-TTA config 0.1.0"}))
local = {"profile_id": "local_dev", "environment": "local", "permissions": {k: False for k in permissions["properties"]}, "device": "cpu", "physical_gpu_ids": [], "episode_batch_size": 1, "synchronize_episode_gradients": False}
remote = dict(local, profile_id="remote_a6000", environment="remote", physical_gpu_ids=[0, 1], episode_batch_size=128,
              permissions=dict(local["permissions"], real_data_io=True, model_weight_io=True, real_source_training=True))
bindings = {"work_root": None, "dataset_roots": dict.fromkeys(DATASETS), "protocol_files": dict.fromkeys(DATASETS),
            "source_repos": {"aasist": None, "ssl_aasist": None}, "generic_ssl_initialization": None}
selected = {"dataset_ids": ["asvspoof2019_la"], "model_id": "ssl_aasist_source", "method_id": "ep_tta"}
base = {"schema_version": VERSION, "project_version": VERSION, "design_version": VERSION, "method_contract_version": VERSION,
        "project": "ep_tta", "protocol": {"canonical_labels": {"bonafide": 0, "spoof": 1}, "score_direction": "larger_is_spoof", "decision_rule": "strict_greater", "target_labels_visible": False, "target_history_visible": False, "target_population_statistics_visible": False},
        "selection": selected, "runtime": local, "paths": bindings,
        "model_policy": {"task_weights": "train_in_project", "allow_external_task_checkpoint": False, "allow_random_weight_fallback": False},
        "probe": {"num_views": 3, "view0": "identity", "regenerate_per_step": False, "seed": 13},
        "defaults": {"rank": 8, "steps": 3, "lr": .01, "rho": .2, "gamma": .1, "regularizer_weight": 1., "momentum": 0., "weight_decay": 0.},
        "contracts": {"datasets": {d: {k: templates[k] for k in ("raw", "label", "group", "snapshot")} for d in DATASETS},
                      **{k: templates[k] for k in payloads if k not in ("raw", "label", "group", "snapshot")}},
        "safety": {k: False for k in config_schema["properties"]["safety"]["properties"]}}
write("configs/base.yaml", base)
for name, value in [("local_dev", local), ("remote_a6000", remote)]:
    write(f"configs/profiles/{name}.yaml", {"schema_version": VERSION, "runtime": value})
write("configs/paths.remote.yaml.example", {"schema_version": VERSION, "paths": bindings})
write("configs/experiments/source_pilot.yaml", {"schema_version": VERSION, "selection": selected})
for d in DATASETS:
    write(f"configs/data_contracts/{d}.yaml.example", {k: templates[k] for k in ("raw", "label", "group", "snapshot")})
for name, kind in [("preprocess/source", "preprocess"), ("splits/source", "split"), ("splits/incremental", "split")]:
    write(f"configs/{name}.yaml.example", templates[kind])
for model in ["aasist", "ssl_aasist"]:
    write(f"configs/training/{model}.yaml", templates["recipe"])
write("configs/source_training_plan.yaml", {"schema_version": VERSION, "status": "DRAFT", "execution_environment": "remote_only", "source_dataset_id": "asvspoof2019_la", "model_ids": ["aasist_source", "ssl_aasist_source"], "fit_snapshot_ref": None, "source_val_snapshot_ref": None, "split_plan_lock_ref": None, "preprocess_lock_ref": None, "automatic_followup_full_training": False})

models = {f"{m}_source": {"architecture_plugin": f"{m}_author", "repo_commit": None,
          "task_weights": "native_initialization", "pretrained_frontend": "xlsr_300m" if m == "ssl_aasist" else None,
          "pretrained_scope": "generic_ssl_frontend_only" if m == "ssl_aasist" else None,
          "training_regime": "joint_finetune" if m == "ssl_aasist" else "all_native_trainable_parameters",
          "allow_external_task_checkpoint": False, "source_training_required": True,
          "training_recipe_ref": f"configs/training/{m}.yaml", "embedding_dim": None,
          "class_index_map": None, "frozen_bundle_ref": None,
          "implementation_status": "TODO", "training_status": "NOT_RUN"} for m in ["aasist", "ssl_aasist"]}
datasets = {d: {"roles": ROLES[:-1] if d == "asvspoof2019_la" else ["target_test"],
                "raw_contract_status": "UNRESOLVED", "adapter": None, "implementation_status": "TODO"} for d in DATASETS}
methods = {"ep_tta": {"comparison_track": "mechanism", "route": "feature_cache", "parameterization": "matrix", "objective": "view_variance", "regularizer": "margin", "subspace": "response", "reset_policy": "per_sample", "final_output": "original", "projection": "frobenius", "implementation_status": "IMPLEMENTED_UNVERIFIED"}}
for m in ["frozen", "multiview_mean", "static_subspace", "fixed_source_adapter", "frozen_source_shift", "ep_no_keep", "ep_random_U", "ep_feature_pca_U", "ep_no_projection", "entropy_same_adapter_no_keep", "entropy_same_adapter", "memo_same_adapter_no_keep", "memo_same_adapter_keep", "ep_keep_l2", "ep_keep_logit", "ep_keep_fisher", "ep_scalar_adaptive", "ep_diagonal_R", "source_ce_only", "tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full", "eata_audio_ep", "t2a_audio_ep"]:
    methods[m] = {"comparison_track": "published_port" if "audio_ep" in m else "reference" if m in ["frozen", "multiview_mean", "static_subspace", "fixed_source_adapter", "frozen_source_shift"] else "mechanism", "route": "waveform_update" if "audio_ep" in m else "feature_cache", "implementation_status": "TODO"}
for name, values in [("models", models), ("datasets", datasets), ("methods", methods)]:
    catalog = {"registry_version": VERSION, name: values}
    write(f"configs/{name}/registry.yaml", catalog)
    write(f"src/eptta/catalogs/{name}.json", catalog)

write("src/eptta/schemas/source_training_plan.json", obj({"schema_version": V, "status": enum("DRAFT", "APPROVED", "LOCKED"), "execution_environment": enum("remote_only"), "source_dataset_id": enum(*DATASETS), "model_ids": arr(enum(*models), minItems=1, uniqueItems=True), "fit_snapshot_ref": NS, "source_val_snapshot_ref": NS, "split_plan_lock_ref": NS, "preprocess_lock_ref": NS, "automatic_followup_full_training": {"type": "boolean", "enum": [False]}}))
