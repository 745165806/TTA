# Stage-3 development-label access incident (2026-10-08)

Status: **stage-3 target evaluation NOT_RUN; strict evaluator-only access
condition violated during code preparation**. Stage-1 source training and
stage-2 source meta work do not use these target labels and may continue.

Two schema-inspection commands were run before any new target scores existed:

1. Python `json.load` parsed the existing
   `experiments/target10_selection/manifests/inwild_target10.json` to print
   top-level keys, record keys and count. This JSON has a `label` field in its
   records, so the command loaded all 3,178 target10 label values into the
   Python process even though it printed no values.
2. Python `json.loads` parsed the first JSONL row of each Online ADD v2
   evaluator sidecar for ITW, WaveFake, LA21 and DF21 to print field names.
   One label value per sidecar was loaded, but no value was printed.

No target label was used for gradient updates, method choice, threshold,
metric computation, or a continue/stop performance decision. No target90 or
final-holdout file content was opened. These facts limit the contamination;
they do not restore the requested evaluator-only rule. The later scoring
config is label-free and the separate evaluator config is read only after a
complete score file, with a unit test for that boundary.

The fixed target10/WaveFake/LA21/DF21 score and metric commands remain
**NOT_RUN**. Any later use of these four development sets for this version
must cite this incident and the already-opened target development status.
No result may be described as satisfying a never-opened-label protocol.
