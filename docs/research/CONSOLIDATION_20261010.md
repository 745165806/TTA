# Research consolidation record — 2026-10-10

This document records the integration branch `integration/consolidate-main-20261010`
and the scientific boundaries of the historical studies. The integration
starts from `origin/main` at `6ba82fbe677281c40cf158bfd054f4bcb8ef9548`.
It does not change the production EP-TTA registry, the frozen source model,
target10/target90 assignments, source thresholds, or existing evaluator.
See the root `RESEARCH_ARCHIVE_INDEX.md` for each original branch HEAD and tag.

## Code integration map

| Source tip | Imported material | Main default behavior | Validation at first import |
|---|---|---|---|
| `exp-matched-head-oracle` `093c9a94ab86cd1766f3b493383e02613d0b0f8f` (includes `exp-local-distribution-tta`, `exp-large-scale-confirmation`, `exp-capacity-audit`, `exp-head-tta`, `exp-gradient-alignment-audit`) | Experiment-local Codecfake compatibility, local-distribution, large confirmation, capacity, head, gradient, O2-strength and supervised-oracle scripts/contracts; ten unit tests; three **label-free** fixed confirmation manifests. No checkpoint, private label or run output copied. | None; scripts remain under `experiments/`. The supervised oracle is a development diagnostic, never an unsupervised TTA method. | 28/28 selected tests passed after adding the required label-free manifests. Full dataset/cache execution NOT_RUN. |
| `codex/online-add-baselines-v2` `a1bedd41cee39621e96ba62d551ae99c094fc671` (includes meta-rank, distribution-conditioned head, overnight and representation research) | Experiment-local online baseline implementations/evaluators and the earlier-layer/alternative-head research scripts and reports. Locked streams, evaluator labels, result trees, checkpoints and large LL caches remain at their historical locations. | None. Online predict-then-adapt B16, memory methods and offline batch adaptation are separate protocols from per-sample episodic EP-TTA. | New CPU contract checks for native spoof-minus-bonafide score and independent stream counters: 2/2 PASS. Full stream evaluation NOT_RUN. |
| `codex/meta-audio-tta-v2-mechanism` `2e639fc3f8ae8d5fcd98026f4fe1eba09a474371` (includes `exp/meta-audio-tta` `98df7f255e956f98a33da9dbf575edfbeb7e9303`) | Source training/meta-training core, bounded source diagnostic scripts, selected source-only configs and reports. Shared source checkpoint/paired-flip helpers were extracted from historical Stage 3 code to avoid importing its content-checksum seal. Historical Stage 1/3 configs, target-label config and run trees remain only in the archive. | None. No new source or target run is launched by importing this code. | 10/10 source/meta unit tests PASS. Stage 3 seal/evaluator tests NOT_RUN because that historical checksum gate was not imported. |
| `exp-p3.1-stat-asym` `2f0b82cfefa366f4e411c318f6b8f11b32d1e7a1` | No executable code imported yet. Its archived fixed calU/evalU split is defined by SHA-ordering, which conflicts with the current prohibition on content-digest-based project workflows; changing the algorithm would change historical membership. | None. | Import NOT_RUN; preserved by remote archive tag. |
| `codex/ep-capacity-geometry-20261009` `e3ce4c229591549cd5ce962471d31886adbeb052` | Source-only task/mixed subspace construction, unlabeled fixed-512 geometry audit and SVG renderer, small historical summary/figures, full historical report and two source-subspace tests. Ignored per-sample `local/` products remain in the original worktree. | None; no geometry candidate is registered as a default method. | 3/3 source-subspace and analytic score-bound tests PASS; geometry modules import. The historical 512-ID numerical run was not repeated. |

`codex/sync-necessary-20261009` at `8b9e4b8dd2bbc5d5678443b62be9767beabe0050`
was already merged into `origin/main`; its tree equals that main snapshot. The
WaveFake/DFADD embedded import and O1/O2/O3 task-objective records in the
dirty root worktree match main files, so they were not duplicated here.

## Scientific reading of the archived results

- O1/O2/O3 on fixed In-the-Wild mechanism development made only small AUC
  movements with intervals spanning zero and worse EER. The auxiliary PA
  selected group was single-class, so it cannot establish two-domain ranking
  correction. The Codecfake first fixed selection failed full production
  extraction because its rates were mixed. Preserve all three negative facts;
  do not reselect groups after observing outcomes.
- EPDC paired-order and normalized-margin prototypes did not yield useful
  target ranking correction at their tested settings. Guard/oracle/capacity
  studies constrain those *specific* candidate spaces; they do not justify a
  universal impossibility claim. Matched supervised oracle results are
  diagnostic upper bounds, not label-free methods.
- Head, distribution-conditioned, meta-rank, overnight and earlier-layer
  studies have different source training and adaptation protocols. Any
  improvement over another source model must be separated from an ON-versus-OFF
  TTA effect. The earlier-layer seed-13 ON-versus-OFF development comparison
  was negative; unrun seeds remain NOT_RUN.
- Online ADD v2 completed 180 development streams and found no mean stationary
  or dynamic segment-macro AUC gain over Frozen among its tested methods. It
  uses B16 online and memory protocols and cannot be presented as an episodic
  EP-TTA result or a final-holdout claim.
- The meta-audio Stage 3 four-domain evaluation is explicitly
  **disclosed-access target development**. Its within-checkpoint K=1 changes
  were small and not consistently beneficial across domains, despite extra
  computation. The subsequent source-only mechanism subset found negative
  BYOL/CE gradient alignment on several selected items and counterexamples on
  others; it does not prove all BYOL adaptation fails.
- The fixed 512-ID unlabeled EP capacity audit found that the original U has
  nonzero score capacity, while the hard guard greatly reduces actual score
  movement. Its initial view-loss gradient has no stable signed score
  direction. These are geometric and mechanistic observations; the audit did
  not open target labels or establish an EER/AUC benefit. The historical
  numbers, failed preliminary runs and local artifact paths are recorded in
  `EP_CAPACITY_AUDIT.md`; only the small summary and figures were imported.
- The earlier ASV evaluation-label-file scan boundary correction in
  `experiments/multidomain_mechanism/AUDIT_BOUNDARY_CORRECTION.md` remains in
  force. Historical `final_holdout_labels_accessed=false` fields must be read
  with that correction. Do not describe a development subset from an official
  eval release as an untouched final holdout.

## Verification and remaining gates

All listed PASS values came from commands run in the `tta` conda environment
on this integration worktree. They are engineering checks, not new scientific
outcomes. The broader `tests/unit tests/contracts` run finished with **276
passed, 1 skipped**. The skip is the existing target90-versus-target10 check:
its ignored local `target_test.jsonl` is absent in this clean worktree, so the
real-data assertion remains **NOT_RUN**, rather than a fabricated pass. The
`tests/integration` run finished with **4 passed, 1 skipped**; real SSL-AASIST
waveform parity is **NOT_RUN_RESOURCE** because the model/GPU inputs are not
available here. The passing synthetic suite covers core model interfaces,
Frozen/cached scoring, per-item reset, score direction, parser contracts and
role/group validation, but does not replace a real-model parity run. The
`eptta.cli --help` entrypoint smoke exited successfully. Historical Meta Audio
Stage 3 documents contain reproduction commands for scripts and configs kept
at their archive tag; those commands are not runnable from this integration
tree. Final PR review remains pending. No full training,
full cache materialization, new target scoring or new final-holdout access is
part of the consolidation.

The synchronized `/tmp/tta-sync-20261009` worktree and its local/remote branch
were removed after the remote tag, clean-status, ignored-file and process
checks passed. Fifteen annotated archive tags were pushed and independently
verified against their original commit tips. Two further tags are local-only
pending privacy review of their Git objects.
Ignored and untracked artifacts still require an independently located backup
with file count, byte size and checksum verification before any worktree or
branch deletion. Until then, branch/archive status is **retained**.
