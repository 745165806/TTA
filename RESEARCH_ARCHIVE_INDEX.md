# Research archive index (2026-10-10)

This index separates Git history from local run artifacts. An annotated tag
preserves committed source and small tracked evidence; it does **not** preserve
ignored checkpoints, scores, caches, private label sidecars, logs, or a dirty
working tree. No worktree or branch listed below may be removed until those
artifacts have a verified backup and process ownership has been rechecked.

`remote` means the annotated tag was pushed to `origin` and its peeled commit
was verified against the branch tip on 2026-10-10. `local` means the tag has
not been pushed; the branch may contain Git objects that require privacy review
before they can be made public. `pending` means no exit tag yet.

| Historical branch | Original HEAD | Archive tag | Tag state | Distinct research record / local artifacts |
|---|---|---|---|---|
| `codex/sync-necessary-20261009` | `8b9e4b8dd2bbc5d5678443b62be9767beabe0050` | `archive/20261010/codex-sync-necessary-20261009` | remote | Same tree as `origin/main` at `6ba82fb`; `/tmp/tta-sync-20261009` has no known research run. |
| `exp-task-objective-discovery` | `66680217720532388ff241bbb88a8565788cf297` | `archive/20261010/exp-task-objective-discovery` | local | O1/O2/O3 and PA single-class/Codecfake incompatibility negative evidence; root worktree remains dirty relative to this old HEAD. |
| `exp-audio-native-tta` | `cfc2be3c5e4df69ff574c6149377889f883cc4ba` | `archive/20261010/exp-audio-native-tta` | remote | Audio-native target10 baseline results and ignored logs in `/media/dell/data/fakeAudioDection/TTA_audio_native`. |
| `exp-p3.1-stat-asym` | `2f0b82cfefa366f4e411c318f6b8f11b32d1e7a1` | `archive/20261010/exp-p3.1-stat-asym` | remote | Calibrated-teacher sensitivity/asymmetry results and ignored logs in `/media/dell/data/fakeAudioDection/TTA_p3`. Historical SHA-ordered split must not be regenerated with a changed algorithm. |
| `exp-local-distribution-tta` | `a1da16482657e82f9306d5ceb5de43669314a625` | `archive/20261010/exp-local-distribution-tta` | remote | Codecfake compatibility and local-distribution development; ignored `experiments/{codecfake_compat,local_distribution_tta}/results/`. |
| `exp-large-scale-confirmation` | `314ae7e70de8b24026f3b2e31ff5778705cfc342` | `archive/20261010/exp-large-scale-confirmation` | remote | Fixed development assignments, order-specific confirmation and ignored cache/results. |
| `exp-capacity-audit` | `e7185f0676864e6632dc27bd1b9e6eaf35c8b4e5` | `archive/20261010/exp-capacity-audit` | remote | Supervised capacity and WaveFake pairing diagnostics; ignored capacity results and `outputs_v2`. |
| `exp-head-tta` | `8706781d777112743c2a2d755246e88a75e84c85` | `archive/20261010/exp-head-tta` | remote | Head geometry/head adaptation development and ignored checkpoint/score directories. |
| `exp-gradient-alignment-audit` | `0ed4a681ccf26764a083e847d5cc563cbb2e4e6f` | `archive/20261010/exp-gradient-alignment-audit` | remote | Two-domain gradient alignment plus shared-cache access correction; local `outputs_v2` remains. |
| `exp-matched-head-oracle` | `093c9a94ab86cd1766f3b493383e02613d0b0f8f` | `archive/20261010/exp-matched-head-oracle` | remote | Target10-only cache, O2 strength and explicitly supervised oracle; ignored scores and linked capacity results. Worktree folder is `exp-o2-strength-audit`. |
| `exp-meta-rank-head` | `1c7f899d49056a870e88bcabc0d660e380b5586f` | `archive/20261010/exp-meta-rank-head` | remote | Four-seed meta-rank development and ignored model checkpoints/scores. |
| `exp-distribution-conditioned-head` | `fa37e9a14a4f0f23ffb3d294b32c2cd11879052a` | `archive/20261010/exp-distribution-conditioned-head` | remote | Four-seed distribution-conditioned head and ignored checkpoints/scores. |
| `exp-overnight-tta` | `f58f3c268b3804f7265f2f8fb43b3cf95f4f6e86` | `archive/20261010/exp-overnight-tta` | remote | Residual/T3A development, static-versus-adaptation controls, ignored checkpoints/scores. |
| `exp-representation-tta` | `29f98ac16d2b4594a0953016712906eaa22ed01f` | `archive/20261010/exp-representation-tta` | remote | Earlier-layer SSL-AASIST negative result, failed first evaluation, large LL cache and checkpoint artifacts. |
| `codex/online-add-baselines-v2` | `a1bedd41cee39621e96ba62d551ae99c094fc671` | `archive/20261010/codex-online-add-baselines-v2` | remote | 180/180 online development streams, scores/logs, ignored private evaluator labels and LL cache. Continual/batch baseline, separate from episodic EP-TTA. |
| `exp/meta-audio-tta` | `98df7f255e956f98a33da9dbf575edfbeb7e9303` | `archive/20261010/exp-meta-audio-tta` | local | Project-trained source/meta checkpoints, source selection/cal0, disclosed-access four-domain development evaluation; ignored epoch checkpoints remain. |
| `codex/meta-audio-tta-v2-mechanism` | `2e639fc3f8ae8d5fcd98026f4fe1eba09a474371` | `archive/20261010/codex-meta-audio-tta-v2-mechanism` | local | V1 failure mechanism, source-only matched-gradient diagnostics and tracked analysis. |
| `codex/ep-capacity-geometry-20261009` | `e3ce4c229591549cd5ce962471d31886adbeb052` | pending | Active/recent geometry line; `local/` results must be inventoried after work has stopped. This HEAD is a snapshot and may advance. |

The historical `origin/main` merge already contains P0/P1, P2, protocol/oracle,
guard, task-objective negative evidence and the WaveFake/DFADD embedded-audio
import bridge. Tagged experiments retain the numerical and access-boundary
records that are not promoted to a default method.

## Removal gate

Before removing any worktree: record its current branch HEAD and Git status;
identify all ignored/untracked artifacts; back them up to a separate location;
verify per-file count, byte size and checksum against the source; confirm no
process or Codex task uses the directory; verify the remote archive tag's
peeled commit equals the original HEAD. A remote tag alone never satisfies the
artifact-backup gate. No historical worktree had been removed when this index
was first written.
