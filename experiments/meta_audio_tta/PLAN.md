# SSL-AASIST MABN-inspired episodic audio TTA

## Scientific identity and sources

This is a single-original-audio episodic transfer of MABN mechanisms, not a
reproduction of MABN's target-domain few-shot protocol. MABN's BN affine,
self-supervised auxiliary branch, joint training and bi-level objective are from
[Wu et al., AAAI 2024, Eq. 2–4 and Algorithm 1](https://ojs.aaai.org/index.php/AAAI/article/download/29527/30876).
The single-item episode and two-view support are protocol mappings informed by
[MT3, AISTATS 2022](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf)
and [TTT, ICML 2020](https://proceedings.mlr.press/v119/sun20b.html).
The projector, predictor, stop-gradient target and EMA follow
[BYOL, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf).
MT3's meta-model target without a separate EMA is a different variant and is
not silently substituted. The MABN paper's linked author repository was not
readable during planning; paper-level evidence is distinguished from code audit.
The [MT3 training implementation](https://raw.githubusercontent.com/AlexanderBartler/MT3/main/model/train_meta_obj.py)
copies a meta-model into differentiable inner models, computes BYOL-style
inner gradients, and differentiates the outer CE+BYOL objective through those
updates. Its [test implementation](https://raw.githubusercontent.com/AlexanderBartler/MT3/main/model/test_meta.py)
restores online weights per image and uses zero-momentum SGD. Those graph and
reset semantics support this port; the explicit EMA target and restricted BN
scope here remain declared MABN/BYOL-inspired audio choices.

## Fixed information and comparison contract

- Reuse existing ASVspoof2019 LA assignments. `fit` alone supplies training and
  meta-training gradients; `source_val` selects epochs, `select` chooses method
  hyperparameters, and `cal0` sets thresholds. Do not regenerate assignments.
- The four selected development subsets are ITW target10, WaveFake, LA21 and
  DF21. Target10 feedback affecting a subsequent version or a stop decision is
  disclosed chronologically. Target90 and final holdout are prohibited.
- CE and CE+BYOL start from identical generic XLS-R initialization and receive
  matched `fit` IDs, augmentations, supervised backbone updates and budget.
  CE-only has a Frozen baseline; it has no BYOL TTA without a trained head.
- Every TTA gain is K=1 versus K=0 from the **same concrete checkpoint and
  waveform path**. Old epoch 7 and Online ADD v2 are historical context only.
- Report EER and AUC separately, fixed source-threshold errors, per-item
  correct-to-wrong and wrong-to-correct flips, fake-real pair ranking inversions,
  update magnitude, wall time and peak memory. Preserve every positive and
  negative result.

## Implementation stages and state semantics

0. Audit the real SSL-AASIST parameter graph; implement BYOL, functional
   second-order fast weights and single-item runner. Test real-model K=0 parity,
   BN source-buffer stability, A→B→A reset and nonzero backend-BN impact on
   spoof score. Measure real one-step memory/time before full training. Commit
   the plan, code, exact commands, exit codes and smoke report independently.
1. Train matching CE and CE+BYOL source runs from the same initialization;
   `source_val` chooses immutable epoch files. Online branch and target BYOL
   branch are separate author-model constructions. Joint training updates EMA
   only after an online optimizer step, copies source BN buffers, and never
   differentiates the target. New run paths are exclusive. Commit separately.
2. From a valid joint checkpoint, compare Cross-sample Meta (support A,
   independent query B) and Same-sample Meta (support A, labeled query A).
   Both use one original audio with two deterministic views for the inner BYOL
   update, the same backbone/auxiliary architecture/BN scope/task schedule and
   matching optimizer steps. Report different unique-audio exposure explicitly.
   The query's canonical label is source-only. Outer gradients differentiate
   through the BN fast update. The target branch and its BN statistics are fixed
   during each inner task; EMA moves only after an outer optimizer step.
   A fixed balanced source_val subset reports the same-item K=0/K=1 EER and
   number of changed detection scores as a separate diagnostic. Stage 2
   requires the selected joint checkpoint to pass a real-model source gate.
3. Select method parameters using source `select`, thresholds using `cal0`, then
   write K=0/K=1 scores for each fixed development subset before reading labels
   in the independent evaluator. Each item restores source checkpoint parameters,
   buffers, stateless SGD inner optimizer and deterministic item RNG. The two
   views are generated once and EMA never advances at test time. Commit each
   stage with run/config/epoch paths, commands, logs and result report.

All stage 0 checks must use the actual author SSL-AASIST; toy unit tests are
supplementary. A runtime/contract failure stops the current stage for repair.
Absent credible benefit, stop extending that version's seeds or target scoring;
this does not establish that meta-adaptation is generally ineffective. Unrun
work is `NOT_RUN`.

## Stage 0 real-model finding

The pinned author `Residual_block.forward` computes `bn1(x)` but then replaces
the result with `conv1(x)`. In the existing epoch-7 model, the five
`encoder.{1..5}.0.bn1` modules (10 affine tensors) receive no BYOL gradient.
The stage 0 fast-weight scope therefore excludes them; it does not patch the
author architecture. The 30 active backend BN affine tensors contain 1,538
scalars, plus four affine tensors in the auxiliary projector/predictor. This
scope must be re-audited on each newly trained source checkpoint.
