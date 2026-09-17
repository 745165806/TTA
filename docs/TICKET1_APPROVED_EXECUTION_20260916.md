# Ticket 1 approved recovery execution record

- Approval plan: `ticket1-recovery-20260916`
- Bound plan SHA-256: `07e5318c87f286f71ec6df4e484431bc317c2ec6ba7b62df7ed01231b01785a6`
- Validation cap, amended by user: **1.5 aggregate GPU hours**
- Formal AASIST continuation cap: **7.3 GPU hours**
- Target snapshot publication, target scoring and R5--R9: **not authorized**

## Isolated recovery validation

Both models completed the declared `2 + 1 + 1` full-epoch protocol. The six
successful branches consumed 5,031.7835 GPU-seconds (1.39772 GPU-hours). Including
the conservative wall time of dependency/RNG-format startup failures remains below
1.412 GPU-hours and below the 1.5-hour hard cap.

The original py310/PyTorch 2.1 runtime could not restore the parent's 816-byte CUDA
RNG state (the runtime expected 16 bytes). Existing py38/PyTorch 2.0.1+cu118 was
verified to accept both parents' RNG states. A missing SSL `bitarray` dependency in
base py38 was resolved by using the existing, unmodified `eptta` py38 environment;
no package was installed.

Strict result: **FAIL for both models**. Continuous and restarted branches agree on
epoch/global-step, scheduler/LR, scaler, final RNG, sampler and lineage, but model
parameters/buffers, Adam state and loss/validation records diverge. AASIST already
diverged between the two independent epoch14 executions, before the tested restart
boundary. The runs are therefore not claimed as deterministic/exact resumes.

- AASIST comparison: `docs/test_logs/20260916-ticket1-approved/aasist-resume-comparison.json`
- SSL-AASIST comparison: `docs/test_logs/20260916-ticket1-approved/ssl-resume-comparison.json`
- SSL administrative finalization: `NOT_RUN_VALIDATION_NOT_PASSED`

All failed/superseded v2--v4 preparation directories and logs remain retained as
evidence. No parent checkpoint or history file was modified.

## AASIST explicitly authorized non-exact continuation

After receiving the strict FAIL result, the user explicitly directed completion of
AASIST training and required every epoch's model parameters to be retained. The
supplemental authorization is recorded in
`docs/TICKET1_AASIST_NONEXACT_AUTHORIZATION_20260916.json` and does not change the
bound recovery plan or assert exact-resume status.

- Parent: epoch13 / global step 7,406 / last SHA-256
  `d712df099c19e3d0d191065eb7cdbfb84ddae45cd17e84725e8334c7e39c3b19`
- Child run: `/media/dell/data/fakedata/eptta_work/training_runs/aasist_source/child-plan07e5318c-nonexact-001`
- Child identity: `source-run-91cd36fae7770aca4b51`
- Executed: epochs 14--79, 66/66 complete; final global step 42,320
- Scheduler: parent 100-epoch cosine horizon retained
- Formal GPU time: 25,625.0933 seconds = **7.11808 GPU-hours**
- Checkpoints: all `epoch-0014.pt` through `epoch-0079.pt` and sidecars present;
  none missing, plus `last.pt` and `best.pt` aliases
- Final status: `TRAINED` then `FINALIZED`
- Selection: child epoch69, source-val EER `0.00040257648953301306`
- Selected checkpoint SHA-256:
  `076ca355cec3358c7181769459de27279609ab1cda35f186670bc49e9a1cbf0a`
- Finalized id: `training-final-a095c4459547c47c9481`

No frozen export, target effect access, target scoring, new target snapshot, or
R5--R9 execution was performed.
