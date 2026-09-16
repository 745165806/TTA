# Ticket 1 approved recovery execution record

- Approval plan: `ticket1-recovery-20260916`
- Bound plan SHA-256: `07e5318c87f286f71ec6df4e484431bc317c2ec6ba7b62df7ed01231b01785a6`
- Approval received: 2026-09-16 (Asia/Shanghai)
- Validation GPU cap: **1.0 aggregate GPU hour**
- Formal AASIST continuation cap after validation PASS: **7.3 GPU hours**

## Outcome

The route-B implementation and four initial isolated validation jobs were prepared
and checked without starting CUDA training.  The complete approved validation is
`2 + 1 + 1 = 4` full fit epochs per model, or `2,116` optimizer-update executions
per model (`4,232` across AASIST and SSL-AASIST).  The bound plan estimates this
complete two-model protocol at 1.5 aggregate GPU hours.  That exceeds the newly
authorized hard cap of 1.0 GPU hour.

Consequently:

- isolated GPU validation: `NOT_RUN_BUDGET_INSUFFICIENT`;
- aggregate GPU time consumed by this execution: `0`;
- AASIST formal continuation: `NOT_RUN_VALIDATION_NOT_PASSED`;
- SSL-AASIST administrative child finalization: `NOT_RUN_VALIDATION_NOT_PASSED`;
- target snapshot publication, target scoring and R5--R9: `NOT_AUTHORIZED`.

Running only a prefix, fewer epochs, fewer batches, or a data subset would not
satisfy the plan's declared PASS rule and therefore was not substituted silently.

## Prepared isolated jobs

All jobs have `exact_resume_claim=false`, bind the approved plan/checkpoint bytes,
use a new child recipe identity, and preserve the parent's 100-epoch scheduler
horizon.  They are under the ignored private-artifact tree:

- `artifacts/private/ticket1-recovery-20260916/validation/aasist-continuous-v2`
- `artifacts/private/ticket1-recovery-20260916/validation/aasist-restart-stage1-v2`
- `artifacts/private/ticket1-recovery-20260916/validation/ssl-continuous-v2`
- `artifacts/private/ticket1-recovery-20260916/validation/ssl-restart-stage1-v2`

The restart stage-2 jobs are intentionally not created until stage-1 immutable
checkpoints exist.  No parent file was modified.
The four same-named directories without the `-v2` suffix bind an earlier worker
hash and are retained as superseded preparation evidence; strict orchestration
verification rejects them after the final resume-hardening edit.
