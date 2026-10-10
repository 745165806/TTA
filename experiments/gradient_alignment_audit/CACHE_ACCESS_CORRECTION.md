# ITW feature-cache access correction — 2026-09-27

The first run, `alignment_dev_20260927a` at diagnostic-code commit `2f4894e`,
used the 31,779-row shared `cache-target-in_the_wild` container. It selected
exactly the fixed 3,178 target10 `sample_index` rows via NumPy memory mapping,
did not read cache ID sidecars, and matched the previous target10 Frozen scores
with maximum difference 0. It did not intentionally index a target90 feature
row, and it did not read target90 labels or metrics.

**Strict protocol status: `NONCOMPLIANT_SHARED_CACHE_ACCESS`.** The worker
opened shared `.npy` chunk files containing target90 rows. Memory mapping and
operating-system page reads do not prove that no target90 feature-cache bytes
were accessed. Therefore the original run does not meet the user's instruction
to use an already isolated selected-only feature cache and never open target90
feature/cache data. Its numerical alignment table is retained as exploratory
development evidence, not presented as a protocol-clean confirmation. This
correction is append-only; it does not change or erase the original files.

A read-only search of 25 feature-cache `index.json` files under the known
`outputs_v2/ssl_aasist` and research-worktree paths found no standalone
3,178-row ITW cache. This is a scoped search, not proof that one does not exist
elsewhere on the workstation. WaveFake's existing 4,096-row cache is selected
only. No waveform encoder was rerun.

The runner has now been tightened to require `--itw-selected-cache` with exact
3,178-ID coverage and matching source identity. It rejects the known shared
target_test cache before opening its index. A protocol-clean rerun requires an
existing standalone target10 feature-cache path; it must use a **new run ID**.
Until then the formal decision is `INCONCLUSIVE` on protocol compliance, while
the exploratory numerical observation remains: O2's head gradient is positive
on both development domains and ENT/PL/O1 have domain-dependent signs.
