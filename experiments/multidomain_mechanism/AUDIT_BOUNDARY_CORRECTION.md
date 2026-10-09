# Selected-only audit correction — 2026-09-27

The first post-score audit scanned full ASV official evaluation label files with `rg` to emit only selected mechanism records. This scanned final-holdout label-file bytes and violated the strict access boundary, even though no final-holdout row was returned or parsed. See the append-only correction in the successful run's `analysis/boundary_correction.md`.

`materialize_selected_audit.py` now reads only the completed guard run's post-score `sample_mechanisms.csv` and writes exact selected-ID labels into local, ignored `experiments/multidomain_mechanism/audit_labels/`. The analysis reader now requires that selected-only artifact; it never opens the official ASV label files. Its exact ID and canonical 0/1 checks remain active. The fixed group assignment is unchanged. No new labels or scores were derived from final holdout after the correction.
