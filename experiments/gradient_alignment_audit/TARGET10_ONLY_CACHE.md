# ITW target10-only feature artifact — 2026-09-27

Status: **PASS for application-level selected-row feature access**. The new
`target10_only_cache/` is an independently materialized, read-only-after-write
3,178-row `FeatureCache` and is excluded from Git. The fixed existing target10
manifest was reused without reselection or label access.

| Check | Observed |
|---|---:|
| Selected samples / unique IDs | 3,178 / 3,178 |
| Per-sample shape / dtype | 3×160 / float32 |
| Finite values | all |
| Maximum absolute difference from historical ID-based selected rows | 0 |
| Source shared-cache row count | 31,779 |
| Source chunk files memory-mapped | 125 |
| Source ID metadata rows scanned | 31,779 |
| Nonselected feature rows indexed or copied by project code | 0 |
| Target90 labels, scores, metrics read | NO |
| Waveform encoder rerun | NO |

The builder copied only fixed `sample_index` row slices from NumPy memory maps.
An independent selected-ID lookup from the same historical cache reproduced
all 3,178 tensors exactly; the derived `FeatureCache` was reopened and checked
for exact ID and value coverage. The source cache ID, source run, checkpoint,
manifest, selected-row policy and derived cache ID are recorded in
`target10_only_cache_provenance.json`. The extracted tensors are in fixed
manifest order. No target90 feature value was intentionally indexed or copied.

The historical shared chunk files were necessarily **opened** during this
one-time extraction. NumPy memory mapping does not certify physical page-byte
isolation; this report does not claim that. Subsequent gradient and O2 runs
read only the independent 3,178-row artifact, not the shared cache. This is the
strictly available guarantee under the user's selective-extraction allowance.

No SHA/checksum was computed: automatic approval review rejected hash code
because the repository `AGENTS.md` prohibits project business-code digests.
The source IDs and exact tensor parity above supply provenance and validation
without a hash gate. The rejected hash addition was removed before extraction.
