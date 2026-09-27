# Large-scale development confirmation contract (fixed before scores)

This is a confirmation of four existing methods, not method development. Frozen, production `ep_no_keep` per sample, Local-Base B32, and Local-O1 B32 use the same Frozen SSL-AASIST bundle, three safe views, original-view score, source resources, 8×8 R, projected SGD, K=5, lr=0.03, rho=0.1, gamma=0.1, lambda_keep=0. Every B32 buffer starts with R=0 and no optimizer state. The final short buffer is retained. No guard, preservation, gate, memory or new objective is added.

## Immutable development assignments

- ITW: all 3,178 IDs from the existing target10 `inwild_target10_select.json`, unchanged. Target90 is outside this study.
- Codecfake: 5,000 official-dev available WAV IDs. Include the existing fixed 512 mechanism IDs, then choose 4,488 additional IDs with `random.Random(2026).sample` over the sorted remaining available dev WAV IDs. Sort the final 5,000 IDs for `manifest_order`. Preserve the old 512 `sample_index` values 0–511 to keep their view seeds identical; assign new IDs indices 512–4999 in sorted new-ID order. This inclusion makes the large set a nested confirmation; report old-512 versus additional-4,488 effects post hoc, but never alter membership. Native 16 kHz uses the unchanged production loader; other rates use the existing SciPy 1.13.0 `resample_poly` contract.
- ASVspoof2019 LA: choose 5,000 IDs using `random.Random(2026).sample` over sorted, available `ASVspoof2019_LA_dev/flac/*.flac` basenames. The official dev label protocol is closed until complete score validation. No ASVspoof2021 pool is used.
- WaveFake: only reader/resource audit in this stage; it cannot delay these three domains.

No label, Frozen score, or prediction enters selection, order generation, compatibility processing, feature extraction, or adaptation. Assignment manifests are created exactly once and never redrawn for class composition or observed effects. Selected-only audit labels may be opened only after all required scores for every order and domain have exact coverage and finite values. A single-class selected domain is marked non-ranking; no replacement selection.

## Orders and shared stochastic state

`manifest_order`, followed by seeds 2026, 2027, 2028, 2029, 2030. Each shuffled order uses `random.Random(seed).shuffle` on only the fixed label-free IDs. All four methods use the identical order within each domain/seed. Frozen and per-sample scores must be exactly invariant to order for each ID. Local buffers are formed independently per domain and order, B=32. Feature views are fixed by the selected manifest's `sample_index`, not recomputed for each order.

## Analysis and preregistered decision

Per domain/order/method: EER, AUC, balanced accuracy, FPR, FNR, mean absolute score delta, mean R norm, source evidence damage, numeric failures and runtime. Compute 1,000 paired stratified bootstrap draws with seed 2026 for ΔAUC and ΔEER versus Frozen and local-versus-per-sample comparisons. These are development uncertainty diagnostics, not a final statistical claim. Report six-order mean, sample std (`ddof=1`), min, max, mean and std of paired ΔAUC, and `abs(mean_delta_auc)/(std_delta_auc+1e-12)` as descriptive effect-to-order variability. Report absolute effect sizes even when intervals exclude zero.

A mechanism is promoted **only if all** hold for the same local arm on at least two independent two-class domains:

1. Mean ΔAUC versus Frozen is at least `+0.005` absolute in each domain, and at least five of six orders have positive ΔAUC in each. `0.005` is a prespecified practical effect floor; a stable `+0.0002` remains negligible.
2. `effect_to_order_variability >= 2` in each domain. This means mean effect is at least twice the six-order sample std. No post-hoc threshold change.
3. The paired 95% development bootstrap ΔAUC lower bound exceeds zero in at least four of six orders in each domain.
4. Mean ΔEER versus Frozen is no worse than `+0.005` absolute in each domain, with no single order worse than `+0.010` absolute. EER improvement is not required.
5. Post-hoc class, attack and sample-rate composition is reported where metadata legitimately exist. A result restricted to one composition is described as domain-specific; no data reassembly follows.

`STRONG_REPLICATED_EFFECT`: all conditions pass. `DOMAIN_SPECIFIC_EFFECT`: a local arm passes the practical/stability checks in one domain but not two. `SMALL_DEVELOPMENT_ARTIFACT`: the Codecfake large-set mean ΔAUC is below `+0.005` or changes sign, and no two-domain confirmation exists. Otherwise `INCONCLUSIVE`. These labels do not authorize method implementation in this stage. No final or target90 data are opened.
