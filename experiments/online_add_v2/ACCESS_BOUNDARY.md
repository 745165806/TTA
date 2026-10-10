# Online ADD v2 file and information boundary

This record describes this benchmark run (`online_add_v2_20260928a`) and its own programs. It does not make claims about unrelated historical experiments on the workstation.

| Phase | Files actually read | Information reaching an online method |
|---|---|---|
| LA/DF subset builder | Full official LA/DF CM key text and existing source inference manifests | None; the builder writes 4,096 selected IDs per domain with labels separated into evaluator-only sidecars. Reading the full key for selection is disclosed. |
| ITW/WaveFake evaluator-label builder | Existing ITW target10 audit labels; WaveFake selection plus `audio_id`/`real_or_fake` parquet columns for referenced files | None; only 3,178/4,096 selected labels are stored in evaluator sidecars. |
| LL extraction | Only selected LA/DF runner manifests and their 8,192 raw audio records; existing selected ITW/WaveFake three-view LL caches are reused read-only | Frozen LL tensors `[3,201,128]` are stored by opaque ID. No global target statistic, clustering or target label is computed. |
| Formal runner | Label-free stream manifests, selected LL chunks, source-trained bundle and fixed source-only summaries | Current B16 LL batch, its own earlier state, and for RoTTA the three LL views of those B16 records. Neither label, attack, group, path nor future LL batch enters `OnlineMethod`. |
| Source setup and calibration | Source fit/select IDs, source labels and selected source LL cache | Source-only Fisher, diversity setting, selected learning rates and, for the v2.1 supplement, score thresholds. Fresh method instances are used for source calibration. |
| Evaluator | Completed score files, locked stream manifests and four evaluator-only label sidecars | No feedback into formal method states or hyperparameters. Interim target metrics were produced during the running campaign for engineering checks; adaptation configuration was not changed in response. |

Within this run, `target90` and the separately designated final holdout were not read or scored. The unselected LA/DF audio was not extracted. The ASV official key files were read by the builder to select the development cohorts; the selected LA/DF records originate from official eval releases and are not claimed as untouched final holdouts.

The v2.1 execution supplement was supplied after order2026 and part of the later orders had already generated target metrics. Its method-specific source-select threshold requirement therefore cannot be represented as chronologically pre-target in this run. The stored first online scores and AUC/EER are unaffected by threshold selection. The additional calibration runs use only source-select and are reported as a disclosed evaluator correction, with the earlier shared-threshold metrics retained as historical files.
