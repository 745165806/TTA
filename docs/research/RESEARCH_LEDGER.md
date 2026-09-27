# Research ledger

Append a new dated entry for each formal experiment. Do not edit previous entries; append a correction that names the affected entry. `PASS` requires a real command, exit code, and log. `NOT_RUN` is not a scientific result.

Required fields: date, branch, commit, experiment_id, purpose, datasets, manifests, method, parameters, command, result_directory, key_metrics, interpretation, next_decision.

## Historical evidence imported 2026-09-27

- date: 2026-09-27
- branch: exp-oracle-diagnosis (source commit)
- commit: 6f99d3092e186f52f6fffc2393cee496aa57de79
- experiment_id: protocol_tta/run_20260927_104009_W78MLZ
- purpose: isolate episodic, periodic reset, and continual protocol behavior
- datasets: In-the-Wild target10, 3,178 samples
- manifests: `experiments/target10_selection/manifests/inwild_target10_select.json`
- method: TENT audio-native v1, four protocols, Frozen reference
- parameters: Adam, lr=0.001, steps=1, seed=2026, source BN statistics, source tau0
- command: historical runner `experiments/protocol_tta/run_protocol_tta.sh`; exact invocation in its run config/logs
- result_directory: `experiments/protocol_tta/results/run_20260927_104009_W78MLZ/`
- key_metrics: Frozen EER 0.0985707245; episodic 0.0983463882; reset32 0.1074420897; reset128 0.1070496084; continual 0.1679721497, AUC 0.8998983838
- interpretation: single-order continual drift is substantial; entropy decline does not establish task improvement
- next_decision: isolate correction capacity and evidence damage before proposing accumulation

## Historical oracle imported 2026-09-27

- date: 2026-09-27
- branch: exp-oracle-diagnosis (source commit)
- commit: 6f99d3092e186f52f6fffc2393cee496aa57de79
- experiment_id: oracle_diagnosis/oracle_20260927_110841_839245
- purpose: quantify best available guarded EP correction and selector regret
- datasets: In-the-Wild target10, 3,178 samples
- manifests: existing target10 select and audit manifests
- method: production guarded episodic EP, 113 candidates including Frozen
- parameters: K=[0,1,3,5,10], lr=[0.0003,0.001,0.003,0.01,0.03,0.1,0.3], rho=[0.05,0.1,0.2,0.4], gamma=0.1, lambda_keep=1, seed=2026
- command: historical runner `experiments/oracle_diagnosis/run_oracle_diagnosis.sh`; exact invocation in its run config/logs
- result_directory: `experiments/oracle_diagnosis/results/oracle_20260927_110841_839245/`
- key_metrics: Frozen EER 0.0985707245; oracle and five-fold oracle EER 0.0983463882; 0.02243363 pp gain; selector regret 0
- interpretation: limited capacity in this guarded parameter space; no basis to generalize to all TTA
- next_decision: vary guard only on a fixed multi-domain mechanism sandbox

## Guard analysis failure 2026-09-27

- date: 2026-09-27
- branch: exp-multidomain-mechanism
- commit: f7c17b8 (score code); analysis code was not yet committed
- experiment_id: multidomain_mechanism/guard_20260927T040400Z
- purpose: first post-score guard-capacity audit
- datasets: In-the-Wild target10 mechanism 512; ASV2021 LA 282; ASV2021 DF 370
- manifests: fixed `experiments/multidomain_mechanism/manifests/*_mechanism_select.json`; audit sources opened only after score coverage
- method: Frozen and hard/relaxed/unguarded EP at A/B/C
- parameters: A=(5,0.01,0.05); B=(5,0.03,0.1); C=(10,0.3,0.2); gamma=0.1; lambda_keep=1
- command: `PYTHONPATH=src:. conda run -n tta python experiments/multidomain_mechanism/guard_worker.py` then `.../guard_analysis.py --run experiments/multidomain_mechanism/results/guard_20260927T040400Z`
- result_directory: `experiments/multidomain_mechanism/results/guard_20260927T040400Z/`
- key_metrics: NOT_RUN as a complete analysis; score generation and exact coverage PASS, post-score plot generation exit 1 because Matplotlib could not import `pyparsing`
- interpretation: environment plotting failure; partial CSV, `failure.json`, and log preserved; no scientific claim from this failed run
- next_decision: use installed Pillow for standalone plots and a new run_id

## Guard capacity partial three-domain mechanism result 2026-09-27

- date: 2026-09-27
- branch: exp-multidomain-mechanism
- commit: 6614bf7 (run configuration records full hash)
- experiment_id: multidomain_mechanism/guard_20260927T040841Z
- purpose: test whether margin guard severity alone explains weak correction, and whether unguarded updates damage source evidence
- datasets: In-the-Wild target10 mechanism 512 (310 bonafide, 202 spoof); ASV2021 LA 282 (bonafide only); ASV2021 DF 370 (bonafide only); Codecfake/WaveFake NOT_RUN
- manifests: fixed select and audit manifests under `experiments/multidomain_mechanism/manifests/`; ASV final-holdout labels excluded
- method: Frozen and hard/relaxed/unguarded EP at A/B/C; every score generated before selected audit labels were read
- parameters: A=(K5,lr0.01,rho0.05); B=(K5,lr0.03,rho0.1); C=(K10,lr0.3,rho0.2); gamma=0.1; lambda_keep=1; CPU production feature path
- command: `PYTHONPATH=src:. conda run -n tta python experiments/multidomain_mechanism/guard_worker.py` then `PYTHONPATH=src:. conda run -n tta python experiments/multidomain_mechanism/guard_analysis.py --run experiments/multidomain_mechanism/results/guard_20260927T040841Z`
- result_directory: `experiments/multidomain_mechanism/results/guard_20260927T040841Z/`
- key_metrics: In-the-Wild Frozen EER/AUC 0.099010/0.957841; A unguarded 0.103226/0.958623; B unguarded 0.103960/0.958879; C unguarded 0.103960/0.947653. C hard 0.099010/0.955589. C unguarded evidence damage: In-the-Wild 0.172484, ASV LA 0.130316, ASV DF 0.586509. ASV EER/AUC undefined due single-class assignment.
- interpretation: relaxation gives larger changes but no In-the-Wild EER gain; stronger unguarded update loses ranking and raises evidence damage. Small A/B AUC gains coexist with worse EER. Single-class ASV groups prevent a multi-domain ranking conclusion. Ranking "macro" in this run equals In-the-Wild only; see `analysis/coverage_correction.md`.
- next_decision: test an explicit source decision-order preservation loss at fixed moderate strength before fitting a reliability gate; keep ASV signals descriptive, do not change the saved group assignment

## EPDC paired-order v0-A negative result 2026-09-27

- date: 2026-09-27
- branch: exp-multidomain-mechanism
- commit: 6815281 (run configuration records full hash)
- experiment_id: epdc_development/v0a_20260927T041709Z
- purpose: test whether soft preservation of source cross-class decision ordering controls harmful evidence loss without a hard guard
- datasets: fixed In-the-Wild mechanism 512; ASV2021 LA 282; ASV2021 DF 370; Codecfake/WaveFake NOT_RUN
- manifests: existing fixed mechanism select/audit manifests; score coverage validated before selected audit labels
- method: Frozen; production `ep_no_keep` Base Adapt; Base + paired-order soft preservation
- parameters: K=5, lr=0.03, rho=0.1, gamma=0.1, lambda_keep=0, lambda_preserve=1, order retention=0.9, episodic CPU
- command: `PYTHONPATH=src:. conda run -n tta python experiments/epdc_development/v0a_worker.py` then `PYTHONPATH=src:. conda run -n tta python experiments/epdc_development/v0a_analysis.py --run experiments/epdc_development/results/v0a_20260927T041709Z`
- result_directory: `experiments/epdc_development/results/v0a_20260927T041709Z/`
- key_metrics: In-the-Wild Frozen EER/AUC 0.0990099010/0.9578409454; Base and Preserve both 0.1039603960/0.9587831364; macro source-margin damage Base 0.247588 vs Preserve 0.247582; mean order damage about 1e-7; ASV EER/AUC undefined (single-class)
- interpretation: paired-order term is almost inactive and does not protect the threshold-relevant source margins observed to degrade; no task-useful EER gain
- next_decision: retire this module, replace the evidence loss with normalized source-anchor margin deficit; do not add a gate or continual accumulation on the failed v0-A

## EPDC normalized-margin v0-B negative result 2026-09-27

- date: 2026-09-27
- branch: exp-multidomain-mechanism
- commit: 41d8c56 (run configuration records full hash)
- experiment_id: epdc_development/v0b_20260927T042152Z
- purpose: test whether a soft normalized source-anchor margin penalty prevents measured evidence damage and improves task ranking
- datasets: fixed In-the-Wild mechanism 512; ASV2021 LA 282; ASV2021 DF 370; Codecfake/WaveFake NOT_RUN
- manifests: fixed mechanism select/audit manifests; score coverage checked before selected audit labels
- method: Frozen; production `ep_no_keep` Base Adapt; Base + normalized-margin Preserve
- parameters: K=5, lr=0.03, rho=0.1, gamma=0.1, lambda_keep=0, lambda_preserve=1, margin retention=0.9, episodic CPU
- command: `PYTHONPATH=src:. conda run -n tta python experiments/epdc_development/v0b_worker.py` then `PYTHONPATH=src:. conda run -n tta python experiments/epdc_development/v0a_analysis.py --run experiments/epdc_development/results/v0b_20260927T042152Z`
- result_directory: `experiments/epdc_development/results/v0b_20260927T042152Z/`
- key_metrics: In-the-Wild Frozen EER/AUC 0.0990099010/0.9578409454; Base 0.1039603960/0.9587831364; Base+Preserve 0.1039603960/0.9564037049. Macro source damage Base 0.247588 vs Preserve 0.00000116; mean absolute score delta 0.048707 vs 0.180887. ASV EER/AUC undefined (single-class).
- interpretation: source-margin evidence is protected, but target ranking is worse; source preservation alone does not solve the objective/correction-capacity bottleneck
- next_decision: retire this module; do not add a gate/drift accumulator or lock a method; use existing P1 pseudo-label failure plus this result to design a genuinely task-aligned unlabeled objective, and restore two-class multi-domain development coverage without changing saved assignments

## Correction 2026-09-27 — ASV audit file access boundary

- date: 2026-09-27
- branch: exp-multidomain-mechanism
- commit: 66b78ab (first detection before this correction commit)
- experiment_id: correction to multidomain_mechanism/guard_20260927T040841Z and EPDC v0-A/v0-B selected-label analysis
- purpose: record an overly narrow access flag and close the raw-file scan path
- datasets: ASV2021 LA/DF official eval label artifacts; In-the-Wild selected target10 unaffected
- manifests: fixed mechanism select/audit IDs unchanged; new local selected-only audit labels derived from completed post-score CSV
- method: original `rg` selected-row extraction replaced by selected-only local audit reader
- parameters: exact selected ID coverage and canonical labels; no sampling or reranking
- command: `PYTHONPATH=src:. conda run -n tta python experiments/multidomain_mechanism/materialize_selected_audit.py --source-run experiments/multidomain_mechanism/results/guard_20260927T040841Z` (exit 0)
- result_directory: `experiments/multidomain_mechanism/audit_labels/` (local ignored artifacts), plus append-only `analysis/boundary_correction.md`
- key_metrics: not a scientific experiment; raw full ASV eval label files were scanned previously, though only selected rows were returned; target90 sample labels/metrics were not opened
- interpretation: strict final-holdout access rule was violated at file-scan level; earlier `final_holdout_labels_accessed=false` run fields must not be read as "file never opened"
- next_decision: future analysis requires selected-only artifacts; ASV label-dependent observations are not used for method selection or held-out claims

## Codecfake production-feature smoke 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: 67bb1a3 (new extraction helper was uncommitted during this engineering run; later committed as 8e797d1)
- experiment_id: task_objective_discovery/codecfake_cache_32_20260927a
- purpose: verify the first 32 fixed Codecfake official-dev waveforms against the existing Frozen SSL-AASIST production extractor before any objective analysis
- datasets: Codecfake official dev, first 32 of fixed selected 512
- manifests: existing label-free `experiments/multidomain_mechanism/manifests/codecfake_mechanism_select.json`; generated extraction JSONL contains no label
- method: production `workers/baseline_bridge.py extract` with existing Frozen bundle and three-view numerical settings
- parameters: 16 kHz, 64,600 waveform samples, 3 views, 160-dimensional embedding, float32, seed 2026 selection and source view seed 13
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/prepare_codecfake_cache.py --count 32 --run-id codecfake_cache_32_20260927a`
- result_directory: `experiments/task_objective_discovery/results/codecfake_cache_32_20260927a/`
- key_metrics: 32/32 16-kHz mono preflight; 32/32 finite feature IDs with exact coverage, 3×160 dimensions; PASS; no target labels read
- interpretation: first 32 are production-compatible; this alone does not establish compatibility of the fixed 512
- next_decision: attempt the fixed 512 without changing preprocessing or IDs

## Codecfake fixed-512 production-feature failure 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: 67bb1a3 (helper committed later as 8e797d1)
- experiment_id: task_objective_discovery/codecfake_cache_512_20260927a
- purpose: materialize the complete fixed Codecfake dev selection under the identical production waveform contract
- datasets: Codecfake official dev, fixed selected 512
- manifests: same fixed label-free Codecfake mechanism select; no audit labels opened
- method: production Frozen SSL-AASIST extraction preflight
- parameters: 16-kHz required input, 64,600 waveform samples, 3 views, 160-dimensional embedding; no resampling or redraw
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/prepare_codecfake_cache.py --count 512 --run-id codecfake_cache_512_20260927a`
- result_directory: `experiments/task_objective_discovery/results/codecfake_cache_512_20260927a/`
- key_metrics: FAIL at selected `dev/F04_SSB00090180.wav` due 24-kHz sample rate; follow-up label-free metadata audit found 222 at 16 kHz, 209 at 24 kHz, 52 at 44.1 kHz and 29 at 48 kHz; `failure.json` retained
- interpretation: full fixed selection cannot be extracted strictly through the present production preprocessing; the 32-sample smoke was unrepresentative of sample-rate coverage
- next_decision: no Codecfake full objective score or label audit; seek a separate authorized two-class official-dev domain while retaining this negative result

## Task-objective engineering smoke 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: 67bb1a3 (objective code committed later as a227bed)
- experiment_id: task_objective_discovery/objective_smoke_20260927a
- purpose: verify O1/O2/O3 finite label-free gradients and diagnostics on real cached waveforms before ranking analysis
- datasets: In-the-Wild fixed mechanism first 32; Codecfake official dev fixed first 32
- manifests: existing label-free mechanism select manifests; no audit reader or target label in worker
- method: Frozen, `ep_no_keep`, O1 soft source affinity, O2 decision-logit consistency, O3 reliability-weighted soft affinity
- parameters: episodic 8×8 R, K=5, lr=0.03, rho=0.1, projected SGD, original-view score, configuration `experiments/task_objective_discovery/objective_config.json`
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_worker.py --smoke --run-id objective_smoke_20260927a --codecfake-cache experiments/task_objective_discovery/results/codecfake_cache_32_20260927a/diagnostics/feature_cache`
- result_directory: `experiments/task_objective_discovery/results/objective_smoke_20260927a/`
- key_metrics: 32/32 samples per domain and all five arms; 160 rows/domain; finite scores and diagnostics; no EER/AUC computed
- interpretation: engineering path passes for available 32-sample caches; no task-correction conclusion
- next_decision: generate full fixed In-the-Wild scores, then perform selected-only post-score audit

## First fixed In-the-Wild objective run 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: 67bb1a3 (new objective code was uncommitted during this run; provenance defect addressed by committed-code replay below)
- experiment_id: task_objective_discovery/objective_inwild_512_20260927a
- purpose: compare three task-aligned objectives to Frozen and Base on the fixed mechanism-dev selection
- datasets: In-the-Wild target10-derived mechanism 512, 310 bona fide and 202 spoof after post-score selected audit
- manifests: existing fixed `in_the_wild_mechanism_select.json`, selected-only local audit opened after 2,560 complete score rows
- method: Frozen, `ep_no_keep`, O1/O2/O3 v0
- parameters: fixed `objective_config.json`, K=5, lr=0.03, rho=0.1, episodic source-only anchors
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_worker.py --run-id objective_inwild_512_20260927a --domains in_the_wild`; then `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_analysis.py --run experiments/task_objective_discovery/results/objective_inwild_512_20260927a`
- result_directory: `experiments/task_objective_discovery/results/objective_inwild_512_20260927a/`
- key_metrics: Frozen EER/AUC 0.099010/0.957841; O1 0.103960/0.958799; O2 0.103960/0.958895; O3 0.103960/0.958879; all paired ΔAUC bootstrap intervals include zero
- interpretation: no compelling task-ranking correction; uncommitted code makes this first run preliminary
- next_decision: replay under committed code and keep promotion closed pending an independent two-class domain

## Committed-code In-the-Wild objective replay 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: a227bed (run configuration); result-producing objective code committed
- experiment_id: task_objective_discovery/objective_inwild_512_20260927b
- purpose: establish reproducible fixed-budget single-domain evidence after the preliminary run's uncommitted-code provenance defect
- datasets: In-the-Wild same fixed mechanism 512, 310 bona fide and 202 spoof
- manifests: same saved label-free select; selected-only local audit read after exact score coverage
- method: Frozen, production `ep_no_keep`, O1/O2/O3 v0
- parameters: K=5, lr=0.03, rho=0.1, 8×8 episodic R, three views, original score, `objective_config.json`
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_worker.py --run-id objective_inwild_512_20260927b --domains in_the_wild`; then `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_analysis.py --run experiments/task_objective_discovery/results/objective_inwild_512_20260927b`
- result_directory: `experiments/task_objective_discovery/results/objective_inwild_512_20260927b/`
- key_metrics: replay reproduces Frozen 0.099010/0.957841 and O1/O2/O3 EER 0.103960 with AUC 0.958799/0.958895/0.958879; O1 paired ΔAUC 95% development interval [-0.000415, 0.002891]; 23 helpful and 0 harmful fixed-threshold flips
- interpretation: O1 changes some threshold decisions but does not establish ranking improvement; all methods retain worse EER than Frozen and only one two-class domain is scored
- next_decision: complete an independently named ASVspoof2019 PA official-dev auxiliary domain if production cache passes, without relabeling it as ASVspoof2021

## Auxiliary PA official-dev group preflight and fixed selection 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: a227bed during preflight; final manifest code and saved assignment committed as ba0fa75
- experiment_id: task_objective_discovery/asv2019_pa_dev_fixed_group_2026
- purpose: establish a second independent official development resource without altering ASVspoof2021 eval assignments
- datasets: ASVspoof2019 PA official dev, 20 explicit speaker groups in protocol; selected one audio-complete 270-record speaker group `PA_0105`
- manifests: `experiments/task_objective_discovery/manifests/asv2019_pa_dev_mechanism_select.json` and matching deferred audit; written once after aborted no-manifest preflights
- method: sorted explicit speaker-group eligibility from protocol IDs and audio availability, `random.Random(2026)` fixed selection; label column ignored during selection
- parameters: seed 2026, one complete 270-record group, 16-kHz production preflight, no target label in select
- command: `PYTHONPATH=src:. conda run -n tta python experiments/task_objective_discovery/build_pa_dev.py` (two preflight failures before final script revision; final exit 0)
- result_directory: fixed assignment in `experiments/task_objective_discovery/manifests/`; retrospective observed preflight failure at `experiments/task_objective_discovery/results/pa_group_preflight_20260927a/`
- key_metrics: ten 270-record groups existed; only `PA_0105` had all protocol audio files; selected 270/270 files at 16 kHz; selected labels unread at assignment time
- interpretation: explicit group-level development assignment is fixed, but whether it has two classes was deferred until all adaptation scores
- next_decision: production 32-waveform cache smoke, then fixed full cache and scores

## Auxiliary PA production-cache engineering smoke 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: a227bed (genericized extraction helper and PA manifest committed after this engineering run as ba0fa75)
- experiment_id: task_objective_discovery/pa_cache_32_20260927a
- purpose: validate production Frozen feature path on the first 32 fixed PA official-dev waveforms
- datasets: ASVspoof2019 PA dev, selected group first 32 of 270
- manifests: saved label-free PA select, generated worker JSONL with no label
- method: unchanged `workers/baseline_bridge.py extract` and Frozen SSL-AASIST bundle
- parameters: 16 kHz, 64,600 samples, three views, 160 embedding dimensions, float32, source view seed 13
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/prepare_codecfake_cache.py --domain asv2019_pa_dev --count 32 --run-id pa_cache_32_20260927a`
- result_directory: `experiments/task_objective_discovery/results/pa_cache_32_20260927a/`
- key_metrics: PASS, finite 3×160 features, exact 32/32 ID coverage; no selected label read
- interpretation: the first 32 can traverse the production model path; full 270 still required
- next_decision: run 32-sample objective smoke without EER/AUC, then materialize 270 cache

## Auxiliary PA objective engineering smoke 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: ba0fa75 (worker PA support uncommitted at the time, later committed as b2175fc)
- experiment_id: task_objective_discovery/objective_pa_smoke_20260927a
- purpose: verify finite objective/update behavior for O1/O2/O3 on PA real features without using labels
- datasets: ASVspoof2019 PA official dev, first 32 fixed group records
- manifests: saved label-free PA select; audit not opened
- method: Frozen, production `ep_no_keep`, O1/O2/O3 v0
- parameters: K=5, lr=0.03, rho=0.1, episodic R, three views, original score
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_worker.py --smoke --run-id objective_pa_smoke_20260927a --domains asv2019_pa_dev --pa-cache experiments/task_objective_discovery/results/pa_cache_32_20260927a/diagnostics/feature_cache`
- result_directory: `experiments/task_objective_discovery/results/objective_pa_smoke_20260927a/`
- key_metrics: 32/32 samples and 160 arm rows, zero numeric failures; mean O1 update norm 0.009595 and source damage 0.000059; no EER/AUC calculated
- interpretation: engineering behavior is finite, not evidence of task benefit
- next_decision: extract and validate all fixed 270 PA features

## Auxiliary PA fixed-group production cache 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: a227bed in run configuration (production extractor unchanged; helper/manifest were committed during this long running extraction as ba0fa75)
- experiment_id: task_objective_discovery/pa_cache_270_20260927a
- purpose: materialize exact production Frozen features for the once-fixed PA official-dev group
- datasets: ASVspoof2019 PA official dev, selected group `PA_0105`, 270 WAV-equivalent FLAC files
- manifests: saved label-free PA select and generated extraction JSONL; no audit labels opened
- method: unchanged `workers/baseline_bridge.py extract`
- parameters: 16 kHz, 64,600 waveform samples, 3 views, 160 embedding dimensions, float32, source view seed 13
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/prepare_codecfake_cache.py --domain asv2019_pa_dev --count 270 --run-id pa_cache_270_20260927a`
- result_directory: `experiments/task_objective_discovery/results/pa_cache_270_20260927a/`
- key_metrics: PASS, finite 3×160 features, exact 270/270 ID coverage, source bundle/cache identity validated; selected labels still unread
- interpretation: PA development feature path is production-compatible; class coverage still unknown before scores
- next_decision: generate full O1/O2/O3 and control scores on In-the-Wild plus PA, then audit selected labels

## Fixed In-the-Wild plus auxiliary PA objective study 2026-09-27

- date: 2026-09-27
- branch: exp-task-objective-discovery
- commit: d723f8c (score generation, committed O1/O2/O3 and worker code)
- experiment_id: task_objective_discovery/objective_two_domain_20260927a
- purpose: test whether task-aligned objectives generate repeatable spoof-discriminative correction on an independent official-dev domain
- datasets: In-the-Wild fixed 512 (310 bonafide, 202 spoof); ASVspoof2019 PA official dev fixed 270 (270 bonafide, 0 spoof after post-score audit)
- manifests: fixed label-free In-the-Wild and PA select manifests; selected-only In-the-Wild audit and PA official-dev protocol opened only after all 3,910 score rows passed exact finite coverage
- method: Frozen, production `ep_no_keep`, O1 source-anchor soft affinity, O2 decision-sensitive consistency, O3 reliability-weighted soft affinity
- parameters: preregistered `objective_config.json`: K=5, lr=0.03, rho=0.1, gamma=0.1, lambda_keep=0, episodic 8×8 R, three views, original-view score, 1,000 paired bootstrap draws where both classes exist
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_worker.py --run-id objective_two_domain_20260927a --domains in_the_wild asv2019_pa_dev --pa-cache experiments/task_objective_discovery/results/pa_cache_270_20260927a/diagnostics/feature_cache`; then `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/task_objective_discovery/objective_analysis.py --run experiments/task_objective_discovery/results/objective_two_domain_20260927a`
- result_directory: `experiments/task_objective_discovery/results/objective_two_domain_20260927a/`; small tracked summary at `experiments/task_objective_discovery/summary/`
- key_metrics: In-the-Wild Frozen EER/AUC 0.099010/0.957841; O1/O2/O3 EER 0.103960 and AUC 0.958799/0.958895/0.958879, all paired ΔAUC intervals span zero. PA EER/AUC undefined (single-class); Base/O1/O2/O3 PA source damage 0.193014/0.041150/0.236152/0.192282, harmful flips 16/3/23/14 respectively; zero numeric failures.
- interpretation: O1 changes class-mean score gap and some threshold decisions in In-the-Wild but does not show reliable ranking benefit; PA cannot validate a second-domain ranking effect. Source-damage reduction on PA is not a task gain.
- next_decision: keep all three objectives unpromoted; do not implement preservation, gate, continual, or method lock. Preserve fixed PA and all historical assignments; resolve genuine two-class development coverage only through a distinct permitted resource or report blocker.

## Fixed Codecfake 16-kHz compatibility cache 2026-09-27

- date: 2026-09-27
- branch: exp-local-distribution-tta
- commit: 409e94f (implementation used by the run)
- experiment_id: codecfake_compat/compat_20260927a
- purpose: unlock the existing fixed 512 official-dev Codecfake IDs without changing production `load_audio` or selected membership
- datasets: Codecfake official dev fixed 512; 222 native 16 kHz, 209 at 24 kHz, 52 at 44.1 kHz, 29 at 48 kHz
- manifests: unchanged `codecfake_mechanism_select.json`; no audit manifest or target label opened by the compatibility worker
- method: native 16-kHz direct production bypass; non-16-kHz mono float32 waveform resampled to 16 kHz by SciPy 1.13.0 `signal.resample_poly` (Kaiser beta 5, constant padding), then unchanged Frozen SSL-AASIST extraction
- parameters: original views=3, embedding=160, source bundle/checkpoint unchanged, native parity tolerance 1e-5, fixed order/seed=2026
- command: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/codecfake_compat/run_compat.py --run-id compat_20260927a`
- result_directory: `experiments/codecfake_compat/results/compat_20260927a/`; tracked small metadata `experiments/codecfake_compat/summary/compat_20260927a/`
- key_metrics: native32 and native222 waveform/feature/score maximum differences all exactly 0; final cache 512 unique IDs, exact fixed selected coverage, finite float32 features of shape 3×160, 222 native and 290 resampled, PASS; target labels unread
- interpretation: explicit compatibility path makes the fixed Codecfake development selection feature-compatible without silently dropping samples or altering production 16-kHz semantics; this is engineering evidence, not task benefit
- next_decision: test whether a fresh adapter shared only inside B=16/32 fixed-order local buffers improves ranking over corresponding per-sample Base/O1 arms on ITW and Codecfake; keep all labels closed until complete score coverage
