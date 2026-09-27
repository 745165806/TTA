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

## Local-distribution engineering smoke 2026-09-27

- date: 2026-09-27
- branch: exp-local-distribution-tta
- commit: e3b63af (worker under development; frozen as a08e20a before formal scoring)
- experiment_id: local_distribution_tta/local_smoke_20260927a
- purpose: establish finite, nonzero, buffer-reset behavior without label access or EER/AUC
- datasets: ITW first 32 fixed select IDs; Codecfake first 32 fixed select IDs with validated complete compatibility cache
- manifests: unchanged select-only mechanism manifests; no audit labels opened
- method: Frozen; per-sample Base/O1; fresh shared-R local Base/O1 at B=16/32
- parameters: fixed manifest order, K=5, lr=0.03, rho=0.1, 8×8 R, projected SGD, per-buffer reset
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/local_distribution_tta/run_study.py --smoke --run-id local_smoke_20260927a --codecfake-cache experiments/codecfake_compat/results/compat_20260927a/feature_cache/compat_full512`
- result_directory: `experiments/local_distribution_tta/results/local_smoke_20260927a/`
- key_metrics: 224 score rows and six local buffer rows per domain, exact 32-ID coverage, finite/nonzero updates; no EER/AUC calculated
- interpretation: engineering path passes and supports formal fixed-size development scoring; no task claim
- next_decision: commit worker, then generate all 7,168 two-domain score rows before selected-only label audit

## Local-distribution two-domain hypothesis test 2026-09-27

- date: 2026-09-27
- branch: exp-local-distribution-tta
- commit: a08e20a (formal score-generation worker); post-score analysis code 9ee2d44
- experiment_id: local_distribution_tta/local_dev_20260927a
- purpose: test whether bounded shared target context gives more spoof-discriminative ranking correction than independent per-sample adaptation
- datasets: fixed ITW 512 (310 bonafide/202 spoof); fixed Codecfake official-dev 512 (81 bonafide/431 spoof), class counts from post-score selected audit
- manifests: original label-free select manifests, fixed manifest order; selected-only ITW audit and Codecfake official-dev protocol opened only after all scores and buffer records passed exact coverage validation
- method: Frozen, per-sample production `ep_no_keep`, per-sample O1, shared-R Local-Base/O1 B16/B32; no preservation, gate, accumulation or pseudo-label BCE
- parameters: seed=2026, K=5, lr=0.03, rho=0.1, gamma=0.1, lambda_keep=0, 8×8 R, original-view score, 1,000 paired stratified bootstrap draws
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/local_distribution_tta/run_study.py --run-id local_dev_20260927a --codecfake-cache experiments/codecfake_compat/results/compat_20260927a/feature_cache/compat_full512`; `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/local_distribution_tta/analyze.py --run experiments/local_distribution_tta/results/local_dev_20260927a`; `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/local_distribution_tta/additional_analysis.py --run experiments/local_distribution_tta/results/local_dev_20260927a`
- result_directory: `experiments/local_distribution_tta/results/local_dev_20260927a/`; tracked small metrics and interpretation at `experiments/local_distribution_tta/summary/local_dev_20260927a/`
- key_metrics: 7,168/7,168 score rows, 192/192 local buffer rows, zero numeric/reset violations. ITW Frozen AUC/EER 0.957841/0.099010; local B32 Base/O1 AUC 0.958128/0.958224 with paired ΔAUC intervals crossing zero and point estimates below per-sample AUC 0.958783/0.958799. Codecfake Frozen AUC/EER 0.813354/0.283951; local B32 Base/O1 AUC 0.823294/0.825585, paired ΔAUC versus Frozen intervals `[0.006130,0.014495]` and `[0.008192,0.017073]`. All local EERs equal Frozen. Max observed local R norm 0.050450. ITW Local-Base threshold-region concentration increases despite unchanged EER.
- interpretation: shared local context produces a clear Codecfake development ranking gain at the fixed budget, but ITW ranking movement is uncertain and weaker than its per-sample counterpart. It is not repeatable two-domain evidence of a task-useful base method.
- next_decision: promote NONE. Stop buffer expansion and do not add local prior/preservation/gate/continual modules. Preserve this domain-specific positive and cross-domain negative result for the next independently defined objective hypothesis.

## Large-scale confirmation assignments and production caches 2026-09-27

- date: 2026-09-27
- branch: exp-large-scale-confirmation
- commit: 6e2b045 (fixed assignments/contract); 7e1f4f2 (production cache builder used for both formal cache runs)
- experiment_id: large_scale_confirmation/codec_large_20260927a and large_scale_confirmation/la_large_20260927a
- purpose: freeze larger label-free development populations and production-compatible features before six-order effect confirmation
- datasets: existing ITW target10 all 3178 IDs; Codecfake official dev fixed 5000 including prior fixed512; ASVspoof2019 LA official dev fixed 5000; WaveFake reader metadata only
- manifests: immutable label-free `experiments/large_scale_confirmation/manifests/*_confirmation_select.json`; Codecfake additional4488 and LA5000 drawn from sorted available IDs with `random.Random(2026).sample`; no protocol label file opened
- method: unchanged Frozen SSL-AASIST production extractor; Codecfake native16 direct production path and non16 SciPy 1.13.0 `resample_poly` compatibility; ASVspoof2019 LA native16 production path
- parameters: original three views, 160-dimensional float32 embeddings, source bundle/checkpoint/views/numerical mode unchanged; GPU0 Codecfake and GPU1 LA; prior Codecfake fixed512 view indices preserved
- command: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/large_scale_confirmation/build_cache.py --domain codecfake --run-id codec_large_20260927a`; `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/large_scale_confirmation/build_cache.py --domain asv2019_la_dev --run-id la_large_20260927a`
- result_directory: `experiments/large_scale_confirmation/results/codec_large_20260927a/` and `results/la_large_20260927a/`; tracked small metadata at `experiments/large_scale_confirmation/summary/`
- key_metrics: both 5000/5000 exact selected-ID coverage, finite float32 3×160 features, zero numeric failures; Codecfake 2140 native16 and 2860 resampled, old fixed512 feature maximum difference 0 against prior cache; LA 5000 native16. Labels unread. WaveFake pyarrow 21.0.0 available, pipeline not materialized.
- interpretation: data-path and sample-coverage prerequisites pass without changing historical selections; no ranking or class-composition claim yet
- next_decision: run the fixed four arms on all three assignments in six label-free orders, verify all scores/buffer resets, and only then open selected development labels

## ASVspoof2019 LA official protocol coverage failure before metrics 2026-09-27

- date: 2026-09-27
- branch: exp-large-scale-confirmation
- commit: ead03e3 (complete score worker); 2d10941 (registered-order audit check); label-coverage correction to analysis pending commit
- experiment_id: large_scale_confirmation/confirmation_large_20260927a, first post-score analysis attempt
- purpose: join official dev labels only after all 18 score files and 4,968 buffer records passed exact finite/order/reset validation
- datasets: ITW target10 fixed3178; Codecfake official dev fixed5000; ASVspoof2019 LA fixed5000 audio IDs
- manifests: unchanged three confirmation selects and six saved ID-only orders per domain
- method: four fixed arms; no method, objective, parameter or assignment change
- parameters: K=5, lr=0.03, rho=0.1, B=32, six registered orders
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/large_scale_confirmation/analyze.py --run experiments/large_scale_confirmation/results/confirmation_large_20260927a`
- result_directory: `experiments/large_scale_confirmation/results/confirmation_large_20260927a/analysis/failure.json`; no metric files were written
- key_metrics: NOT_COMPUTED. All 5000 LA audio IDs were scored; 4972 have official dev protocol rows and 28 do not. The local `dev_label.txt` has the same coverage. ITW and Codecfake selected labels have exact coverage.
- interpretation: official-protocol coverage was incorrectly assumed equal to FLAC availability; this is a label-availability issue, not an adaptation failure or a class-composition finding
- next_decision: retain all 5000 scores and original assignment, compute LA ranking only on the fixed 4972 official-protocol-covered IDs, report the 28 missing IDs, and keep every preregistered comparison and threshold unchanged

## Large-scale six-order development confirmation 2026-09-27

- date: 2026-09-27
- branch: exp-large-scale-confirmation
- commit: ead03e3 (four-arm score worker), 2d10941 (registered-order audit), 92ba179 (LA protocol coverage correction), d1f265e (post-score rate/class composition audit)
- experiment_id: large_scale_confirmation/confirmation_large_20260927a
- purpose: determine whether the earlier small Codecfake B32 positive AUC change exceeds sample/order noise and replicates across larger independent two-class development domains
- datasets: ITW full existing target10 3178 (2029 bonafide/1149 spoof); Codecfake official dev fixed5000 (686/4314); ASVspoof2019 LA dev fixed5000 scored, official-protocol-labelled 4972 (482/4490), 28 unlabelled IDs retained in scores
- manifests: fixed label-free confirmation selects; `manifest_order` and `random.Random(seed)` orders 2026–2030, identical within each domain across methods; no reselect or label-controlled buffer assignment
- method: Frozen, production `ep_no_keep` per sample, Local-Base B32, Local-O1 B32; Frozen and per-sample independently rerun per order
- parameters: K=5, lr=0.03, rho=0.1, gamma=0.1, lambda_keep=0, projected SGD, 8×8 R, three fixed views, original-view score, full reset per buffer, 1000 paired stratified bootstrap draws seed2026
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/large_scale_confirmation/run_scores.py --run-id confirmation_large_20260927a --codecfake-cache experiments/large_scale_confirmation/results/codec_large_20260927a/feature_cache --la-cache experiments/large_scale_confirmation/results/la_large_20260927a/feature_cache`; after exact label-free score gate, `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/large_scale_confirmation/analyze.py --run experiments/large_scale_confirmation/results/confirmation_large_20260927a`; `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/large_scale_confirmation/composition_audit.py --run experiments/large_scale_confirmation/results/confirmation_large_20260927a`
- result_directory: `experiments/large_scale_confirmation/results/confirmation_large_20260927a/`; small tracked tables and report under `experiments/large_scale_confirmation/summary/confirmation_large_20260927a/`
- key_metrics: 316272/316272 score rows, 4968/4968 buffers, zero numeric/reset failures. ITW local Base/O1 mean ΔAUC `+0.000008/+0.000031` with order std `0.000036/0.000021`; Codecfake `+0.000706/+0.001033` with order std `0.002108/0.002770`, positive paired bootstrap lower bounds only 2/6 orders each and 3/6 positive point estimates each; LA `-0.000009/-0.000010`, Frozen AUC already `0.999962`. Codecfake rate/class perfect confound: 16/24 kHz all spoof, 44.1/48 kHz all bonafide. Local Codecfake EER unchanged.
- interpretation: the prior Codecfake 512 `~+0.01` AUC result is not reproduced at 5000 across six orders; the large-set mean effect is below +0.005 and smaller than order std. ITW effect is practically negligible. LA gives no positive replication and is near ceiling. The Codecfake rate/class confound prevents a clean spoof-versus-format mechanism claim.
- next_decision: `SMALL_DEVELOPMENT_ARTIFACT`; promote NONE, stop Local-TTA as a method mainline, make no new TTA module or method lock, and preserve all negative/failed records. Target90 and final held-out metrics remain unopened.

## Local-TTA line closure and capacity-audit transition 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: 314ae7e (input evidence; this append committed separately)
- experiment_id: large_scale_confirmation/confirmation_large_20260927a (formal interpretation; no rerun)
- purpose: formally close the Local-TTA method mainline before starting supervised development capacity diagnosis
- datasets: ITW target10 3178; Codecfake official dev fixed5000 (RATE_CLASS_CONFOUNDED); ASVspoof2019 LA dev 4972 protocol-labelled (NEAR_CEILING)
- manifests: unchanged fixed large-scale confirmation assignments and six label-free orders
- method: frozen, per-sample Base, Local-Base B32, Local-O1 B32; no new method run in this entry
- parameters: K=5, lr=0.03, rho=0.1, B=32, projected 8×8 R, six orders, 1000 paired bootstrap draws
- command: no new experiment; interpretation of preserved `experiments/large_scale_confirmation/summary/confirmation_large_20260927a/`
- result_directory: `experiments/large_scale_confirmation/results/confirmation_large_20260927a/` (read-only historical run)
- key_metrics: ITW Local-Base/O1 mean ΔAUC +0.000008/+0.000031; Codecfake +0.000706/+0.001033 versus order std 0.002108/0.002770; LA −0.000009/−0.000010
- interpretation: `SMALL_DEVELOPMENT_ARTIFACT`; Local-TTA mainline = CLOSED; promoted candidate = NONE. Codecfake rate/class confounding and LA ceiling restrict mechanism interpretation.
- next_decision: audit clean WaveFake paired development data; then run a preregistered **supervised development diagnosis**, not a TTA method or final evaluation, to locate representation versus adapter capacity. Target90 and final held-out metrics remain unopened.

## WaveFake local paired-development resource audit 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: 314ae7e input; audit/assignment committed with this entry
- experiment_id: capacity_audit/wavefake_resource_audit_20260927a
- purpose: determine whether an independent two-class, content-related development resource is locally available without a rate/codec class confound
- datasets: local WaveFake 131 Parquet partitions, 104800 rows, 13100 complete `audio_id` groups
- manifests: newly fixed `experiments/capacity_audit/manifests/wavefake_capacity_select.json` and separate supervised-development labels sidecar; 2048 groups / 4096 waveform rows selected once using seed2026
- method: Parquet metadata and WAV-header audit; no detector adaptation or ranking experiment
- parameters: pyarrow 21.0.0; `random.Random(2026).sample(sorted_audio_ids,2048)`; R plus one WF1..WF7 generator per group in cyclic order
- command: `conda run -n tta python experiments/capacity_audit/audit_wavefake.py --out experiments/capacity_audit/wavefake_audit.json`; `conda run -n tta python experiments/capacity_audit/make_wavefake_assignment.py`
- result_directory: `experiments/capacity_audit/` (audit JSON/MD and fixed manifests; no supervised metrics yet)
- key_metrics: 13100/13100 IDs have R plus WF1..WF7; all 104800 WAV headers mono, 22050 Hz, uncompressed 16-bit PCM; zero invalid WAV; selected 4096 unique rows with 2048 real and 2048 generated; generator counts 292–293 each
- interpretation: matched-format paired development is feasible, with an unresolved small systematic duration-offset cue and no independent speaker/language/transcript metadata. This is not proof of spoof-specific generalization.
- next_decision: use the fixed explicit 22050→16000 compatibility path and production Frozen extractor; require 32-waveform smoke and full selected-cache validation before five-fold supervised capacity analysis. No target90 or final holdout access.

## ITW supervised capacity ladder 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: 79b2d2b (pre-registered implementation)
- experiment_id: capacity_audit/itw_capacity_20260927a
- purpose: locate whether the current bounded 8×8 adapter or frozen 160D representation limits task correction on the existing complete target10 development set
- datasets: In-the-Wild existing target10, 3178 rows (2029 bonafide, 1149 spoof); no target90
- manifests: unchanged fixed target10 selection and selected-only target10 development labels; five held-out stratified folds saved in run
- method: C0 Frozen, C1 supervised 8×8 R with fixed U/w/b and radius 0.1, C2 160D linear probe, C3 160→32→1 nonlinear probe triggered by preregistered C2 condition; **SUPERVISED DEVELOPMENT DIAGNOSIS, NOT TTA OR FINAL PERFORMANCE**
- parameters: seed2026; 5 folds; Adam lr0.01, batch256, L2 1e-4, max100 epochs, inner training-only 10% validation, patience10, min improvement1e-4; original-view frozen feature
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/capacity_audit/run_ladder.py --domain itw --run-id itw_capacity_20260927a`
- result_directory: `experiments/capacity_audit/results/itw_capacity_20260927a/`
- key_metrics: C0 AUC/EER 0.963309/0.098571; C1 0.963385/0.099064 (ΔAUC +0.000076, EER worse 0.000493); C2 0.973159/0.085292 (ΔAUC +0.009850, EER improvement +0.013279); C3 0.974185/0.082681 (ΔAUC +0.010876, EER improvement +0.015890). C2 and C3 AUC/EER improve versus C0 in all five held-out folds; C1 R norms reach radius 0.1.
- interpretation: ITW current bounded R has SMALL adapter gap; frozen 160D representation has a moderate AUC and actionable EER linear-readout gap in held-out CV. This is one primary domain only; pooling fold scores may have calibration effects, so per-fold directions are essential. No general TTA candidate follows yet.
- next_decision: complete WaveFake 32-waveform smoke and fixed 4096-feature cache, then repeat identical supervised ladder with content groups held out; only cross-domain replication can support a representation-versus-adapter conclusion. No gradient-direction study because C1 is not large.

## WaveFake fixed paired production cache 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: 79b2d2b (extractor), 4a1d2d8 (paired inner-validation correction; before WaveFake metrics)
- experiment_id: capacity_audit/wavefake_smoke32_20260927a and wavefake_capacity_cache_20260927a
- purpose: materialize the once-fixed 2048 real/generated content pairs through an explicit deterministic resampler and unchanged production Frozen SSL-AASIST feature path
- datasets: WaveFake local paired development assignment, 4096 selected waveforms from 2048 `audio_id` groups; all source 22050 Hz mono PCM16
- manifests: `experiments/capacity_audit/manifests/wavefake_capacity_select.json`; separate supervised labels sidecar was not read by the cache worker
- method: SciPy 1.13.0 `resample_poly(up=320,down=441,window=('kaiser',5.0),padtype='constant')` to float32 16-kHz WAV; unchanged Frozen production extractor; 3×160 float32 features
- parameters: same source bundle, checkpoint, crop/pad 64600, safe-view definitions, score direction; smoke ran CPU because sandbox did not expose CUDA; full 4096 extraction used approved GPU0 production path
- command: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/capacity_audit/build_wavefake_cache.py --run-id wavefake_smoke32_20260927a --smoke`; same without `--smoke` and run-id `wavefake_capacity_cache_20260927a` under GPU-visible execution
- result_directory: `experiments/capacity_audit/results/wavefake_smoke32_20260927a/` and `results/wavefake_capacity_cache_20260927a/`
- key_metrics: smoke 32/32 PASS in 117.80 seconds; complete cache 4096/4096 PASS in 227.76 seconds, 4096 unique IDs, finite 3×160 float32, exact selected-ID coverage, no label sidecar read, no source overwrite
- interpretation: production-compatible paired feature access is established; it supplies no supervised ranking or TTA benefit by itself. GPU/CPU implementation uses the same production worker and declared numerical mode; no cross-device bitwise parity claim is made.
- next_decision: use only the fixed selected supervised label sidecar to run preregistered five-fold grouped capacity ladder; never use labels to change assignment or preprocessing.

## WaveFake supervised capacity ladder 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: 4a1d2d8 (paired inner/outer split implementation), 50e3881 (validated cache metadata)
- experiment_id: capacity_audit/wavefake_capacity_20260927a
- purpose: test whether the bounded current R or frozen 160D representation can support task-relevant correction on a second independent paired-content development domain
- datasets: fixed WaveFake 2048 related-content IDs, one real and one generated per ID, 4096 selected development waveforms
- manifests: `wavefake_capacity_select.json`, separate `wavefake_capacity_labels.json`; saved five outer content-group folds in result; selected labels used only by supervised diagnostic after full 4096 cache PASS
- method: C0 Frozen, C1 supervised radius-0.1 8×8 R, C2 160D linear probe, conditional C3 160→32→1 probe; **SUPERVISED DEVELOPMENT DIAGNOSIS, NOT TTA OR FINAL PERFORMANCE**
- parameters: seed2026; five outer stratified content-group folds; inner 10% content-group validation; Adam lr0.01, batch256, L2 1e-4, max100 epochs, patience10, min delta1e-4
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/capacity_audit/run_ladder.py --domain wavefake --run-id wavefake_capacity_20260927a --cache experiments/capacity_audit/results/wavefake_capacity_cache_20260927a/feature_cache --assignment experiments/capacity_audit/manifests/wavefake_capacity_select.json --labels experiments/capacity_audit/manifests/wavefake_capacity_labels.json`
- result_directory: `experiments/capacity_audit/results/wavefake_capacity_20260927a/`
- key_metrics: C0 AUC/EER 0.915003/0.157715; C1 0.912394/0.161621 (worse); C2 0.951711/0.114258 (ΔAUC +0.036708, EER improvement +0.043457); C3 0.952616/0.113281. C2 improves AUC and EER in all five held-out folds. Post-hoc production-crop duration stratum: 1722 both-long content pairs, Frozen/C2 AUC 0.923785/0.961800.
- interpretation: second-domain large/actionable linear-readout gap and absent bounded-R gain support Case B, with WaveFake duration/source-artifact caveat. C3 incremental gain over C2 is small. This remains supervised development evidence.
- next_decision: run the preplanned bidirectional cross-domain linear probe to check whether readout transfers; verify bounded-R optimizer with exact convex equivalent because all C1 folds reach radius. No new TTA arm or final evaluation.

## Bidirectional cross-domain supervised development probe 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: f194af9 (cross-domain runner), 4a1d2d8 (paired inner-validation correction)
- experiment_id: capacity_audit/itw_wavefake_crossdomain_20260927a
- purpose: distinguish a shared spoof direction from domain-specific linear readout in already frozen features
- datasets: ITW target10 3178 and fixed WaveFake paired development 4096; both development labels are permitted for training/diagnosis
- manifests: unchanged ITW target10 and WaveFake fixed paired assignment; no final holdout
- method: same supervised C2 linear probe trained on one development domain, tested on the other; both directions
- parameters: same Adam lr0.01/batch256/L2 1e-4/max100/inner 10%/patience10; WaveFake training inner split grouped by `audio_id`
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/capacity_audit/run_crossdomain.py --run-id itw_wavefake_crossdomain_20260927a --cache experiments/capacity_audit/results/wavefake_capacity_cache_20260927a/feature_cache --assignment experiments/capacity_audit/manifests/wavefake_capacity_select.json --labels experiments/capacity_audit/manifests/wavefake_capacity_labels.json`
- result_directory: `experiments/capacity_audit/results/itw_wavefake_crossdomain_20260927a/`
- key_metrics: ITW→WaveFake AUC/EER 0.903675/0.172363 versus destination Frozen 0.915003/0.157715; WaveFake→ITW 0.960763/0.100049 versus destination Frozen 0.963309/0.098571
- interpretation: supervised within-domain linear gains do not transfer as a simple shared probe across these two development sources. This does not prove no universal spoof direction under other representations.
- next_decision: retain domain-specific readout limitation; do not propose source-trained head swap as method. Finish C1 optimization check before the parameterization decision.

## Bounded-R convex solver verification 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: 016fc4e (precommitted C1 solver-check contract and implementation)
- experiment_id: capacity_audit/itw_c1_convex_check_20260927a and wavefake_c1_convex_check_20260927a
- purpose: check whether Adam early stopping falsely made the current bounded-R parameter space appear incapable, using the exact eight-dimensional convex score-equivalent under the same radius/BCE
- datasets: ITW target10 3178; WaveFake fixed paired development 4096; same saved five outer folds as original capacity runs
- manifests: unchanged assignments and saved fold IDs; no target90/final holdout
- method: supervised BCE + same 1e-4 minimum-R-norm L2, SciPy 1.13.0 SLSQP with analytic gradient and norm constraint; train on four folds, evaluate only fifth
- parameters: rho0.1, maxiter500, ftol1e-10, zero initialization, no search; all ten fold solves successful and feasible
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/capacity_audit/verify_c1_convex.py --domain itw --run-id itw_c1_convex_check_20260927a`; same for `--domain wavefake --run-id wavefake_c1_convex_check_20260927a` with fixed `--cache`, `--assignment`, `--labels` paths from the WaveFake ladder command
- result_directory: `experiments/capacity_audit/results/itw_c1_convex_check_20260927a/` and `results/wavefake_c1_convex_check_20260927a/`
- key_metrics: ITW optimized C1 AUC/EER 0.963335/0.098571 (ΔAUC +0.000027); WaveFake 0.908047/0.166992 (ΔAUC −0.006956, EER worse 0.009277). Both remain non-actionable; R-equivalent q reaches its radius in all folds.
- interpretation: the original C1 non-result is not an Adam early-stop artifact for supervised BCE in the same bounded score family. This does not rule out another supervised ranking loss or a larger radius/subspace.
- next_decision: classify current frozen-head plus bounded-R space as the bottleneck relative to linearly readable target features; preserve cross-domain nontransfer and WaveFake duration caveats. No gradient-direction study because C1 has no actionable gain; no optional backend fine-tune because C2 already answers the representation-readout question.

## Target90 feature-cache I/O boundary correction 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: afc0639 (capacity results before correction); correction appended in a new commit
- experiment_id: capacity_audit/target90_feature_io_correction_20260927a
- purpose: correct the strict access declaration after reviewing the ITW feature reader used in the completed capacity runs
- datasets: existing ITW cache index contains 31779 target_test feature rows, while the fixed target10 selection contains 3178
- manifests: unchanged selected-only target10 labels and selection; no target90 labels or metrics opened
- method: read-only code/path audit of `load_itw()`→`load_context()`→`FeatureCache.iter_chunks()`; no metric rerun
- parameters: no adaptation or training change
- command: read `experiments/multidomain_mechanism/guard_worker.py::load_context`, `src/eptta/cache/reader.py::iter_chunks`, and the existing cache `index.json` sample_count
- result_directory: `experiments/capacity_audit/TARGET90_FEATURE_IO_CORRECTION.md`
- key_metrics: the loader reads all 31779 feature-cache rows at chunk I/O, then retains only the wanted 3178 target10 rows. Under strict byte-access semantics, target90 feature bytes accessed = YES; target90 labels/metrics/selection/adaptation = NO.
- interpretation: earlier `target90_accessed=false` capacity run fields were too broad; they correctly describe label/metric/selection use, but not feature-cache I/O. Scientific CV inputs remain target10-only, yet the strict access boundary was crossed.
- next_decision: make no new target90 read; require a target10-only feature artifact or selective reader for any future run that needs a strict `target90 accessed=NO` claim. Preserve all original runs and this correction.

## WaveFake future-holdout boundary clarification 2026-09-27

- date: 2026-09-27
- branch: exp-capacity-audit
- commit: afc0639 result context; clarification appended in a new commit
- experiment_id: capacity_audit/wavefake_future_holdout_boundary_20260927a
- purpose: prevent the already audited unselected WaveFake rows from later being misrepresented as untouched final data
- datasets: all 104800 local WaveFake rows were covered by class-code/WAV-header resource audit; 4096 fixed selected rows entered supervised development CV
- manifests: fixed WaveFake capacity selection unchanged; no final-holdout manifest created
- method: boundary clarification only; no new metric or selection
- parameters: none
- command: review `experiments/capacity_audit/audit_wavefake.py` and `wavefake_audit.json`
- result_directory: `experiments/capacity_audit/WAVEFAKE_FUTURE_HOLDOUT_BOUNDARY.md`
- key_metrics: no final-holdout metric was computed; full local resource metadata/headers and label codes were read during the authorized audit
- interpretation: unselected local WaveFake rows are not untouched under byte-access semantics
- next_decision: retain audited resource for development only; any future final claim requires independently protected data or a transparent prior-access account.

## Supervised five-fold head geometry 2026-09-27

- date: 2026-09-27
- branch: exp-head-tta
- commit: a780b5c (precommitted H0–H4 contract/runner)
- experiment_id: head_capacity_geometry/itw_head_geometry_20260927a and wavefake_head_geometry_20260927a
- purpose: separate source-head calibration, fixed-norm direction change, and full linear target-head capacity before any new unlabeled adaptation
- datasets: existing ITW target10 3178; fixed WaveFake related-content paired development 4096
- manifests: exact saved five outer capacity folds; WaveFake `audio_id` kept together in outer and inner splits; ITW target10-only labels and WaveFake selected-only labels used solely by supervised audit
- method: H0 bias-only; H1 positive scale+bias; H2 independently trained fixed-source-norm direction with source bias; H3 full linear reproducing prior C2; H4 reused prior nonlinear C3 held-out scores
- parameters: H0/H1 convex BCE, L2 1e-4; H2/H3 Adam lr0.01, batch256, max100 epochs, inner10% validation, patience10, seed2026; source 160D original-view embeddings, no encoder update
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/head_capacity_geometry/run_geometry.py --domain itw --run-id itw_head_geometry_20260927a`; same with `--domain wavefake --run-id wavefake_head_geometry_20260927a`
- result_directory: `experiments/head_capacity_geometry/results/itw_head_geometry_20260927a/` and `results/wavefake_head_geometry_20260927a/`
- key_metrics: ITW H2/H3 AUC `0.967247/0.973159` vs Frozen `0.963309`; WaveFake `0.934795/0.951711` vs Frozen `0.915003`; H2/H3 EER ITW `0.096106/0.085292`, WaveFake `0.135742/0.114258`. H2 AUC improvements occur in all five folds of both domains. H2 recovers `39.98%/53.92%` of H3 AUC gain, below predeclared 80%. H3 prior C2 score parity max difference 0; H0/H1 preserve within-fold ranking exactly.
- interpretation: supervised direction change contributes but independently constrained H2 is not equivalent to H3; `DECISION_DIRECTION_SHIFT` by preregistered H2≈H3 rule is NOT_CONFIRMED. Pooled H0/H1 shifts are fold-specific calibration, not new within-fold discrimination. Supervised linear probe remains only an upper bound, not TTA.
- next_decision: quantify actual fold head vectors and R-expressible correction; do not implement H-UA1 until geometry/R audits complete.

## Supervised source-target head shift 2026-09-27

- date: 2026-09-27
- branch: exp-head-tta
- commit: 7355fc6 (geometry analysis code)
- experiment_id: head_capacity_geometry/head_shift_20260927a
- purpose: measure source versus supervised target decision direction and bias across saved five-fold heads, plus descriptive cross-domain delta similarity
- datasets: ITW target10 and fixed WaveFake paired development; no new sample selection
- manifests: H3 head weights from the two saved five-fold geometry runs
- method: normalize each w to unit length; report cosine, angle, norm of `unit(w*)−unit(w_s)`, raw bias delta, and mean/25 pairwise cross-domain delta cosines
- parameters: deterministic vector analysis; no fit or hyperparameter
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/head_capacity_geometry/analyze_geometry.py --run-id head_shift_20260927a`
- result_directory: `experiments/head_capacity_geometry/results/head_shift_20260927a/`
- key_metrics: mean source-target angle ITW `83.37°`, WaveFake `85.50°`; mean cross-domain unit-delta cosine `0.664`, 25 pairwise cosines `0.566–0.705`; raw bias delta means `−9.94/+10.70`
- interpretation: large target-specific head rotations exist in supervised development, but geometric delta similarity does not establish direct transfer; historical bidirectional direct probe transfer was negative.
- next_decision: test how much of the normalized supervised head displacement lies in the actual production R-induced effective-head family.

## Production R effective-head expressibility 2026-09-27

- date: 2026-09-27
- branch: exp-head-tta
- commit: 7355fc6 (precommitted effective-head contract and runner)
- experiment_id: head_capacity_geometry/r_expressibility_20260927a
- purpose: quantify why the supervised target linear-boundary correction could not be reached by the previous frozen-head 8×8 R space
- datasets: fixed ITW target10 feature rows for parity; 5 ITW and 5 WaveFake supervised H3 fold heads
- manifests: unchanged geometry fold heads; no new target labels used in this projection
- method: derive `w_eff=w_s+U Rᵀ(Uᵀw_s)` from actual `apply_adapter`; verify 20 random radius-0.1 R on 32 real embeddings; project scale-normalized H3 head deltas to span(U), then impose q-radius `0.1||Uᵀw_s||`
- parameters: seed2026, rho0.1, parity tolerance1e-5, predeclared mismatch threshold ≥4/5 folds per domain with explained fraction<0.5
- command: `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/head_capacity_geometry/audit_r_expressibility.py --run-id r_expressibility_20260927a`
- result_directory: `experiments/head_capacity_geometry/results/r_expressibility_20260927a/`
- key_metrics: direct/derived score parity max `1.907e-6`; U orthogonality error `2.533e-8`; `||Uᵀw_s||=0.483962`, allowed q norm `0.048396`. Mean subspace explained fraction ITW/WaveFake `0.172633/0.171386`; radius-limited `0.033830/0.033058`; all 5 folds in each domain <0.5.
- interpretation: `R_PARAMETERIZATION_MISMATCH` confirmed under the predeclared rule for these supervised H3 directions. Previous bounded-R loss searches cannot recover most of this measured head correction.
- next_decision: retire R-adapter TTA research and implement only the predeclared label-free episodic source-anchored linear head prototype H-UA1, with fixed B256 and three alpha values.
