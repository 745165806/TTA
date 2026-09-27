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
