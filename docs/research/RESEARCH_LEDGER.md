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
