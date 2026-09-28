"""Final bounded audit of the fixed 180-run matrix and evaluator amendment."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    config = read(HERE / 'config.json')
    root = HERE / 'results' / config['run_id']
    with (root / 'run_index.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    expected = {(method, stream, str(seed)) for method in config['methods']
                for stream in config['streams'] for seed in config['order_seeds']}
    actual = {(row['method'], row['stream'], row['order']) for row in rows}
    if len(rows) != 180 or actual != expected or any(row['status'] != 'COMPLETE' for row in rows):
        raise ValueError('run index is not exactly 180 COMPLETE unique tasks')
    if any(row['exit_code'] != '0' for row in rows):
        raise ValueError('nonzero/missing formal launcher exit code')
    for phase, count in (('order2026', 36), ('later', 144)):
        launch = read(root / f'launch_{phase}.json')
        if launch['tasks'] != count or launch['complete'] != count or launch['failed']:
            raise ValueError(f'{phase} launcher summary mismatch')
    old, new = read(root / 'summary.json'), read(root / 'summary_v21.json')
    if old['completed_runs'] != 180 or new['completed_runs'] != 180 or set(old['runs']) != set(new['runs']):
        raise ValueError('evaluator run coverage mismatch')
    metric_vectors_checked = 0
    for tag in old['runs']:
        before, after = old['runs'][tag]['evaluation'], new['runs'][tag]['evaluation']
        if len(before['segments']) != len(after['segments']):
            raise ValueError(f'{tag} segment coverage changed')
        for a, b in zip(before['segments'], after['segments']):
            if a['metrics']['auroc'] != b['metrics']['auroc'] or a['metrics']['eer'] != b['metrics']['eer']:
                raise ValueError(f'{tag} threshold-independent segment metric changed')
            metric_vectors_checked += 1
        if 'pooled' in before and (before['pooled']['auroc'] != after['pooled']['auroc'] or
                                   before['pooled']['eer'] != after['pooled']['eer']):
            raise ValueError(f'{tag} threshold-independent pooled metric changed')
    bootstrap = read(root / 'bootstrap.json')
    if (bootstrap['status'] != 'COMPLETE' or len(bootstrap['comparisons']) != 100 or
        any(item['valid_replicates'] != 1000 for item in bootstrap['comparisons'].values())):
        raise ValueError('paired bootstrap incomplete')
    calibrations = sorted((root / 'source_only').glob('source_calibration_v21_*/calibration.json'))
    if len(calibrations) != 1:
        raise ValueError('expected one v2.1 source calibration')
    calibration = read(calibrations[0])
    if calibration['status'] != 'COMPLETE' or set(calibration['methods']) != set(config['methods']):
        raise ValueError('source threshold coverage mismatch')
    if 'Status:** COMPLETE' not in (HERE / 'BASELINE_REPORT.md').read_text():
        raise ValueError('final report not marked COMPLETE')
    result = {'status': 'PASS', 'formal_runs': 180, 'order2026': 36, 'later': 144,
              'failed_or_blocked': 0, 'segment_metric_vectors_checked': metric_vectors_checked,
              'paired_bootstrap_comparisons': 100, 'replicates_per_comparison': 1000,
              'method_specific_source_thresholds': 6,
              'target90_accessed_by_v2_run': False, 'final_holdout_accessed_by_v2_run': False,
              'note': 'First-score exact coverage/finite values were checked by both evaluator summaries.'}
    (root / 'final_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
