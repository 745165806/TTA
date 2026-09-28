"""Evaluator-only score join and the current complete/incomplete benchmark report."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from eptta.evaluation.metrics import binary_metrics

HERE = Path(__file__).resolve().parent


def jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def labels_for(config):
    sidecars = {
        'ITW': HERE / 'private_eval_labels_v2/itw.jsonl',
        'WaveFake': HERE / 'private_eval_labels_v2/wavefake.jsonl',
        'LA21': HERE / 'manifests/asv2021_la_online_dev/private_eval_labels.jsonl',
        'DF21': HERE / 'manifests/asv2021_df_online_dev/private_eval_labels.jsonl',
    }
    result = {}
    for domain, path in sidecars.items():
        rows = jsonl(path)
        mapping = {row['sample_id']: row for row in rows}
        if len(mapping) != config['domains'][domain]['count'] or len(mapping) != len(rows):
            raise ValueError(f'{domain} evaluator-only label coverage mismatch')
        if any(row['canonical_label'] not in (0, 1) for row in rows):
            raise ValueError('nonbinary evaluator label')
        result[domain] = mapping
    return result


def realized_run(config, root, labels, method, stream, seed):
    tag = f'{method}_{stream}_order{seed}'
    directory = root / 'runs' / tag
    if not directory.exists():
        return None
    attempts = sorted(path for path in directory.iterdir() if path.is_dir())
    complete = []
    for path in attempts:
        status_path = path / 'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status['status'] == 'COMPLETE':
                complete.append((path, status))
    if not complete:
        return None
    path, status = complete[0]
    expected_stream = jsonl(Path(config['stream_manifest_root']) / f'{stream}_order{seed}.jsonl')
    scores = jsonl(path / 'scores.jsonl')
    if len(scores) != len(expected_stream) or status['count'] != len(scores):
        raise ValueError(f'{tag} score count mismatch')
    for i, (score, expected) in enumerate(zip(scores, expected_stream)):
        if (score['sample_id'] != expected['sample_id'] or score['position'] != i or
            score['segment'] != expected['segment'] or score['batch'] != i // config['batch_size'] or
            type(score['score']) not in (int, float) or not math.isfinite(score['score'])):
            raise ValueError(f'{tag} first-score order/value mismatch at {i}')
        if score['sample_id'] not in labels[expected['domain']]:
            raise ValueError(f'{tag} evaluator label ID missing')
    return {'path': path, 'status': status, 'scores': scores, 'stream': expected_stream}


def metric(scores, labels, threshold):
    if len(set(labels)) != 2:
        return None
    return binary_metrics(scores, labels, threshold)


def score_segments(run, config, labels, threshold):
    scores = run['scores']
    rows = run['stream']
    segments = []
    for segment in sorted({item['segment'] for item in rows}):
        indices = [i for i, item in enumerate(rows) if item['segment'] == segment]
        domain = rows[indices[0]]['domain']
        x = [scores[i]['score'] for i in indices]
        y = [labels[domain][rows[i]['sample_id']]['canonical_label'] for i in indices]
        full = metric(x, y, threshold)
        rolling = []
        for start in range(0, len(x)-255, 128):
            result = metric(x[start:start+256], y[start:start+256], threshold)
            rolling.append({'start': start, 'metrics': result})
        valid = [item for item in rolling if item['metrics'] is not None]
        first256 = metric(x[:256], y[:256], threshold) if segment > 0 else None
        segments.append({'segment': segment, 'domain': domain, 'count': len(indices),
                         'metrics': full, 'first256_after_switch': first256,
                         'rolling_valid_count': len(valid),
                         'rolling_na_count': len(rolling)-len(valid),
                         'rolling_mean_auc': statistics.mean(item['metrics']['auroc'] for item in valid) if valid else None,
                         'rolling_mean_eer': statistics.mean(item['metrics']['eer'] for item in valid) if valid else None,
                         'rolling_worst_auc': min(item['metrics']['auroc'] for item in valid) if valid else None,
                         'rolling_worst_eer': max(item['metrics']['eer'] for item in valid) if valid else None})
    output = {'segments': segments}
    if len(segments) > 1:
        output['segment_macro_auc'] = statistics.mean(item['metrics']['auroc'] for item in segments)
        output['segment_macro_eer'] = statistics.mean(item['metrics']['eer'] for item in segments)
        pooled_y = [labels[item['domain']][item['sample_id']]['canonical_label'] for item in rows]
        output['pooled'] = metric([item['score'] for item in scores], pooled_y, threshold)
        output['worst_rolling_auc'] = min(item['rolling_worst_auc'] for item in segments if item['rolling_worst_auc'] is not None)
        output['worst_rolling_eer'] = max(item['rolling_worst_eer'] for item in segments if item['rolling_worst_eer'] is not None)
        first = [item['first256_after_switch'] for item in segments[1:] if item['first256_after_switch'] is not None]
        output['switch_first256_mean_auc'] = statistics.mean(item['auroc'] for item in first) if first else None
        output['switch_first256_mean_eer'] = statistics.mean(item['eer'] for item in first) if first else None
    return output


def fmt(value, percent=False):
    if value is None:
        return 'NA'
    return f'{100*value:.2f}' if percent else f'{value:.4f}'


def stats(values):
    if not values:
        return None
    return {'mean': statistics.mean(values),
            'std_ddof1': statistics.stdev(values) if len(values) > 1 else None,
            'min': min(values), 'max': max(values), 'n_orders': len(values)}


def write_report(config, results, root, thresholds=None, calibration=None):
    methods = config['methods']
    streams = config['streams']
    seeds = config['order_seeds']
    count = len(results)
    calibrated = thresholds is not None
    metrics_filename = 'metrics_v21.csv' if calibrated else 'metrics.csv'
    summary_filename = 'summary_v21.json' if calibrated else 'summary.json'
    lines = ['# BASELINE_REPORT — Online ADD Baselines v2', '',
             f'**Stage:** Online ADD Baselines v2 · **Status:** {"COMPLETE" if count == 180 else "INCOMPLETE"}',
             f'**Completed runs:** {count} / 180',
             '**Target domains:** ITW = 3,178; WaveFake = 4,096; LA21 = 4,096; DF21 = 4,096.',
             '**Methods:** ' + '; '.join(f'{name} = {"COMPLETE" if sum((name, stream, seed) in results for stream in streams for seed in seeds)==30 else "INCOMPLETE"} ({sum((name, stream, seed) in results for stream in streams for seed in seeds)}/30)' for name in methods) + '.',
             '**Operating thresholds:** ' + ('method-specific source-select calibration (v2.1 supplement; chronology disclosed below).' if calibrated else 'shared existing source-cal0 threshold (original v2 protocol).'),
             '', '## Stationary mean over five orders', '',
             '| Method | ITW EER% / AUC | WaveFake EER% / AUC | LA21 EER% / AUC | DF21 EER% / AUC |',
             '|---|---:|---:|---:|---:|']
    for method in methods:
        cells = []
        for number in range(1, 5):
            values = [results[(method, f'S{number}', seed)]['evaluation']['segments'][0]['metrics']
                      for seed in seeds if (method, f'S{number}', seed) in results]
            cells.append(f'{fmt(statistics.mean(row["eer"] for row in values), True)} / {fmt(statistics.mean(row["auroc"] for row in values))}' if len(values)==5 else 'NA')
        lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Dynamic mean over five orders', '',
              '| Method | S5 macro EER% / AUC | S6 macro EER% / AUC | Worst rolling EER% / AUC | Switch first256 EER% / AUC |',
              '|---|---:|---:|---:|---:|']
    for method in methods:
        domains = [[results[(method, stream, seed)]['evaluation'] for seed in seeds
                    if (method, stream, seed) in results] for stream in ('S5', 'S6')]
        cells = [f'{fmt(statistics.mean(row["segment_macro_eer"] for row in values), True)} / {fmt(statistics.mean(row["segment_macro_auc"] for row in values))}' if len(values)==5 else 'NA' for values in domains]
        all_values = domains[0] + domains[1]
        cells.append(f'{fmt(max(row["worst_rolling_eer"] for row in all_values), True)} / {fmt(min(row["worst_rolling_auc"] for row in all_values))}' if len(all_values)==10 else 'NA')
        switch_values = [row for row in all_values if row['switch_first256_mean_eer'] is not None]
        cells.append(f'{fmt(statistics.mean(row["switch_first256_mean_eer"] for row in switch_values), True)} / {fmt(statistics.mean(row["switch_first256_mean_auc"] for row in switch_values))} ({len(switch_values)}/10)' if len(all_values)==10 and switch_values else 'NA')
        lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Efficiency', '',
              '| Method | Updates / 100 | Adapted fraction | Backend ms/audio | Peak GPU MB | Memory records |',
              '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        rows = [value['run']['status'] for key, value in results.items() if key[0] == method]
        cells = [fmt(statistics.mean(row['updates']*100/row['count'] for row in rows)) if rows else 'NA',
                 fmt(statistics.mean(row['adapted_samples']/row['count'] for row in rows)) if rows else 'NA',
                 f'{statistics.mean(row["ms_per_audio_backend"] for row in rows):.2f}' if rows else 'NA',
                 f'{max(row["peak_gpu_mb"] for row in rows):.0f}' if rows else 'NA',
                 str(max(row['memory_size'] for row in rows)) if rows else 'NA']
        lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Adapted fraction counts samples selected for a post-prediction optimizer update. BN-only uses current-batch statistics and LAME refines current-batch output without such an update, so both show zero in that column. Memory records counts retained target examples; model/optimizer state is described in `PROTOCOL.md` and per-run status.']
    lines += ['', '## Main observations', '']
    if count < 180:
        lines += ['1. Formal results remain incomplete; no cross-method conclusion is claimed.',
                  '2. Frozen parity and future-blind prefix passed on the locked stream; see validation artifacts.',
                  '3. Source-only LR selection and Fisher are fixed before formal target scoring.']
        useful = failing = 'Pending complete comparison.'
    else:
        stationary = {}
        dynamic = {}
        for method in methods[1:]:
            stationary[method] = []
            dynamic[method] = []
            for seed in seeds:
                for number in range(1, 5):
                    current = results[(method, f'S{number}', seed)]['evaluation']['segments'][0]['metrics']
                    frozen = results[('frozen', f'S{number}', seed)]['evaluation']['segments'][0]['metrics']
                    stationary[method].append((current['auroc']-frozen['auroc'], frozen['eer']-current['eer']))
                for stream in ('S5', 'S6'):
                    current = results[(method, stream, seed)]['evaluation']
                    frozen = results[('frozen', stream, seed)]['evaluation']
                    dynamic[method].append((current['segment_macro_auc']-frozen['segment_macro_auc'],
                                            frozen['segment_macro_eer']-current['segment_macro_eer']))
        top = max(methods[1:], key=lambda method: statistics.mean(row[0] for row in stationary[method]))
        top_gain = statistics.mean(row[0] for row in stationary[top])
        top_positive = sum(row[0] > 0 for row in stationary[top])
        dynamic_top = max(methods[1:], key=lambda method: statistics.mean(row[0] for row in dynamic[method]))
        dynamic_gain = statistics.mean(row[0] for row in dynamic[dynamic_top])
        negative = [method for method in methods[1:]
                    if statistics.mean(row[0] for row in stationary[method]) < 0]
        stationary_positive = sum(row[0] > 0 for values in stationary.values() for row in values)
        dynamic_positive = sum(row[0] > 0 for values in dynamic.values() for row in values)
        domain_effects = []
        for method in methods[1:]:
            for number, domain in enumerate(('ITW', 'WaveFake', 'LA21', 'DF21'), 1):
                paired = [(results[(method, f'S{number}', seed)]['evaluation']['segments'][0]['metrics']['auroc'] -
                           results[('frozen', f'S{number}', seed)]['evaluation']['segments'][0]['metrics']['auroc'])
                          for seed in seeds]
                domain_effects.append((statistics.mean(paired), method, domain))
        worst_effect, worst_method, worst_domain = min(domain_effects)
        lines += [f'1. Adapted methods gain AUC in {stationary_positive}/100 stationary and {dynamic_positive}/50 dynamic segment-macro paired comparisons with Frozen.',
                  f'2. Least negative stationary mean ΔAUC: {top} {top_gain:+.4f}; least negative dynamic mean ΔAUC: {dynamic_top} {dynamic_gain:+.4f}.',
                  f'3. Largest domain-mean AUC drop: {worst_method} on {worst_domain} ({worst_effect:+.4f}). Five arrival orders are not independent target datasets.']
        if top_gain <= 0:
            useful = (f'None improves mean stationary or dynamic segment-macro AUC over Frozen. '
                      f'{dynamic_top} is least damaging by dynamic mean ΔAUC ({dynamic_gain:+.4f}), '
                      'but that is not an overall benefit.')
        else:
            useful = (f'{top} has the highest observed stationary mean ΔAUC ({top_gain:+.4f}, '
                      f'{top_positive}/20 positive); inspect per-domain effects and conditional intervals '
                      'before treating this as a stable benefit.')
        failing = ((f'{", ".join(negative)} have negative mean stationary ΔAUC. ' if negative else
                    'No method has negative mean stationary ΔAUC. ') +
                   f'{worst_method} on {worst_domain} has the largest domain-mean drop ({worst_effect:+.4f}); '
                   'switch and rolling metrics show its dynamic behavior. These are development-stream observations, not a final-holdout claim.')
    lines += ['', '**Which existing mechanisms appear useful for audio ADD online TTA?** ' + useful,
              '', '**Which mechanisms fail or drift?** ' + failing,
              '', '**Can this benchmark support designing a new method?** ' + ('PARTIAL' if count < 180 else 'YES'),
              '', '**Target90 accessed:** NO · **Final holdout accessed:** NO',
              '', '## Protocol and limitations', '',
              'All scores are first predictions on B16 predict-then-adapt streams; LAME refines its current batch output. The four domains are fixed development resources. LA21/DF21 are selected from official eval releases, so they are development subsets, not untouched final holdouts. ' + ('Each method uses its own source-select threshold for FPR/FNR/BA; the v2.1 supplement arrived after interim target metrics, so this is a disclosed source-only evaluator correction. The earlier shared-threshold metrics remain in historical files.' if calibrated else 'The same source-cal0 operating threshold is applied to all spoof-oriented scores.') + ' Backend timing is synchronized prediction/update elapsed wall time, including host launch overhead but excluding frozen frontend extraction, cache I/O and score-file writes. See `ACCESS_BOUNDARY.md` for file and label visibility.',
              '', '## Source-only operating points', '']
    if calibration is not None:
        lines += ['The same fixed 1,024 source-select IDs (512 per class) are scored once per freshly reset method. The threshold minimizes |FPR−FNR|, then mean error, then the numeric threshold; no target score or label enters calibration.',
                  '', '| Method | Score form | Source-select threshold | Source-select EER% |',
                  '|---|---|---:|---:|']
        for method in methods:
            item = calibration['methods'][method]
            lines.append(f'| {method} | {item["score_form"]} | {item["threshold"]:.6f} | {fmt(item["source_select_eer"], True)} |')
    else:
        lines.append(f'Historical shared source-cal0 threshold: {config["source_threshold"]:.6f}.')
    lines += [
              '', '## Stationary paired effects', '',
              'Positive ΔAUC and positive EER gain favor the method over Frozen. Each cell averages available arrival orders on the same fixed domain subset.',
              '', '| Method | Domain | Orders | Mean ΔAUC | Mean EER gain pp | AUC gains / orders |',
              '|---|---|---:|---:|---:|---:|']
    for method in methods[1:]:
        for number, domain in enumerate(('ITW', 'WaveFake', 'LA21', 'DF21'), 1):
            pairs = []
            for seed in seeds:
                key = (method, f'S{number}', seed)
                frozen_key = ('frozen', f'S{number}', seed)
                if key not in results or frozen_key not in results:
                    continue
                current = results[key]['evaluation']['segments'][0]['metrics']
                frozen = results[frozen_key]['evaluation']['segments'][0]['metrics']
                pairs.append((current['auroc']-frozen['auroc'],
                              100*(frozen['eer']-current['eer'])))
            lines.append(f'| {method} | {domain} | {len(pairs)}/5 | '
                         f'{fmt(statistics.mean(pair[0] for pair in pairs)) if pairs else "NA"} | '
                         f'{fmt(statistics.mean(pair[1] for pair in pairs)) if pairs else "NA"} | '
                         f'{sum(pair[0] > 0 for pair in pairs)}/{len(pairs)} |')
    lines += [
              '', '## Dynamic segments', '',
              f'Within-domain rolling windows use 256 records and stride 128; single-class windows are NA. First256 applies only after a switch. Full per-order segment metrics, rolling counts and supplementary pooled metrics are in `{metrics_filename}` and `{summary_filename}`.',
              '', '| Method | Stream | Segment / domain | Orders | EER% / AUC | First256 EER% / AUC | Rolling mean EER% / AUC | Worst rolling EER% / AUC | Valid windows total |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        for stream in ('S5', 'S6'):
            for segment_number, domain in enumerate(config['streams'][stream]):
                values = [results[(method, stream, seed)]['evaluation']['segments'][segment_number]
                          for seed in seeds if (method, stream, seed) in results]
                if not values:
                    lines.append(f'| {method} | {stream} | {segment_number} / {domain} | 0/5 | NA | NA | NA | NA | 0 |')
                    continue
                first = [row['first256_after_switch'] for row in values
                         if row['first256_after_switch'] is not None]
                rolling = [row for row in values if row['rolling_mean_eer'] is not None]
                full_text = f'{fmt(statistics.mean(row["metrics"]["eer"] for row in values), True)} / {fmt(statistics.mean(row["metrics"]["auroc"] for row in values))}'
                first_text = f'{fmt(statistics.mean(row["eer"] for row in first), True)} / {fmt(statistics.mean(row["auroc"] for row in first))}' if first else 'NA'
                rolling_text = f'{fmt(statistics.mean(row["rolling_mean_eer"] for row in rolling), True)} / {fmt(statistics.mean(row["rolling_mean_auc"] for row in rolling))}' if rolling else 'NA'
                worst_text = f'{fmt(max(row["rolling_worst_eer"] for row in rolling), True)} / {fmt(min(row["rolling_worst_auc"] for row in rolling))}' if rolling else 'NA'
                lines.append(f'| {method} | {stream} | {segment_number} / {domain} | {len(values)}/5 | {full_text} | {first_text} | {rolling_text} | {worst_text} | {sum(row["rolling_valid_count"] for row in values)} |')
    lines += [
              '', '## Run inventory', '',
              '| Method | Stream | Order | Status | EER% | AUC | ΔAUC vs Frozen | EER gain pp vs Frozen | FPR% @ source τ | FNR% @ source τ | BA% @ source τ |',
              '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed in seeds:
        for stream in streams:
            for method in methods:
                key = (method, stream, seed)
                if key not in results:
                    lines.append(f'| {method} | {stream} | {seed} | NOT_RUN | NA | NA | NA | NA | NA | NA | NA |')
                    continue
                evaluation = results[key]['evaluation']
                metric_values = evaluation['segments'][0]['metrics'] if stream in ('S1','S2','S3','S4') else evaluation['pooled']
                frozen_key = ('frozen', stream, seed)
                frozen = results[frozen_key]['evaluation'] if frozen_key in results else None
                frozen_metric = (frozen['segments'][0]['metrics'] if stream in ('S1','S2','S3','S4') else frozen['pooled']) if frozen else None
                delta_auc = metric_values['auroc'] - frozen_metric['auroc'] if frozen_metric else None
                delta_eer = 100*(frozen_metric['eer'] - metric_values['eer']) if frozen_metric else None
                lines.append(f'| {method} | {stream} | {seed} | COMPLETE | {fmt(metric_values["eer"], True)} | {fmt(metric_values["auroc"])} | {fmt(delta_auc)} | {fmt(delta_eer)} | {fmt(metric_values["fpr"], True)} | {fmt(metric_values["fnr"], True)} | {fmt(metric_values["balanced_accuracy"], True)} |')
    lines += ['', f'Dynamic pooled values in the inventory are supplementary; compare S5/S6 by segment macro and within-segment windows in `{summary_filename}`.',
              '', '## Order sensitivity', '',
              'Mean ± sample std (ddof=1), then min–max. Five orders are arrival-order repeats, not independent target datasets.',
              '', '| Method | Stream | Orders | EER% mean ± std [min, max] | AUC mean ± std [min, max] |',
              '|---|---|---:|---:|---:|']
    for method in methods:
        for stream in streams:
            values = [results[(method, stream, seed)]['evaluation'] for seed in seeds if (method, stream, seed) in results]
            eers = [value['segments'][0]['metrics']['eer'] if stream in ('S1','S2','S3','S4') else value['segment_macro_eer'] for value in values]
            aucs = [value['segments'][0]['metrics']['auroc'] if stream in ('S1','S2','S3','S4') else value['segment_macro_auc'] for value in values]
            es, ac = stats(eers), stats(aucs)
            ec = f'{fmt(es["mean"], True)} ± {fmt(es["std_ddof1"], True)} [{fmt(es["min"], True)}, {fmt(es["max"], True)}]' if es else 'NA'
            av = f'{fmt(ac["mean"])} ± {fmt(ac["std_ddof1"])} [{fmt(ac["min"])}, {fmt(ac["max"])}]' if ac else 'NA'
            lines.append(f'| {method} | {stream} | {len(values)}/5 | {ec} | {av} |')
    lines += ['', '## Bootstrap uncertainty', '',
              'Stationary paired 1,000-resample intervals are conditional on the realized online trajectories. No adaptation is rerun inside bootstrap. Per-order intervals are saved in `bootstrap.json`.',
              'A positive-only 95% interval favors the method; a negative-only interval favors Frozen. Intervals crossing zero are inconclusive.']
    bootstrap_path = root / 'bootstrap.json'
    if bootstrap_path.exists():
        bootstrap = json.loads(bootstrap_path.read_text())
        if bootstrap['status'] == 'COMPLETE':
            lines += ['', '| Method | Domain | ΔAUC CI positive / negative / crosses 0 | EER-gain CI positive / negative / crosses 0 |',
                      '|---|---|---:|---:|']
            for method in methods[1:]:
                for number, domain in enumerate(('ITW', 'WaveFake', 'LA21', 'DF21'), 1):
                    comparisons = [bootstrap['comparisons'][f'{method}_S{number}_order{seed}']
                                   for seed in seeds]
                    def interval_counts(field):
                        low = sum(item[field][0] > 0 for item in comparisons)
                        high = sum(item[field][1] < 0 for item in comparisons)
                        return f'{low} / {high} / {5-low-high}'
                    lines.append(f'| {method} | {domain} | {interval_counts("delta_auc_ci95")} | '
                                 f'{interval_counts("eer_gain_ci95")} |')
            all_intervals = list(bootstrap['comparisons'].values())
            auc_positive = sum(item['delta_auc_ci95'][0] > 0 for item in all_intervals)
            auc_negative = sum(item['delta_auc_ci95'][1] < 0 for item in all_intervals)
            lines += ['', f'Across {len(all_intervals)} stationary method/domain/order comparisons, AUC intervals are positive-only in {auc_positive}, negative-only in {auc_negative}, and cross zero in {len(all_intervals)-auc_positive-auc_negative}.']
        else:
            lines.append(f'Bootstrap status: {bootstrap["status"]}; complete 20 stationary stream-order groups before interpreting intervals.')
    else:
        lines.append('Bootstrap status: NOT_RUN.')
    lines += [
              '', '## Reproduction', '',
              'See `PROTOCOL.md`, `ACCESS_BOUNDARY.md`, `config.json`, `streams_locked/`, `results/<run_id>/run_index.csv`, `results/<run_id>/final_audit.json`, `results/<run_id>/source_only/`, each run’s `command.json`, `status.json`, `scores.jsonl`, and the launcher logs. First scores and large LL caches stay on this workstation; compact summary tables and the report are versioned.',
              '', '## One next research step', '',
              f'Investigate the {worst_method} failure on {worst_domain} ({worst_effect:+.4f} mean paired ΔAUC) on the fixed development subset, with Frozen as the reference, before proposing a new online rule.' if count == 180 else 'Finish all available fixed-protocol stream runs and then inspect the complete five-order comparison.', '']
    (HERE / 'BASELINE_REPORT.md').write_text('\n'.join(lines))


def main(root, calibration_path=None):
    config = json.loads((HERE / 'config.json').read_text())
    labels = labels_for(config)
    thresholds = None
    calibration = None
    if calibration_path is not None:
        calibration = json.loads(calibration_path.read_text())
        if calibration['status'] != 'COMPLETE' or set(calibration['methods']) != set(config['methods']):
            raise ValueError('source-only method-threshold calibration incomplete')
        thresholds = {method: calibration['methods'][method]['threshold']
                      for method in config['methods']}
    suffix = '_v21' if thresholds is not None else ''
    results = {}
    for seed in config['order_seeds']:
        for stream in config['streams']:
            for method in config['methods']:
                run = realized_run(config, root, labels, method, stream, seed)
                if run is None:
                    continue
                threshold = thresholds[method] if thresholds is not None else config['source_threshold']
                evaluation = score_segments(run, config, labels, threshold)
                per_run_metrics = run['path'] / f'metrics{suffix}.json'
                if per_run_metrics.exists():
                    if json.loads(per_run_metrics.read_text()) != evaluation:
                        raise ValueError('previous per-run metrics differ from current evaluation')
                else:
                    with per_run_metrics.open('x') as handle:
                        json.dump(evaluation, handle, indent=2)
                        handle.write('\n')
                results[(method, stream, seed)] = {'run': run, 'evaluation': evaluation}
    compact = {}
    for (method, stream, seed), item in results.items():
        compact[f'{method}_{stream}_order{seed}'] = {'status': item['run']['status'],
                                                    'evaluation': item['evaluation'],
                                                    'score_ref': str(item['run']['path'] / 'scores.jsonl')}
    (root / f'summary{suffix}.json').write_text(json.dumps({'status': 'COMPLETE' if len(results)==180 else 'INCOMPLETE',
        'completed_runs': len(results), 'expected_runs': 180, 'runs': compact,
        'operating_thresholds': thresholds if thresholds is not None else config['source_threshold'],
        'calibration_ref': str(calibration_path) if calibration_path is not None else None,
        'target90_accessed': False, 'final_holdout_accessed': False}, indent=2) + '\n')
    with (root / f'metrics{suffix}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['method','stream','order_seed','segment','domain','count','auc','eer','fpr','fnr','balanced_accuracy','rolling_valid_count','rolling_na_count','rolling_mean_auc','rolling_mean_eer','rolling_worst_auc','rolling_worst_eer','first256_auc','first256_eer'])
        for (method, stream, seed), item in results.items():
            for segment in item['evaluation']['segments']:
                m = segment['metrics']
                first = segment['first256_after_switch']
                writer.writerow([method,stream,seed,segment['segment'],segment['domain'],segment['count'],m['auroc'],m['eer'],m['fpr'],m['fnr'],m['balanced_accuracy'],segment['rolling_valid_count'],segment['rolling_na_count'],segment['rolling_mean_auc'],segment['rolling_mean_eer'],segment['rolling_worst_auc'],segment['rolling_worst_eer'],first['auroc'] if first else '',first['eer'] if first else ''])
    write_report(config, results, root, thresholds, calibration)
    print(json.dumps({'status': 'COMPLETE' if len(results)==180 else 'INCOMPLETE', 'completed_runs': len(results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    main(args.root, args.calibration)
