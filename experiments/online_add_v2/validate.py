"""Three bounded engineering checks, with no evaluator labels."""
import argparse
import json
import sys
from pathlib import Path

import torch

from cache_reader import SelectedLL, manifest_rows
from methods import OnlineMethod

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE / 'experiments/representation_tta'))
from extract_ll import PROBE, build_model, safe_audio  # noqa: E402
from model import load_frozen_backend  # noqa: E402
from author_training import load_audio  # noqa: E402
from baseline_bridge import _views  # noqa: E402


def parity(config, run_id, device, output):
    cache = SelectedLL(config, run_id, ['ITW'])
    stream = Path(config['stream_manifest_root']) / 'S1_order2026.jsonl'
    rows = [json.loads(line) for line in stream.open()][:32]
    by_id = {row['sample_id']: row for row in manifest_rows(config['domains']['ITW']['runner'])}
    original, bundle = build_model(device)
    backend, _ = load_frozen_backend(device)
    discrepancies = []
    for start in range(0, 32, 4):
        part = rows[start:start+4]
        waves = []
        for row in part:
            manifest = by_id[row['sample_id']]
            wave = torch.from_numpy(load_audio(safe_audio(Path(config['domains']['ITW']['audio_root']),
                                                       manifest['audio_relpath'])))
            waves.append(_views(wave, manifest['sample_index'], PROBE))
        batch = torch.stack(waves).reshape(-1, 64600).to(device)
        with torch.inference_mode():
            native = original(batch).reshape(len(part), 3, 2)[:, 0]
            replay_views = cache.batch([row['sample_id'] for row in part], all_views=True)
            replay = backend(replay_views.reshape(-1, 201, 128).to(device)).reshape(len(part), 3, 2)[:, 0]
        discrepancies.extend((native - replay).abs().amax(1).cpu().tolist())
    maximum = max(discrepancies)
    result = {'status': 'PASS' if maximum <= 1e-5 else 'FAIL', 'count': 32,
              'max_abs_native_vs_ll_logits': maximum, 'tolerance': 1e-5,
              'native_class_index_map': bundle['class_index_map']}
    (output / 'frozen_parity.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'PASS':
        raise RuntimeError(result)
    print(json.dumps(result), flush=True)


def prefix(config, run_id, device, output):
    cache = SelectedLL(config, run_id, ['ITW'])
    rows = [json.loads(line) for line in (Path(config['stream_manifest_root']) / 'S1_order2026.jsonl').open()][:96]
    common = rows[:64]
    futures = [rows[64:80], rows[80:96]]
    values = []
    for future in futures:
        method = OnlineMethod('tent', device, config)
        scores = []
        for start in range(0, 80, 16):
            part = (common + future)[start:start+16]
            features = cache.batch([row['sample_id'] for row in part]).to(device)
            score, logits = method.predict(features)
            scores.extend(score.tolist())
            method.adapt(features, logits)
        values.append(scores[:64])
    maximum = max(abs(a-b) for a, b in zip(values[0], values[1]))
    result = {'status': 'PASS' if maximum == 0 else 'FAIL', 'prefix_count': 64,
              'different_future_count': 16, 'max_prefix_abs_difference': maximum,
              'method': 'tent'}
    (output / 'future_blind_prefix.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'PASS':
        raise RuntimeError(result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', choices=('parity', 'prefix'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--validation-id', required=True)
    args = parser.parse_args()
    config = json.loads((HERE / 'config.json').read_text())
    if args.run_id != config['run_id'] or not torch.cuda.is_available():
        raise RuntimeError('locked run/GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    output = HERE / 'results' / args.run_id / 'validation' / args.validation_id
    output.mkdir(parents=True, exist_ok=False)
    if args.check == 'parity':
        parity(config, args.run_id, device, output)
    else:
        prefix(config, args.run_id, device, output)
