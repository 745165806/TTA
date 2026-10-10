"""Source-only audited label and official LA attack metadata for LL caches."""
import json
from pathlib import Path


def _jsonl(path):
    with Path(path).open() as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    by_id={row['sample_id']:row for row in rows}
    if len(by_id)!=len(rows):
        raise ValueError('duplicate source manifest ID')
    return by_id


def source_metadata(config,role,expected_ids):
    if role not in ('fit','select'):
        raise ValueError('source labels restricted to fit/select')
    root=Path(config['project_root'])/'data/manifests_v2/asv2019_la'
    inference=_jsonl(root/'inference'/f'{role}.jsonl')
    labels=_jsonl(root/'labels'/f'{role}.jsonl')
    groups=_jsonl(root/'groups'/f'{role}.jsonl')
    if set(inference)!=set(labels) or set(inference)!=set(groups) or set(inference)!=set(expected_ids):
        raise ValueError('source role/LL cache ID coverage mismatch')
    protocol_ref=config['fit_protocol'] if role=='fit' else config['select_protocol']
    official={}
    with Path(protocol_ref).open() as stream:
        for line in stream:
            fields=line.split()
            if len(fields)!=5 or fields[1] in official or fields[4] not in ('bonafide','spoof'):
                raise ValueError('malformed official source protocol')
            official[fields[1]]=(fields[3],int(fields[4]=='spoof'))
    result=[]
    for sid in expected_ids:
        row=inference[sid]
        audio_id=Path(row['audio_relpath']).stem
        if audio_id not in official or not groups[sid].get('source_group_id'):
            raise ValueError('source attack/audio/group absent')
        attack,label=official[audio_id]
        if label!=labels[sid]['canonical_label'] or ((label==0)!=(attack=='-')) or (
                label==1 and attack not in {f'A{i:02d}' for i in range(1,7)}):
            raise ValueError('source canonical/official label or attack mismatch')
        result.append({'sample_id':sid,'audio_id':audio_id,'label':label,'attack':attack})
    if len({row['audio_id'] for row in result})!=len(result):
        raise ValueError('duplicate source original audio ID')
    return result
