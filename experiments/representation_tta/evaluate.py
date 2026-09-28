"""Two-domain development evaluation after unlabeled pre-backend adapter TTA."""
import argparse
import csv
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from adapt import adapt_unlabeled, source_prototypes
from cache import load_ll
from model import RepresentationDetector, adapter_parameter_count, load_frozen_backend
from source_data import source_metadata
from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics

HERE=Path(__file__).parent
PROJECT=Path('/media/dell/data/fakeAudioDection/TTA')
FROZEN_CACHES={
    'select':PROJECT/'outputs_v2/ssl_aasist/cache-select',
    'itw_target10':PROJECT/'.worktrees/exp-o2-strength-audit/experiments/gradient_alignment_audit/target10_only_cache',
    'wavefake_dev':PROJECT/'.worktrees/exp-capacity-audit/experiments/capacity_audit/results/wavefake_capacity_cache_20260927a/feature_cache',
}


def source_threshold(scores,labels):
    ordered=sorted(zip(scores,labels),reverse=True)
    pos=sum(labels);neg=len(labels)-pos
    if not pos or not neg: raise ValueError('source select threshold requires both classes')
    tp=fp=i=0;best=(float('inf'),None)
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        for _,y in ordered[i:j]:tp+=y;fp+=1-y
        threshold=(ordered[j-1][0]+ordered[j][0])/2 if j<len(ordered) else ordered[j-1][0]-1e-6
        best=min(best,(abs(fp/neg-(pos-tp)/pos),float(threshold)))
        i=j
    return best[1]


def frozen_scores(domain,ids):
    cache=FeatureCache(FROZEN_CACHES[domain])
    if cache.index['num_views']!=3 or cache.index['feature_dim']!=160:
        raise ValueError('original Frozen cache shape mismatch')
    cache.verify_expected_ids(ids)
    values=cache.load_by_id()
    head=torch.load(PROJECT/'outputs_v2/ssl_aasist/frozen/linear_head.pt',
                    map_location='cpu',weights_only=True)
    if head['score_direction']!='larger_is_spoof':raise ValueError('Frozen score direction mismatch')
    scores=np.array([float(values[sid][0]@head['w'].numpy()+head['b'].item()) for sid in ids])
    if not np.isfinite(scores).all():raise ValueError('nonfinite Frozen scores')
    return scores


def selected_detector(backend,run_id,arm,device):
    selection=json.loads((HERE/'results'/run_id/arm/'selection.json').read_text())
    state=torch.load(selection['checkpoint_ref'],map_location='cpu',weights_only=True)
    if state['arm']!=arm or state['source_cache_run_id'] is None:
        raise ValueError('source adapter checkpoint identity mismatch')
    model=RepresentationDetector(backend).to(device)
    model.adapter.load_state_dict(state['adapter'],strict=True)
    model.eval()
    return model,selection


@torch.inference_mode()
def score_model(model,views,batch_size,device):
    started=time.monotonic()
    values=[]
    model.eval()
    for start in range(0,len(views),batch_size):
        batch=views[start:start+batch_size,0].to(device)
        values.extend(model(batch).detach().cpu().tolist())
    array=np.asarray(values,dtype=np.float64)
    if len(array)!=len(views) or not np.isfinite(array).all():
        raise ValueError('nonfinite/incomplete final shared-adapter scores')
    return array,time.monotonic()-started


def interval(a,b,y,groups,repeats,seed=2026):
    by_group=defaultdict(list)
    for i,group in enumerate(groups):by_group[group].append(i)
    keys=sorted(by_group);rng=random.Random(seed)
    auc=[];eer=[]
    for _ in range(repeats):
        indices=[i for group in rng.choices(keys,k=len(keys)) for i in by_group[group]]
        labels=[y[i] for i in indices]
        if set(labels)!={0,1}:continue
        m=binary_metrics([float(a[i]) for i in indices],labels,0.0)
        n=binary_metrics([float(b[i]) for i in indices],labels,0.0)
        auc.append(m['auroc']-n['auroc'])
        eer.append(n['eer']-m['eer'])
    if not auc:raise ValueError('no valid paired bootstrap replicate')
    return np.quantile(auc,[.025,.975]).tolist(),np.quantile(eer,[.025,.975]).tolist()


def development_labels(domain,ids,config):
    if domain=='itw_target10':
        audit=json.loads(Path(config['itw_audit']).read_text())
        mapping={row['sample_id']:row['label'] for row in audit['records']}
        if len(mapping)!=audit['count'] or set(mapping)!=set(ids):
            raise ValueError('ITW target10 selected label coverage mismatch')
        return [mapping[sid] for sid in ids],ids
    selection=json.loads(Path(config['wavefake_select']).read_text())
    if [row['sample_id'] for row in selection['records']]!=ids:
        raise ValueError('WaveFake fixed assignment order mismatch')
    native={'R':0,**{f'WF{i}':1 for i in range(1,8)}}
    labels=[native[sid.rsplit(':',1)[1]] for sid in ids]
    groups=[row['audio_id'] for row in selection['records']]
    if len(set(groups))!=selection['content_groups']:
        raise ValueError('WaveFake content-pair coverage mismatch')
    return labels,groups


def run(config_ref,run_id,seed,gpu):
    config=json.loads(Path(config_ref).read_text())
    if not torch.cuda.is_available() or torch.cuda.device_count()<=gpu:
        raise RuntimeError('representation-level evaluation requires a visible GPU')
    torch.set_num_threads(2)
    device=torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    output=HERE/'results'/run_id/'evaluation'
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps({**config,'run_id':run_id,'seed':seed,'gpu':gpu},indent=2)+'\n')
    started=time.monotonic()
    backend,bundle=load_frozen_backend(device)
    static,static_selection=selected_detector(backend,run_id,'static_bce',device)
    off,task_selection=selected_detector(backend,run_id,'task_supcon',device)
    fit_ids,fit=load_ll('fit',config['cache_run_id'])
    fit_rows=source_metadata(config,'fit',fit_ids)
    prototypes=source_prototypes(off,fit,fit_rows,config['target_batch_size'],device)
    torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in prototypes.items()},
               output/'source_prototypes.pt')
    del fit,fit_ids,fit_rows
    gc.collect()
    source_ids,source_views=load_ll('select',config['cache_run_id'])
    source_y=[row['label'] for row in source_metadata(config,'select',source_ids)]
    source_scores={'frozen':frozen_scores('select',source_ids)}
    source_times={}
    for name,model in (('static_source',static),('adapter_off',off)):
        source_scores[name],source_times[name]=score_model(model,source_views,
                                                          config['target_batch_size'],device)
    source_on,source_adapt_info=adapt_unlabeled(off,source_views,prototypes,config,seed,device)
    source_scores['adapter_on'],source_times['adapter_on']=score_model(source_on,source_views,
                                                                       config['target_batch_size'],device)
    thresholds={name:source_threshold(value.tolist(),source_y) for name,value in source_scores.items()}
    (output/'source_thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n')
    (output/'source_select_adapt.json').write_text(json.dumps(source_adapt_info,indent=2)+'\n')
    del source_views,source_on
    gc.collect()
    all_rows=[]
    for domain in ('itw_target10','wavefake_dev'):
        ids,views=load_ll(domain,config['cache_run_id'])
        torch.cuda.reset_peak_memory_stats(device)
        adapted,info=adapt_unlabeled(off,views,prototypes,config,seed,device)
        scores={}
        timings={}
        t=time.monotonic();scores['frozen']=frozen_scores(domain,ids);timings['frozen']=time.monotonic()-t
        for name,model in (('static_source',static),('adapter_off',off),('adapter_on',adapted)):
            scores[name],timings[name]=score_model(model,views,config['target_batch_size'],device)
        if not all(np.isfinite(s).all() for s in scores.values()):
            raise ValueError('nonfinite target scores')
        # This is the same candidate source checkpoint with adaptation disabled.
        with (output/f'{domain}_scores.csv').open('x',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['sample_id','frozen','static_source','adapter_off','adapter_on'])
            writer.writerows((sid,*(float(scores[name][i]) for name in
                                  ('frozen','static_source','adapter_off','adapter_on')))
                                 for i,sid in enumerate(ids))
        with (output/f'{domain}_target_adapter.pt').open('xb') as stream:
            torch.save({'adapter':{k:v.detach().cpu() for k,v in adapted.adapter.state_dict().items()},
                        'source_checkpoint':task_selection['checkpoint_ref'],'domain':domain,
                        'seed':seed,'target_learning_rate':config['target_learning_rate']},stream)
        info.update({'domain':domain,'count':len(ids),'inference_seconds':timings,
                     'adapter_parameters':adapter_parameter_count(adapted.adapter),
                     'peak_gpu_bytes':torch.cuda.max_memory_allocated(device),
                     'source_checkpoint':task_selection['checkpoint_ref'],
                     'static_checkpoint':static_selection['checkpoint_ref']})
        (output/f'{domain}_runtime.json').write_text(json.dumps(info,indent=2)+'\n')
        # Development labels are opened only after every arm's score is durable.
        y,groups=development_labels(domain,ids,config)
        metrics={name:binary_metrics(value.tolist(),y,thresholds[name]) for name,value in scores.items()}
        for name,m in metrics.items():
            row={'domain':domain,'arm':name,'seed':seed,'count':len(ids),
                 'auc':m['auroc'],'eer':m['eer'],'eer_percent':100*m['eer'],
                 'source_threshold':thresholds[name],'fpr':m['fpr'],'fnr':m['fnr'],
                 'balanced_accuracy':m['balanced_accuracy'],
                 'delta_auc_vs_frozen':m['auroc']-metrics['frozen']['auroc'],
                 'delta_auc_vs_static':m['auroc']-metrics['static_source']['auroc'],
                 'delta_auc_vs_own_off':m['auroc']-metrics['adapter_off']['auroc'] if name=='adapter_on' else '',
                 'eer_gain_pp_vs_frozen':100*(metrics['frozen']['eer']-m['eer']),
                 'eer_gain_pp_vs_static':100*(metrics['static_source']['eer']-m['eer']),
                 'eer_gain_pp_vs_own_off':100*(metrics['adapter_off']['eer']-m['eer']) if name=='adapter_on' else '',
                 'relative_eer_drop_vs_frozen':(metrics['frozen']['eer']-m['eer'])/metrics['frozen']['eer'],
                 'adapt_seconds':info['adapt_seconds'] if name=='adapter_on' else 0,
                 'inference_seconds':timings[name],
                 'peak_gpu_bytes':info['peak_gpu_bytes'] if name=='adapter_on' else 0,
                 'updates':info['updates'] if name=='adapter_on' else 0,
                 'numerical_failures':info['numerical_failures'] if name=='adapter_on' else 0}
            if name=='adapter_on':
                for ref in ('frozen','static_source','adapter_off'):
                    auc_ci,eer_ci=interval(scores[name],scores[ref],y,groups,config['bootstrap_replicates'])
                    row[f'auc_gain_ci_vs_{ref}']=json.dumps(auc_ci)
                    row[f'eer_gain_ci_vs_{ref}']=json.dumps(eer_ci)
            all_rows.append(row)
        print(json.dumps({'domain':domain,'status':'PASS','count':len(ids),'metrics':metrics,
                          'adapt_info':info}),flush=True)
        del views,adapted
        gc.collect()
    columns=list(dict.fromkeys(key for row in all_rows for key in row))
    with (output/'metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
        writer.writeheader();writer.writerows(all_rows)
    report={'status':'PASS','run_id':run_id,'seed':seed,'gpu':gpu,
            'elapsed_seconds':time.monotonic()-started,'target90_accessed':False,
            'final_holdout_accessed':False}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    run(args.config,args.run_id,args.seed,args.gpu)
