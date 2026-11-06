from copy import deepcopy
from pathlib import Path
import yaml
from .common import ROOT,DATA,HORIZONS
MAIN=['numeric_only','gfs_raw','text_mean','latest','semantic','semantic_temporal','full_pivot']
ABLATIONS=['without_ctsr','without_str','without_temporal','random_eligible','text_missing','topk1','topk8','timezone_cet_common','timezone_berlin','timezone_utc','lag3','lag6']

def configuration(horizon,variant='full_pivot',seed=2024):
    if horizon not in HORIZONS or variant not in MAIN+ABLATIONS:raise ValueError('Unknown frozen experiment')
    model={'name':'pivot','dim':32,'layers':1,'heads':4,'patch':16,'stride':8,'dropout':.1,'use_ctsr':True,'channel_top_k':4,'fusion':'cross_attention'}
    retrieval={'strategy':'semantic_temporal','top_k':4,'temporal_weight':.2,'lag_hours':12,'observation_latency_minutes':10}
    tz='fixed_cet'
    if variant in ('numeric_only','gfs_raw','text_missing'):
        model['fusion']='none';retrieval['strategy']='none'
    if variant=='gfs_raw':model['name']='gfs_raw'
    if variant in ('text_mean','latest','semantic','semantic_temporal','without_ctsr'):model['use_ctsr']=False
    if variant in ('text_mean','without_str'):retrieval['strategy']='mean'
    if variant=='text_mean':model['fusion']='mean'
    if variant in ('latest','semantic'):retrieval['strategy']=variant
    if variant=='without_temporal':retrieval['strategy']='semantic'
    if variant=='random_eligible':retrieval['strategy']='random'
    if variant in ('topk1','topk8'):retrieval['top_k']=int(variant[-1])
    if variant in ('lag3','lag6'):retrieval['lag_hours']=int(variant[-1])
    if variant.startswith('timezone_'):
        retrieval['common_sensitivity']=True
        tz='berlin' if variant=='timezone_berlin' else 'utc' if variant=='timezone_utc' else 'fixed_cet'
    return {'dataset':'jena','data_directory':str(DATA/('jena_'+tz)),'variant':variant,'enabled':True,'horizon':horizon,
      'optional_horizon':horizon==336,'seed':seed,'formal_seeds':[2024,2025,2026],'device':'auto','cpu_threads':2,'batch_size':8,
      'learning_rate':.001,'weight_decay':.0001,'epochs':30,'patience':5,'amp':True,'selection_metric':'mse_standardized',
      'model':model,'retrieval':retrieval,'timezone_hypothesis':tz,'comparison_scope':'common_timezone_rows' if retrieval.get('common_sensitivity') else 'all_selected_timezone_rows',
      'information_permission':'OpenMeteo GFS fixed nominal lead archive; inferred availability, not verified individual run publication',
      'origin_pairing_clock':'original naive station history-end label; physical UTC retrieval origin stored separately'}

def generate():
    root=ROOT/'configs/pivot_stage2_v2/jena';root.mkdir(parents=True,exist_ok=True)
    for h in HORIZONS:
        for v in MAIN+ABLATIONS:
            p=root/f'H{h}_{v}.yaml';text=yaml.safe_dump(configuration(h,v),sort_keys=False,allow_unicode=True)
            if p.exists() and p.read_text()!=text:raise ValueError('Changed immutable config: '+str(p))
            p.write_text(text)
if __name__=='__main__':generate()
