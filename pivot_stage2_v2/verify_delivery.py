"""Evidence-only final probes and source/data/config binding verification."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
import yaml
from .common import ROOT,DATA,ART,sha,digest,read_json,write_json,verify
from .train import CODE_BINDING,features
from .models import build_model
from .data import MIMODataset,loader

def probes():
    torch.set_num_threads(2)
    root=ART/'smoke/H168/full_pivot/seed2024'
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    model=build_model(saved['binding']['model']).eval();model.load_state_dict(saved['best_model'])
    result=[]
    for split,wanted in [('train','2022-01-15 12:10'),('validation','2024-07-15 12:10'),('test','2025-03-30 12:10')]:
        ds=MIMODataset(DATA/'jena_fixed_cet',split,168,check_hashes=False)
        i=int(np.argmin(abs(ds.naive_times[ds.starts+359]-pd.Timestamp(wanted).value)))
        ds.starts=ds.starts[i:i+1];batch=next(iter(loader(ds,1)))
        with torch.inference_mode():pred=model(**features(batch,torch.device('cpu')))
        assert pred.shape==(1,168,21) and torch.isfinite(pred).all() and batch['text_mask'].any()
        result.append({'split':split,'wanted_history_end_naive':wanted,'history_end_UTC':pd.Timestamp(batch['history_end_time'][0],tz='UTC').isoformat(),
            'forecast_origin_UTC':pd.Timestamp(batch['origin_time'][0],tz='UTC').isoformat(),'first_target_UTC':pd.Timestamp(batch['first_target_time'][0],tz='UTC').isoformat(),
            'output_shape':list(pred.shape),'all_finite':True,'legal_GFS_facts':len(batch['gfs_pool_ids'][0]),'sample_id':batch['sample_id'][0],'optimizer_steps_in_probe':0})
    write_json(ART/'seasonal_model_probes.json',{'checkpoint_sha256':sha(root/'best.pt'),'gfs_success_sha256':sha(DATA/'gfs/_SUCCESS.json'),
      'real_predictor':'full_pivot','probes':result,'no_future_station_observations_used_to_generate_text':True})
    ds=MIMODataset(DATA/'jena_fixed_cet','validation',24,check_hashes=False)
    found=None
    for i,start in enumerate(ds.starts):
        stop=int(start)+360;origin=int(ds.times[stop-1])+10*60*10**9
        if not ds.store.pool(origin,ds.times[stop:stop+24],12)[1].any():found=i;break
    assert found is not None
    ds.starts=ds.starts[found:found+1];batch=next(iter(loader(ds,1)))
    cp=ART/'smoke/H24/full_pivot/seed2024/best.pt';s=torch.load(cp,map_location='cpu',weights_only=False)
    m=build_model(s['binding']['model']).eval();m.load_state_dict(s['best_model'])
    with torch.inference_mode():p=m(**features(batch,torch.device('cpu')))
    assert torch.isfinite(p).all() and not batch['text_mask'].any() and not batch['gfs_mask'].any()
    write_json(ART/'real_no_GFS_inference.json',{'split':'validation','horizon':24,'origin_UTC':pd.Timestamp(batch['origin_time'][0],tz='UTC').isoformat(),
      'sample_id':batch['sample_id'][0],'real_missing_GFS_sample_retained':True,'text_mask_empty':True,'raw_forecast_mask_empty':True,
      'prediction_shape':list(p.shape),'prediction_all_finite':True,'optimizer_steps':0,'checkpoint_sha256':sha(cp),'gfs_success_sha256':sha(DATA/'gfs/_SUCCESS.json'),'not_a_research_result':True})

def main():
    unit=next(ET.parse(ART/'logs/validation_tests.xml').getroot().iter('testsuite'))
    assert int(unit.attrib['errors'])==int(unit.attrib['failures'])==0
    summaries=[];fairness={};pools={}
    for p in sorted((ART/'smoke').rglob('_SUCCESS.json')):
        r=verify(p.parent);c=yaml.safe_load((p.parent/'config.yaml').read_text());b=r['binding'];s=read_json(p.parent/'summary.json')
        assert b['code_sha256']==CODE_BINDING and b['prepared_data_sha256']==sha(Path(c['data_directory'])/'_SUCCESS.json')
        assert b['gfs_data_sha256']==sha(DATA/'gfs/_SUCCESS.json') and r['config_digest']==digest(c)
        assert s['smoke_only'] and s['optimizer_steps']==1 and s['checkpoint_roundtrip'] and s['no_text_finite'] and s['test']['samples']==1
        h=s['horizon'];ids=read_json(p.parent/'test/sample_ids.json');trace=read_json(p.parent/'retrieval_trace.json')
        truth=np.load(p.parent/'test/truth.npy');mask=np.load(p.parent/'test/mask.npy')
        if h in fairness:
            a=fairness[h];assert ids==a[0] and np.array_equal(truth,a[1],equal_nan=True) and np.array_equal(mask,a[2])
            assert trace['gfs_pool_ids']==pools[h]
        else:fairness[h]=(ids,truth,mask);pools[h]=trace['gfs_pool_ids']
        summaries.append(s)
    assert len(summaries)==21 and all(s['seed']==2024 for s in summaries)
    assert not list((ART/'experiments').rglob('summary.json'))
    collection=Path(read_json(ART/'latest_collection.json')['directory']);verify(collection)
    results=read_json(collection/'results.json');assert results['status']=='no_formal_results_yet' and not results['runs'] and results['smoke_results_excluded']
    probes()
    report={'pytest_passed':int(unit.attrib['tests']),'real_smoke_runs':len(summaries),'all_runs_single_step':True,
      'all_current_code_data_config_bindings_match':True,'identical_test_samples_labels_masks_and_legal_forecast_pools_across_methods':True,
      'three_seasonal_real_forward_probes_passed':True,'real_zero_GFS_inference_passed':True,'formal_experiments_started':False,'formal_completed_runs':0,
      'empty_formal_collection_verified':True,'v1_preservation':read_json(ART/'v1_preservation.json'),
      'code_sha256':CODE_BINDING,'gfs_success_sha256':sha(DATA/'gfs/_SUCCESS.json')}
    write_json(ART/'delivery_verification.json',report)
    print('Final verified:',report['pytest_passed'],'tests,',len(summaries),'real one-step pilots, zero formal experiments')
if __name__=='__main__':main()
