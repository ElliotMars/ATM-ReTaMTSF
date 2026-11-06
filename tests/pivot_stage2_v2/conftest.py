import socket
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from pivot_stage2.common import ROOT,seal,write_json
from pivot_stage2.data import valid_windows
from pivot_stage2_v2.common import HOUR

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    torch.set_num_threads(2)
    def fail(*a,**k):raise AssertionError('Unit tests must be offline')
    monkeypatch.setattr(socket,'create_connection',fail)
    monkeypatch.setattr(socket.socket,'connect',fail)

@pytest.fixture
def bundle(tmp_path):
    numeric=tmp_path/'jena_fixed_cet';numeric.mkdir();gfs=tmp_path/'gfs';gfs.mkdir()
    channels=json.loads((ROOT/'dataset/PIVOT/stage2_v1/jena/manifest.json').read_text())['channels']
    count=750;start=pd.Timestamp('2022-01-01 00:10',tz='UTC').value
    t=start+np.arange(count*3)*HOUR
    rng=np.random.default_rng(921)
    raw=rng.normal(size=(count*3,21));raw[:,0]+=995;raw[:,1]+=10;raw[:,4]+=65
    mask=np.ones(raw.shape,bool);mask[::11,2]=False;mask[:,-1]=False
    means=np.where(mask[:count],raw[:count],0).sum(0)/np.maximum(mask[:count].sum(0),1)
    scale=np.sqrt(np.where(mask[:count],raw[:count]-means,0).__pow__(2).sum(0)/np.maximum(mask[:count].sum(0),1));scale[scale==0]=1
    scaled=((raw-means)/scale).astype(np.float32)
    meta={'dataset':'jena','channels':channels,'lookback':360,'horizons':[24,72,168,336],'source_n2_success_sha256':'synthetic-source',
      'observation_latency_minutes':10,'scaler':{'mean':means.tolist(),'scale':scale.tolist()},'timezone_hypothesis':'fixed_cet'}
    for i,s in enumerate(('train','validation','test')):
        sl=slice(i*count,(i+1)*count);times=t[sl];segments=np.zeros(count,np.int64)
        np.savez(numeric/(s+'.npz'),values=scaled[sl],raw_values=raw[sl],mask=mask[sl],times=times,naive_times=times+HOUR,segments=segments)
        for h in meta['horizons']:np.save(numeric/f'{s}_H{h}_starts.npy',valid_windows(times,segments,mask[sl],360,h,HOUR))
    write_json(numeric/'manifest.json',meta);seal(numeric,{'kind':'synthetic_test_only'})
    # Independent fictional forecasts; no station future observations enter.
    vt=t;forecasts=np.tile(np.linspace(-3,20,7),(len(vt),1)).astype(np.float32)
    ids=np.broadcast_to(np.arange(7),forecasts.shape).copy()
    np.savez(gfs/'forecasts.npz',valid_times=vt,values=forecasts,mask=np.ones_like(ids,bool),template_indices=ids,
       left_source_times=vt-10*60*10**9,right_source_times=vt+50*60*10**9,left_batches=np.zeros(len(vt),int),right_batches=np.zeros(len(vt),int))
    ev=rng.normal(size=(7,384)).astype(np.float32);ev/=np.linalg.norm(ev,axis=1,keepdims=True)
    np.save(gfs/'embeddings.npy',ev);np.save(gfs/'queries.npy',rng.normal(size=(21,12,384)).astype(np.float32))
    (gfs/'text_templates.jsonl').write_text(''.join(json.dumps({'text':f'Historical GFS temperature {k} nominal lead {24*(k+1)} hours'})+'\n' for k in range(7)))
    write_json(gfs/'manifest.json',{'model':'gfs_global','requested_coordinate':{'latitude':50.91,'longitude':11.56},
       'batches':[{'sha256':'synthetic','url':'https://synthetic.invalid/not-a-real-source','response_grid':{'latitude':50.9,'longitude':11.6}}]})
    seal(gfs,{'kind':'synthetic_forecast_test_only'})
    return numeric,gfs
