import numpy as np
import pytest
import torch
from pivot_stage2_v2.configurations import configuration,MAIN,ABLATIONS
from pivot_stage2_v2.models import build_model
from pivot_stage2_v2.train import run
from pivot_stage2_v2.suite import plans
from pivot_stage2.metrics import masked_mse

@pytest.mark.parametrize('horizon',[24,72,168,336])
@pytest.mark.parametrize('raw',[False,True])
def test_batch_one_mimo_all_horizons_backward(horizon,raw):
    cfg={'name':'gfs_raw' if raw else 'pivot','lookback':360,'horizon':horizon,'channels':21,'dim':16,'heads':4,'layers':1,'patch':16,'stride':8,'dropout':0}
    m=build_model(cfg);x=torch.randn(1,360,21);mask=torch.ones_like(x,dtype=torch.bool);mask[:,:,-1]=False
    ext=torch.randn(1,horizon,7);em=torch.ones_like(ext,dtype=torch.bool);em[:,-1]=False
    text=torch.randn(1,21,4,384);tm=torch.ones(1,21,4,dtype=torch.bool);tm[:,-1]=False
    y=m(x_num=x,x_mask=mask,gfs_num=ext,gfs_mask=em,text_embeddings=text,text_mask=tm)
    assert y.shape==(1,horizon,21) and torch.isfinite(y).all()
    loss=masked_mse(y,torch.zeros_like(y),torch.ones_like(y,dtype=torch.bool));loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

@pytest.mark.parametrize('variant',['full_pivot','gfs_raw'])
def test_single_step_checkpoint_and_resume(bundle,tmp_path,variant):
    n,g=bundle;c=configuration(24,variant);c.update(data_directory=str(n),device='cpu')
    c['retrieval']['gfs_directory']=str(g)
    first=run(c,tmp_path/'run1',smoke=True)
    second=run(c,tmp_path/'run2',smoke=True,resume=tmp_path/'run1/latest.pt')
    for s in (first,second):assert s['optimizer_steps']==1 and s['checkpoint_roundtrip'] and s['no_text_finite']

def test_frozen_matrix_default_and_independent_ablations():
    assert len(plans('main'))==63 and len(plans('ablations'))==108
    assert len(plans('pilot'))==21 and all(j['seed']==2024 and not j['formal_training'] for j in plans('pilot'))
    assert {j['horizon'] for j in plans('main')}=={24,72,168}
    assert 336 in {j['horizon'] for j in plans('main',True)}
    assert not configuration(24,'without_ctsr')['model']['use_ctsr']
    assert configuration(24,'without_str')['retrieval']['strategy']=='mean'
    assert configuration(24,'without_temporal')['retrieval']['strategy']=='semantic'
    for v in ('timezone_berlin','timezone_utc','timezone_cet_common'):assert configuration(24,v)['retrieval']['common_sensitivity']

def test_missing_text_time_features_cannot_inject_information():
    m=build_model({'name':'pivot','lookback':20,'horizon':24,'channels':21,'dim':16,'heads':4,'patch':8,'stride':4,'dropout':0}).eval()
    x=torch.randn(1,20,21);mask=torch.ones_like(x,dtype=torch.bool)
    text=torch.zeros(1,21,4,384);tm=torch.zeros(1,21,4,dtype=torch.bool)
    with torch.inference_mode():
        a=m(x_num=x,x_mask=mask,text_embeddings=text,text_mask=tm,text_time_features=torch.full((1,21,4,3),float('nan')))
        b=m(x_num=x,x_mask=mask)
    assert torch.equal(a,b) and torch.isfinite(a).all()
