import numpy as np
import pytest
import torch
from pivot_stage2_v2.data import MIMODataset,query_bin
from pivot_stage2_v2.common import HOUR
from pivot_stage2_v2.models import build_model

@pytest.mark.parametrize('horizon',[24,72,168,336])
@pytest.mark.parametrize('lag',[3,6,12])
def test_real_permission_rule_and_late_seventh_day_mask(bundle,horizon,lag):
    n,g=bundle;ds=MIMODataset(n,'train',horizon,gfs_directory=g,lag_hours=lag);s=ds[0]
    assert s['history_end_time']<s['origin_time']<s['first_target_time']
    assert s['x_num'].shape==(360,21) and s['y'].shape==(horizon,21)
    assert len(s['gfs_pool_ids'])==int(s['gfs_mask'].sum())
    # Both source endpoints are required. Maximum usable target hour is
    # 168 - latency - 1 due to the :10 interpolation phase.
    expect=min(horizon,168-lag-1)
    assert int(s['gfs_mask'].any(-1).sum())==expect
    for channel in s['text_metadata']:
        for d in channel:assert d['assumed_available_at_ns']<=s['origin_time']
    if horizon>=168:
        # This forecast would incorrectly pass if first_target were the cutoff.
        row=ds.store.time_index[int(s['target_times'][168-lag-1])]
        available=int(ds.store.right_source_times[row]-7*24*HOUR+lag*HOUR)
        assert s['origin_time']<available<s['first_target_time']
        assert not s['gfs_mask'][168-lag-1,6]

@pytest.mark.parametrize('strategy',['none','missing','mean','latest','semantic','semantic_temporal','random'])
def test_methods_share_pool_and_numerical_samples(bundle,strategy):
    n,g=bundle
    ref=MIMODataset(n,'train',72,strategy='none',gfs_directory=g)[0]
    s=MIMODataset(n,'train',72,strategy=strategy,gfs_directory=g)[0]
    assert s['sample_id']==ref['sample_id'] and s['gfs_pool_ids']==ref['gfs_pool_ids']
    for k in ('x_num','x_mask','y','y_mask','gfs_num','gfs_mask'):assert torch.equal(s[k],ref[k])
    assert bool(s['text_mask'].any())==(strategy not in ('none','missing'))

def test_future_station_labels_cannot_change_external_text_or_query(bundle):
    n,g=bundle;ds=MIMODataset(n,'train',72,gfs_directory=g)
    a=ds[0];stop=int(ds.starts[0])+360
    ds.values[stop:]+=500;ds.raw_values[stop:]-=500
    b=ds[0]
    for k in ('x_num','text_embeddings','text_mask','text_time_features','gfs_num','gfs_mask'):assert torch.equal(a[k],b[k])
    assert a['text_metadata']==b['text_metadata']

def test_absent_forecasts_mask_every_method_and_raw_model_is_identity(bundle):
    n,g=bundle;ds=MIMODataset(n,'train',24,gfs_directory=g);old=ds.store.mask.copy()
    try:
        ds.store.mask[:]=False;s=ds[0]
        assert not s['gfs_mask'].any() and not s['text_mask'].any() and not s['gfs_pool_ids']
        config={'name':'gfs_raw','lookback':360,'horizon':24,'channels':21,'dim':16,'heads':4,'layers':1,'dropout':0,'patch':16,'stride':8}
        model=build_model(config).eval();x={k:s[k][None] for k in ('x_num','x_mask','gfs_num','gfs_mask')}
        assert torch.equal(model(**x),model.backbone(x_num=x['x_num'],x_mask=x['x_mask']))
    finally:ds.store.mask[:]=old

def test_actual_missing_forecast_value_is_not_inferred_from_target_date(bundle):
    n,g=bundle;ds=MIMODataset(n,'train',24,gfs_directory=g);target=int(ds.times[360]);row=ds.store.time_index[target]
    old=ds.store.mask[row].copy()
    try:
        ds.store.mask[row]=False;s=ds[0]
        assert not s['gfs_mask'][0].any()
        assert all(not identifier.startswith(f'gfs:{target}:') for identifier in s['gfs_pool_ids'])
    finally:ds.store.mask[row]=old

def test_query_is_meteorological_and_masks_missing_history(bundle):
    n,g=bundle;ds=MIMODataset(n,'train',24,gfs_directory=g)
    b,stats=query_bin(ds.raw_values[:360],ds.mask[:360],ds.channels)
    assert 0<=b<12 and 5<stats['historical_temperature_c']<15
    assert 50<stats['historical_relative_humidity_percent']<80
    assert 980<stats['historical_pressure_mbar']<1010
    assert not stats['query_uses_future_y']
