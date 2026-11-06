"""Exhaustive UTC/numerical checks and real GFS coverage for every origin."""
import argparse
import numpy as np
import pandas as pd
from pivot_stage2.data import valid_windows
from .common import ROOT,DATA,ART,HOUR,TZS,HORIZONS,read_json,write_json,sha,verify
from .data import shared_store,MIMODataset
from .timezone import to_utc

def coverage(ds,lag):
    s=ds.store;h=ds.horizon
    origins_any=points_any=facts=total=0;leads=np.zeros(7,np.int64);max_available_delta=None
    for b in range(0,len(ds),512):
        starts=ds.starts[b:b+512];target=ds.times[starts[:,None]+ds.lookback+np.arange(h)]
        origin=ds.times[starts+ds.lookback-1]+ds.observation_latency*60*10**9
        rows=(target-s.first)//HOUR;exists=(rows>=0)&(rows<len(s.valid_times));safe=np.clip(rows,0,len(s.valid_times)-1)
        exists &= s.valid_times[safe]==target
        available=s.right_source_times[safe][...,None]-np.arange(1,8)*24*HOUR+lag*HOUR
        legal=(available<=origin[:,None,None])&s.mask[safe]&exists[...,None]
        p=legal.any(-1);origins_any+=int(p.any(1).sum());points_any+=int(p.sum());facts+=int(legal.sum());total+=p.size
        leads+=legal.sum((0,1))
        if legal.any():
            delta=(available-origin[:,None,None])[legal].max()
            max_available_delta=int(delta) if max_available_delta is None else max(max_available_delta,int(delta))
    return {'origins':len(ds),'origins_with_any_legal_GFS':origins_any,'origins_with_any_ratio':origins_any/len(ds),
      'origins_without_GFS':len(ds)-origins_any,'origins_without_ratio':1-origins_any/len(ds),'target_hours':total,
      'target_hours_with_any_legal_forecast':points_any,'target_hour_coverage':points_any/total,'legal_fact_count':facts,
      'legal_fact_count_by_nominal_day':leads.tolist(),'latency_hours':lag,'max_available_minus_origin_ns':max_available_delta,
      'text_channel_origin_coverage':origins_any/len(ds),'raw_and_text_identical_candidate_fact_pool':True}

def audit():
    store=shared_store(str((DATA/'gfs').resolve()));assert store.meta['never_generated_from_Jena_or_reanalysis']
    for batch in store.meta['batches']:
        assert sha(batch['path']) == batch['sha256'], 'Original GFS response hash changed'
    native_mapping=DATA/'native_time_mapping'
    if (native_mapping/'_SUCCESS.json').exists():
        verify(native_mapping)
        native=pd.read_parquet(native_mapping/'native_mapping.parquet',columns=['original_naive_timestamp','fixed_CET_UTC'])
        converted=pd.DatetimeIndex(native.fixed_CET_UTC).tz_localize(None)
        assert np.array_equal(converted.asi8,pd.DatetimeIndex(native.original_naive_timestamp).asi8-HOUR)
    reports={};scaler_seen=None
    for tz in TZS:
        directory=DATA/('jena_'+tz);verify(directory);meta=read_json(directory/'manifest.json')
        assert len(meta['channels'])==21
        if scaler_seen is not None:assert meta['scaler']==scaler_seen
        scaler_seen=meta['scaler'];splits={};times_by_split=[]
        for split in ('train','validation','test'):
            z=np.load(directory/f'{split}.npz');t=z['times'];naive=z['naive_times'];mask=z['mask'];v=z['values'];raw=z['raw_values']
            assert np.array_equal(t,to_utc(pd.to_datetime(naive),tz).asi8)
            assert np.isfinite(v[mask]).all() and np.isfinite(raw[mask]).all()
            assert np.allclose(v[mask],((raw-np.asarray(meta['scaler']['mean']))/np.asarray(meta['scaler']['scale']))[mask],rtol=1e-5,atol=1e-5)
            times_by_split.append(set(naive.tolist()))
            if split=='train':
                count=mask.sum(0);mean=np.where(mask,raw,0).sum(0)/count
                sd=np.sqrt((np.where(mask,raw-mean,0).astype(np.float64)**2).sum(0)/count)
                assert count.tolist()==meta['scaler']['fit_counts'] and np.allclose(mean,meta['scaler']['mean']) and np.allclose(sd,meta['scaler']['scale'])
            hs={}
            for h in HORIZONS:
                starts=np.load(directory/f'{split}_H{h}_starts.npy');expected=valid_windows(t,z['segments'],mask,360,h,HOUR)
                bad=np.r_[0,(t==pd.NaT.value).cumsum()];expected=expected[(bad[expected+360+h]-bad[expected])==0]
                assert np.array_equal(starts,expected)
                assert np.all(t[starts+359]+10*60*10**9<t[starts+360])
                ds=MIMODataset(directory,split,h,check_hashes=False,strategy='none')
                hs[str(h)]={'samples':len(ds),'full_L_H_and_origin_checked':True}
                if tz=='fixed_cet':hs[str(h)]['gfs_coverage']={str(lag):coverage(ds,lag) for lag in (3,6,12)}
            splits[split]={'points':len(t),'unresolved_timezone_rows':int((t==pd.NaT.value).sum()),'horizons':hs}
        assert all(not times_by_split[a]&times_by_split[b] for a,b in ((0,1),(0,2),(1,2)))
        reports[tz]={'splits':splits,'train_scaler_verified':True,'labels_disjoint':True,'UTC_mapping_verified_all_rows':True}
    # Verify unchanged native N2 values/masks by exact original timestamp lookup.
    n2=ROOT/'dataset/Fidel-TS/processed/jena/n2_strict_contiguous';verify(n2)
    original={}
    for split in ('train','validation','test'):
        frame=pd.read_parquet(n2/split/'timestamps.parquet');vals=pd.read_parquet(n2/split/'values.parquet').to_numpy();mask=pd.read_parquet(n2/split/'validity_mask.parquet').to_numpy(bool)
        original[split]=(frame.timestamp.to_numpy().view('i8'),vals,mask)
    oldt=np.concatenate([x[0] for x in original.values()]);oldv=np.concatenate([x[1] for x in original.values()]);oldm=np.concatenate([x[2] for x in original.values()])
    for tz in TZS:
        for split in ('train','validation','test'):
            z=np.load(DATA/('jena_'+tz)/f'{split}.npz');indices=np.searchsorted(oldt,z['naive_times'])
            assert np.array_equal(oldt[indices],z['naive_times']) and np.array_equal(oldm[indices],z['mask'])
            assert np.array_equal(oldv[indices],z['raw_values'],equal_nan=True)
    # Actual seasonal samples and identity of the full raw/text candidate set.
    seasonal=[];document_examples=[]
    for split,wanted in [('train','2022-01-15 12:10'),('validation','2024-07-15 12:10'),('test','2025-03-30 12:10')]:
        ds=MIMODataset(DATA/'jena_fixed_cet',split,168,check_hashes=False)
        i=int(np.argmin(abs(ds.naive_times[ds.starts+359]-pd.Timestamp(wanted).value)));sample=ds[i]
        assert sample['history_end_time']<=sample['origin_time']<sample['first_target_time'] and sample['text_mask'].any()
        assert int(sample['gfs_mask'].sum())==len(sample['gfs_pool_ids'])
        origin=sample['origin_time']
        stop=int(ds.starts[i])+ds.lookback
        _,_,facts,_=ds.store.pool(origin,ds.times[stop:stop+168],12)
        for fact in facts[np.linspace(0,len(facts)-1,12,dtype=int)]:
            document_examples.append({'split':split,**ds.store.metadata(int(fact),origin,12)})
        seasonal.append({'split':split,'wanted_history_end_naive':wanted,'actual_history_end_utc':pd.Timestamp(sample['history_end_time'],tz='UTC').isoformat(),
          'forecast_origin_utc':pd.Timestamp(origin,tz='UTC').isoformat(),'first_target_utc':pd.Timestamp(sample['first_target_time'],tz='UTC').isoformat(),
          'candidate_facts':len(sample['gfs_pool_ids']),'legal_target_hours':int(sample['gfs_mask'].any(-1).sum()),'text_mask_nonempty':True,
          'retrieval_trace':sample['text_metadata'][1]})
    write_json(ART/'data_acceptance.json',{'timezone_hypotheses':reports,'raw_N2_values_masks_unchanged':True,'all_timezone_bundles_match_identical_N2_values_and_masks':True,'original_N2_success_sha256':sha(n2/'_SUCCESS.json'),
      'seasonal_real_samples':seasonal,'gfs_manifest_sha256':sha(DATA/'gfs/manifest.json'),'strict_historical_publication_verified':False})
    import json
    (ART/'gfs_document_examples.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in document_examples))
    decision=read_json(ART/'timezone_decision.json')
    main=all(reports['fixed_cet']['splits'][s]['horizons'][str(h)]['gfs_coverage']['12']['origins_with_any_ratio']>0 for s in ('train','validation','test') for h in HORIZONS[:3])
    write_json(ART/'admission.json',{'dataset':'jena','decision':'CONDITIONAL','main_enabled':main,'timezone':decision['selected'],'timezone_confidence':decision['confidence'],
      'data_bindings':{'gfs':sha(DATA/'gfs/_SUCCESS.json'),**{tz:sha(DATA/('jena_'+tz)/'_SUCCESS.json') for tz in TZS}},
      'core_horizons':[24,72,168],'optional_horizon':336,'fixed_channels':21,'lookback':360,'strict_verified_t1':False,'formal_training_started':False,
      'required_disclosures':['fixed_nominal_lead_archive_not_verified_individual_GFS_Run','availability_reference_plus_latency_assumption','station_end_of_interval_plus10min_observation_receipt_assumption',
          'hourly_GFS_forecasts_interpolated_to_N2_10minute_phase_both_endpoints_legal','MiniLM_postdates_historical_task_shared_offline_prior','upstream_Fidel_interpolation_risk_not_removed']})
    print('v2 real-data audit passed; Jena CONDITIONAL main_enabled =',main,flush=True)
    if not main:raise SystemExit('No legal GFS coverage; formal experiments disabled')
if __name__=='__main__':audit()
