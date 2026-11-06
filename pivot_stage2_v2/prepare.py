"""Build new Jena splits and pointwise forecast facts without touching N2/v1."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from pivot_stage2.data import valid_windows
from pivot_stage2.prepare import fit_scaler
from pivot_stage2.encoding import TextEncoder
from .common import ROOT,DATA,ART,HOUR,TZS,HORIZONS,sha,digest,read_json,write_json,new_directory,seal,verify
from .timezone import to_utc

N2=ROOT/'dataset/Fidel-TS/processed/jena/n2_strict_contiguous'

def numeric():
    verify(N2)
    channels=read_json(N2/'channels.json')['channels'];assert len(channels)==21
    arrays=[]
    for split in ('train','validation','test'):
        t=pd.read_parquet(N2/split/'timestamps.parquet')
        v=pd.read_parquet(N2/split/'values.parquet');m=pd.read_parquet(N2/split/'validity_mask.parquet')
        assert list(v)==list(m)==channels
        arrays.append((t.timestamp.to_numpy(),v.to_numpy(),m.to_numpy(dtype=bool),t.native_segment_id.to_numpy()))
    naive=pd.DatetimeIndex(np.concatenate([a[0] for a in arrays]));values=np.concatenate([a[1] for a in arrays]);mask=np.concatenate([a[2] for a in arrays]);segments=np.concatenate([a[3] for a in arrays])
    keep=(naive>=pd.Timestamp('2021-04-01'))&(naive<pd.Timestamp('2025-07-01'))
    naive,values,mask,segments=naive[keep],values[keep],mask[keep],segments[keep]
    ids=np.where(naive<pd.Timestamp('2024-01-01'),0,np.where(naive<pd.Timestamp('2025-01-01'),1,2))
    scaler=fit_scaler(values,mask,ids==0)
    scaled=((values-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])).astype(np.float32)
    for tz in TZS:
        directory=DATA/('jena_'+tz)
        if (directory/'_SUCCESS.json').exists():verify(directory);continue
        utc=to_utc(naive,tz).asi8
        with new_directory(directory):
            stats={}
            for si,split in enumerate(('train','validation','test')):
                select=ids==si;t=utc[select];m=mask[select];s=segments[select]
                np.savez(directory/(split+'.npz'),values=scaled[select],raw_values=values[select],mask=m,times=t,naive_times=naive.asi8[select],segments=s)
                horizons={}
                for h in HORIZONS:
                    starts=valid_windows(t,s,m,360,h,HOUR)
                    # Explicitly reject all unresolved Berlin local times.
                    invalid=(t==pd.NaT.value).astype(np.int64);p=np.r_[0,invalid.cumsum()]
                    starts=starts[(p[starts+360+h]-p[starts])==0]
                    np.save(directory/f'{split}_H{h}_starts.npy',starts)
                    horizons[str(h)]=len(starts)
                stats[split]={'points':int(select.sum()),'unresolved_timezone_rows':int((t==pd.NaT.value).sum()),'horizon_samples':horizons,
                  'naive_start':str(naive[select][0]),'naive_end':str(naive[select][-1]),'utc_first_resolved':str(pd.Timestamp(t[t!=pd.NaT.value][0],tz='UTC')),
                  'utc_last_resolved':str(pd.Timestamp(t[t!=pd.NaT.value][-1],tz='UTC')),'missing_cell_ratio':float((~m).mean())}
            meta={'dataset':'jena','version':'stage2_v2','channels':channels,'lookback':360,'horizons':list(HORIZONS),'timezone_hypothesis':tz,
              'selected_timezone':tz=='fixed_cet','scaler':scaler,'splits':stats,'source_n2':str(N2),'source_n2_success_sha256':sha(N2/'_SUCCESS.json'),
              'value_policy':'exact N2 raw values and masks preserved; v2 scaler refit using v2 Train valid cells only',
              'observation_semantics':'10-minute interval ending at timestamp, per station documentation',
              'observation_latency_minutes':10,'observation_availability_policy':'interval_end_plus_10min_polling_assumption_not_verified_receipt',
              'origin':'last_history_end_utc + observation_latency; strictly before first_target; never first target cutoff',
              'split_interpretation':'original station calendar bounds mapped through each timezone hypothesis; no targets shared across splits',
              'upstream_interpolation_risk_remaining':True}
            write_json(directory/'manifest.json',meta);seal(directory,{'kind':'jena_utc_numeric','timezone':tz})
    print('Numerical UTC bundles ready',flush=True)

def template(value,k):
    # Date/grid/provenance are retained on each fact; semantic wording is
    # deliberately time-independent so identical weather statements share vectors.
    return f'Historical GFS forecast for Jena: the predicted 2-meter temperature is {value:.1f} °C, using a forecast with {24*k}-hour nominal lead time.'

def gfs():
    directory=DATA/'gfs'
    if (directory/'_SUCCESS.json').exists():verify(directory);return
    start=time.perf_counter();acq=read_json(ART/'gfs_acquisition.json');times=[];vals=[];batch=[]
    for bi,b in enumerate(acq['batches']):
        assert sha(b['path'])==b['sha256']
        r=read_json(b['path']);ts=np.asarray(r['hourly']['time'],np.int64)*10**9
        v=np.array([[np.nan if x is None else x for x in r['hourly'][f'temperature_2m_previous_day{k}']] for k in range(1,8)],dtype=np.float64).T
        assert v.shape==(len(ts),7)
        times.append(ts);vals.append(v);batch.append(np.full(len(ts),bi,np.int16))
    times=np.concatenate(times);values=np.concatenate(vals);batches=np.concatenate(batch)
    order=np.argsort(times,kind='stable');times,values,batches=times[order],values[order],batches[order]
    assert np.all(np.diff(times)==HOUR),'Incomplete/duplicate GFS response timeline'
    # N2 samples are at :10. Interpolate FORECASTS, never observations.
    # Both endpoints must exist and be as-of legal. Shared raw/text values
    # are rounded to the source's 0.1°C reporting precision.
    aligned_times=times[:-1]+10*60*10**9
    aligned_values=np.round(values[:-1]*5/6+values[1:]/6,1).astype(np.float32)
    masks=np.isfinite(values[:-1])&np.isfinite(values[1:])
    aligned_values[~masks]=np.nan
    descriptors=sorted({template(float(aligned_values[i,k]),k+1) for i,k in np.argwhere(masks)})
    by_text={t:i for i,t in enumerate(descriptors)}
    indices=np.full(masks.shape,-1,np.int32)
    for i,k in np.argwhere(masks):indices[i,k]=by_text[template(float(aligned_values[i,k]),k+1)]
    with new_directory(directory):
        np.savez(directory/'forecasts.npz',valid_times=aligned_times,values=aligned_values,mask=masks,template_indices=indices,
          left_source_times=times[:-1],right_source_times=times[1:],left_batches=batches[:-1],right_batches=batches[1:])
        with (directory/'text_templates.jsonl').open('w') as f:
            import json
            for i,t in enumerate(descriptors):f.write(json.dumps({'template_id':i,'text':t},ensure_ascii=False)+'\n')
        encoder=TextEncoder(ROOT/'dataset/PIVOT/stage2_v1/stocknet/text')
        assert encoder.kind=='frozen_all_MiniLM_L6_v2'
        import torch
        torch.set_num_threads(8)
        vectors=encoder.encode(descriptors);np.save(directory/'embeddings.npy',vectors)
        static=read_json(ROOT/'dataset/Fidel-TS/raw/Jena_Atmospheric_Physics/static_info.json')['channel_info']['weather_large']
        aliases=read_json(ROOT/'configs/fidel_ts/sources.json')['datasets']['jena']['channel_aliases']
        channels=read_json(DATA/'jena_fixed_cet/manifest.json')['channels']
        queries=[]
        for c in channels:
            # Exact source channel names may contain historical encoding damage.
            canonical=aliases.get(c,c)
            if canonical not in static:
                raise ValueError('Missing source channel description: '+c)
            desc=static[canonical]
            for temp in ('cold below 5 degrees Celsius','mild 5 to 20 degrees Celsius','warm above 20 degrees Celsius'):
                for humidity in ('relative humidity below 60 percent','relative humidity at least 60 percent'):
                    for pressure in ('air pressure below 1000 millibar','air pressure at least 1000 millibar'):
                        queries.append(f'Jena weather prediction of {c}: {desc[:150]}. Historical air temperature {temp}; {humidity}; {pressure}. Relevant GFS 2-meter temperature forecast.')
        np.save(directory/'queries.npy',encoder.encode(queries).reshape(21,12,384))
        write_json(directory/'query_templates.json',{'channels':channels,'queries_per_channel':12,'texts':queries,'all_channel_descriptions_resolved':True})
        meta={'model':'gfs_global','api':'https://previous-runs-api.open-meteo.com/v1/forecast','archive_kind':acq['archive_kind'],
          'requested_coordinate':{'latitude':50.91,'longitude':11.56,'elevation_m':220},'batches':acq['batches'],
          'units':'degree_Celsius','nominal_leads_hours':[24*k for k in range(1,8)],'verified_run_time':None,'verified_publication_time':None,
          'availability':'max(bracketing_valid_hour - k*24h) + configured_latency; nominal reference is not a verified Run',
          'alignment':'linear interpolation of two GFS forecasts to source :10 phase; both endpoints legal; rounded 0.1C identically in all methods',
          'raw_hourly_points':len(times),'raw_nonnull_values_by_k':np.isfinite(values).sum(0).tolist(),'aligned_nonnull_values_by_k':masks.sum(0).tolist(),
          'unique_text_templates':len(descriptors),'forecast_fact_count':int(masks.sum()),'encoder':encoder.meta,
          'embedding_text':'complete temperature/nominal-lead semantic template; per-fact UTC valid time and source metadata stored in columnar index and retrieval trace',
          'never_generated_from_Jena_or_reanalysis':True,'preparation_seconds':time.perf_counter()-start}
        write_json(directory/'manifest.json',meta);seal(directory,{'kind':'real_gfs_fixed_lead_forecast_facts','source_acquisition_sha256':sha(ART/'gfs_acquisition.json')})
    print('Real GFS templates/embeddings ready',meta['forecast_fact_count'],len(descriptors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['numeric','gfs','all'],default='all');a=p.parse_args()
    if a.part in ('numeric','all'):numeric()
    if a.part in ('gfs','all'):gfs()
