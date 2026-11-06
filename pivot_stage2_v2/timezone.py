"""Evidence-based timezone diagnostics; never compare forecast errors."""
from datetime import timezone, timedelta
import calendar
import io
import zipfile
import numpy as np
import pandas as pd
from pypdf import PdfReader
from .common import ROOT, DATA, ART, HOUR, sha, read_json, write_json

RAW=ROOT/'dataset/Fidel-TS/raw/Jena_Atmospheric_Physics'

def to_utc(times, hypothesis='fixed_cet'):
    times=pd.DatetimeIndex(times)
    if times.tz is not None: raise ValueError('Expected original naive source timestamps')
    if hypothesis=='fixed_cet': return times.tz_localize(timezone(timedelta(hours=1))).tz_convert('UTC')
    if hypothesis=='utc': return times.tz_localize('UTC')
    if hypothesis=='berlin': return times.tz_localize('Europe/Berlin',ambiguous='NaT',nonexistent='NaT').tz_convert('UTC')
    raise ValueError('Unknown timezone hypothesis')

def last_sunday(year,month):
    last=pd.Timestamp(year,month,calendar.monthrange(year,month)[1])
    return last-pd.Timedelta(days=(last.dayofweek+1)%7)

def diagnose():
    inputs=[];frames=[];reports=[]
    for p in sorted((RAW/'raw_data').glob('*.parquet')):
        f=pd.read_parquet(p)
        t=pd.DatetimeIndex(pd.to_datetime(f['Date Time']))
        frames.append((f,t))
        delta=np.diff(t.asi8)
        inputs.append({'path':str(p),'sha256':sha(p),'rows':len(t),'start':str(t.min()),'end':str(t.max()),
            'duplicates':int(t.duplicated().sum()),'backward_edges':int((delta<0).sum()),'non_10minute_edges':int((delta!=600*10**9).sum())})
    for year in range(2019,2026):
        for month in (3,10):
            day=last_sunday(year,month); ts=[]
            for f,t in frames:
                ts.extend(t[(t>=day-pd.Timedelta(days=1))&(t<day+pd.Timedelta(days=2))])
            t=pd.DatetimeIndex(ts);delta=np.diff(t.asi8)
            reports.append({'year':year,'transition_day':str(day.date()),'raw_points':len(t),'no_source_period':not len(t),
                'duplicates':int(t.duplicated().sum()),'backward_edges':int((delta<0).sum()),'non_10minute_edges':int((delta!=600*10**9).sum()),
                'berlin_unresolved_local_times':int(to_utc(t,'berlin').isna().sum()),'fixed_cet_unresolved':0,'utc_unresolved':0})
    primary=pd.read_parquet(RAW/'time_series/weather_large.parquet')
    pt=pd.DatetimeIndex(primary['Date Time'])
    combined=pd.DatetimeIndex(np.concatenate([t.to_numpy() for _,t in frames]))
    relation={'raw_total':len(combined),'processed_total':len(pt),'exact_timestamp_sequence_preserved':np.array_equal(combined.asi8,pt.asi8),
      'raw_berlin_unresolved_all_years':int(to_utc(combined,'berlin').isna().sum()),'no_time_zone_field_in_local_metadata':True,
      'warning':'Continuous labels alone do not establish timezone; raw_data may itself be provider-normalized'}
    if relation['exact_timestamp_sequence_preserved']:
        columns=[c for c in primary if c!='Date Time']
        rv=np.concatenate([f[columns].to_numpy() for f,_ in frames])
        pv=primary[columns].to_numpy()
        equal=(rv==pv)|(np.isnan(rv)&np.isnan(pv))
        relation['changed_value_cells_by_channel']={c:int(n) for c,n in zip(columns,(~equal).sum(0))}
        relation['raw_invalid_to_processed_valid_cells']=int(((~np.isfinite(rv)|(rv==-9999))&np.isfinite(pv)&(pv!=-9999)).sum())
    write_json(ART/'timezone_diagnostics.json',{'raw_inputs':inputs,'dst_windows':reports,'raw_vs_processed':relation})
    return frames

def station_trace(frames):
    evidence=read_json(ART/'source_evidence.json'); matches=[];extracted=[]
    for name,row in evidence.items():
        if name.startswith('station_documentation_') and row.get('path'):
            p=row['path'];text='\n'.join(page.extract_text() or '' for page in PdfReader(p).pages)
            target=ART/(name+'.txt');target.write_text(text)
            extracted.append({'name':name,'pdf_sha256':sha(p),'text_file':str(target),'text_sha256':sha(target),
              'mentions_MEZ': 'MEZ' in text,'mentions_end_of_averaging_interval': 'Mittelungsintervalls' in text or 'averaging interval' in text})
    f,t=frames[-1]; local=f.copy();local.index=t
    # Numeric fingerprint links the actual Fidel source to this roof archive,
    # rather than applying another station's MEZ convention by proximity.
    for name in ('roof_2024','roof_2019a'):
        row=evidence.get(name,{})
        if not row.get('path'):continue
        with zipfile.ZipFile(row['path']) as z:
            member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
            roof=pd.read_csv(io.BytesIO(z.read(member)),encoding='latin1')
        rt=pd.DatetimeIndex(pd.to_datetime(roof['Date Time'],dayfirst=True))
        roof.index=rt
        source=next((frame.set_index(tt) for frame,tt in frames if len(rt.intersection(tt))>100),None)
        if source is None:continue
        cols=['p (mbar)','T (degC)','rh (%)','CO2 (ppm)']
        joined=rt.intersection(source.index)
        good=source.loc[joined,cols]
        if good.index.duplicated().any():good=good[~good.index.duplicated(keep=False)];joined=good.index
        actual=roof.loc[joined,cols]
        valid=np.isfinite(good.to_numpy())&np.isfinite(actual.to_numpy())
        diff=np.abs(good.to_numpy()-actual.to_numpy())
        matches.append({'archive':name,'source_url':row['url'],'archive_sha256':row['sha256'],'csv_member':member,
            'overlap_timestamps':len(joined),'compared_channels':cols,'finite_cells':int(valid.sum()),
            'exact_equal_cells':int(((diff==0)&valid).sum()),'max_absolute_difference':float(diff[valid].max()),
            'same_naive_time_fingerprint_verified':bool((diff[valid]==0).all())})
    decision={'selected':'fixed_cet','utc_offset_hours':1,'dst':False,'confidence':'high_source_supported_with_historical_clock_config_caveat',
      'official_source_MEZ_and_end_of_interval':any(x['mentions_MEZ'] and x['mentions_end_of_averaging_interval'] for x in extracted),
      'numeric_station_trace':matches,'documentation':extracted,'not_selected_by_model_performance':True,
      'unverified':'Historical logger configuration and reception times for every year are not independently archived; retain alternative timezone sensitivities'}
    if not any(x['same_naive_time_fingerprint_verified'] for x in matches):
        decision['confidence']='conditional_working_assumption_station_fingerprint_incomplete'
    if not decision['official_source_MEZ_and_end_of_interval']:
        decision['confidence']='conditional_working_assumption_no_direct_clock_document'
    write_json(ART/'timezone_decision.json',decision)
    print('Timezone decision',decision['selected'],decision['confidence'],flush=True)

if __name__=='__main__':station_trace(diagnose())
