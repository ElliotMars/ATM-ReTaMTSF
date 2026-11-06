"""Full native clock mapping and original-order anomaly evidence."""
import numpy as np
import pandas as pd
from pypdf import PdfReader
from .common import ROOT,DATA,ART,HOUR,sha,read_json,write_json,new_directory,seal,verify
from .timezone import RAW,to_utc,last_sunday

def main():
    directory=DATA/'native_time_mapping'
    if (directory/'_SUCCESS.json').exists():verify(directory);return
    tables=[];sources=[];events=[]
    for p in sorted((RAW/'raw_data').glob('*.parquet')):
        t=pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(p,columns=['Date Time'])['Date Time']))
        sources.append({'path':str(p),'sha256':sha(p),'rows':len(t)})
        for i in np.flatnonzero(np.diff(t.asi8)!=600*10**9)+1:
            events.append({'source':p.name,'right_source_row':int(i),'left_naive':str(t[i-1]),'right_naive':str(t[i]),
              'delta_seconds':int((t.asi8[i]-t.asi8[i-1])/10**9),'at_expected_DST_date':bool(any(t[i].normalize()==last_sunday(t[i].year,m) for m in (3,10))),
              'cause':'unresolved_source_anomaly_not_automatically_attributed_to_DST'})
        tables.append(pd.DataFrame({'source_file':p.name,'source_row':np.arange(len(t)),
          'original_naive_timestamp':t,'fixed_CET_UTC':to_utc(t,'fixed_cet'),'Europe_Berlin_UTC_or_NaT':to_utc(t,'berlin'),'UTC_hypothesis':to_utc(t,'utc')}))
    frame=pd.concat(tables,ignore_index=True);t=pd.DatetimeIndex(frame.original_naive_timestamp)
    examples=[]
    for year in range(2019,2026):
        wanted=[pd.Timestamp(f'{year}-01-15 12:10'),pd.Timestamp(f'{year}-06-15 12:10')]
        for month in (3,10):wanted.extend(last_sunday(year,month)+pd.Timedelta(hours=h,minutes=10) for h in (1,2,3))
        for target in wanted:
            indices=np.flatnonzero(t==target)
            if not len(indices):continue
            row=frame.iloc[int(indices[0])]
            examples.append({k:(None if pd.isna(v) else str(v)) for k,v in row.to_dict().items()})
    assert len(examples)>=20
    with new_directory(directory):
        frame.to_parquet(directory/'native_mapping.parquet',index=False)
        write_json(directory/'manifest.json',{'original_rows':len(frame),'sources':sources,'fixed_CET_rule':'UTC = original naive - 1 hour, year round',
          'native_original_order_retained':True,'raw_duplicate_timestamp_extras':int(t.duplicated().sum()),'raw_clock_backwards':int((np.diff(t.asi8)<0).sum()),
          'raw_anomaly_policy':'retain and report; use already accepted N2 duplicate resolution/segments for training, never invent timestamp repairs',
          'berlin_ambiguous_or_nonexistent':int(frame.Europe_Berlin_UTC_or_NaT.isna().sum())})
        write_json(directory/'anomalous_original_edges.json',events)
        write_json(directory/'real_mapping_examples.json',examples)
        seal(directory,{'kind':'all_native_source_clock_mapping','rows':len(frame)})
    evidence=read_json(ART/'source_evidence.json')
    historical=evidence.get('historical_mez_pdf',{})
    if historical.get('path'):
        pages=PdfReader(historical['path']).pages
        # Context evidence only. Actual roof source identity and its own
        # station manual determine the decision, not a different Jena station.
        text='\n'.join(p.extract_text() or '' for p in pages)
        (ART/'historical_heimann_context.txt').write_text(text)
    print('Full native mapping',len(frame),'real seasonal examples',len(examples),flush=True)
if __name__=='__main__':main()
