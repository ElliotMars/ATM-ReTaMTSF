"""Finite, cached public requests. No credentials, paid services or reanalysis."""
import argparse
import time
from urllib.parse import urlencode
import requests
import pandas as pd
from .common import DATA, ART, sha, digest, read_json, write_json

class Client:
    def __init__(self):
        self.root=DATA/'sources'; self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'requests.json'
        self.state=read_json(self.path) if self.path.exists() else {'records':{},'attempts':[],'download_bytes':0}
        self.session=requests.Session()
        self.session.headers['User-Agent']='PIVOT-Jena-GFS-research/2.0'
    def get(self,url,limit=20*1024**2):
        key=digest(url); path=self.root/(key+'.raw')
        old=self.state['records'].get(key)
        if old:
            assert sha(path)==old['sha256'], 'Cached source changed'
            return path,old
        for retry in range(3):
            if self.state['download_bytes']+limit>300*1024**2: raise ValueError('v2 download budget exceeded')
            attempt={'url':url}; self.state['attempts'].append(attempt)
            try:
                with self.session.get(url,timeout=(15,90),stream=True) as r:
                    attempt['status']=r.status_code
                    r.raise_for_status()
                    body=bytearray()
                    for chunk in r.iter_content(65536):
                        body.extend(chunk); self.state['download_bytes']+=len(chunk)
                        if len(body)>limit: raise ValueError('Response limit exceeded')
                    path.write_bytes(body)
                    info={'url':url,'final_url':r.url,'bytes':len(body),'sha256':sha(path),'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat()}
                    self.state['records'][key]=info;write_json(self.path,self.state)
                    return path,info
            except (requests.RequestException,ValueError) as e:
                attempt['error']=str(e);write_json(self.path,self.state)
                if attempt.get('status') in (400,401,403,404,429) or retry==2 or isinstance(e,ValueError): raise
                time.sleep(2**retry)
        raise RuntimeError('Unreachable')

SOURCES={
 'fidel_card':'https://huggingface.co/datasets/fidel-ts/Jena_Atmospheric_Physics',
 'station':'https://www.bgc-jena.mpg.de/en/servicegroups/fieldexperiements/locations/beutenberg',
 'weather_home':'https://weather.bgc-jena.mpg.de/',
 'weather_download':'https://weather.bgc-jena.mpg.de/weather_data.html',
 'previous_runs_docs':'https://open-meteo.com/en/docs/previous-runs-api',
 'gfs_docs':'https://open-meteo.com/en/docs/gfs-api',
 'historical_mez_pdf':'https://www.bgc-jena.mpg.de/~martin.heimann/LSB/Talk_Winter-Challenge_Heimann.pdf',
 'roof_2024':'https://weather.bgc-jena.mpg.de/data/mpi_roof_2024.zip',
 'roof_2019a':'https://weather.bgc-jena.mpg.de/data/mpi_roof_2019a.zip',
}

def evidence():
    import re
    c=Client(); result={}
    for name,url in SOURCES.items():
        try:
            path,info=c.get(url)
            result[name]={'path':str(path),**info}
            if 'html' in name or name not in {'historical_mez_pdf','roof_2024','roof_2019a'}:
                html=path.read_text(errors='replace')
                if name=='weather_home':
                    from urllib.parse import urljoin
                    for href, label in re.findall(r'<a[^>]+href=[\"\']([^\"\']+)[\"\'][^>]*>(.*?)</a>',html,re.I|re.S):
                        if 'Documentation Weather Station' in label or 'Dokumentation Wetter Station' in label:
                            u=urljoin(url,href)
                            try:
                                p,r=c.get(u);result['station_documentation_'+str(len(result))]={'path':str(p),**r}
                            except Exception as exc: result['station_documentation_error']={'url':u,'error':str(exc)}
        except Exception as exc: result[name]={'url':url,'error':str(exc)}
    write_json(ART/'source_evidence.json',result)
    print('Official evidence captured',flush=True)

def forecasts():
    c=Client(); batches=[]
    # Two days before the first training label cover alternative UTC mappings.
    # The UTC sensitivity's last :10 target needs the next day's 00:00
    # forecast as its right interpolation endpoint. Fetch one padding day.
    dates=pd.date_range('2021-03-30','2025-07-02',freq='MS').union(pd.DatetimeIndex(['2021-03-30','2025-07-02'])).sort_values()
    starts=[pd.Timestamp('2021-03-30')]+[x for x in dates if x>pd.Timestamp('2021-03-30')]
    for i,start in enumerate(starts[:-1]):
        end=starts[i+1]-pd.Timedelta(days=1)
        params={'latitude':50.91,'longitude':11.56,'elevation':220,'hourly':','.join(f'temperature_2m_previous_day{k}' for k in range(1,8)),
          'models':'gfs_global','start_date':start.strftime('%Y-%m-%d'),'end_date':end.strftime('%Y-%m-%d'),'timezone':'UTC','timeformat':'unixtime','temperature_unit':'celsius'}
        url='https://previous-runs-api.open-meteo.com/v1/forecast?'+urlencode(params)
        path,info=c.get(url,4*1024**2)
        r=read_json(path)
        assert r['utc_offset_seconds']==0 and r['hourly_units']['time']=='unixtime'
        assert all(r['hourly_units'][f'temperature_2m_previous_day{k}']=='°C' for k in range(1,8))
        batches.append({'path':str(path),'requested':params,'response_grid':{k:r[k] for k in ('latitude','longitude','elevation','timezone')},**info})
        print('GFS',params['start_date'],params['end_date'],'hours',len(r['hourly']['time']),flush=True)
        time.sleep(.5)
    write_json(ART/'gfs_acquisition.json',{'model':'gfs_global','archive_kind':'fixed_nominal_lead_previous_days_not_single_run','batches':batches})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['evidence','gfs','all'],default='all');a=p.parse_args()
    if a.part in ('evidence','all'):evidence()
    if a.part in ('gfs','all'):forecasts()
