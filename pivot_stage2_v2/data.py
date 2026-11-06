"""Pointwise, shared raw/text GFS permissions and last-observation origins."""
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pivot_stage2.data import causal_fill, collate, HorizonSampler, loader
from .common import DATA,HOUR,digest,read_json,sha,verify

class GFSStore:
    def __init__(self,directory):
        from pathlib import Path
        self.directory=Path(directory);self.binding=verify(directory)
        self.meta=read_json(self.directory/'manifest.json')
        z=np.load(self.directory/'forecasts.npz')
        for name in z.files:setattr(self,name,z[name])
        self.embeddings=np.load(self.directory/'embeddings.npy',mmap_mode='r')
        self.queries=np.load(self.directory/'queries.npy')
        import json
        self.templates=[json.loads(x)['text'] for x in (self.directory/'text_templates.jsonl').read_text().splitlines()]
        self.time_index={int(t):i for i,t in enumerate(self.valid_times)}
        self.first=int(self.valid_times[0])
    def pool(self,origin,target_times,lag_hours=12):
        if lag_hours not in (3,6,12):raise ValueError('Only frozen 3/6/12h latency scenarios are allowed')
        rows=np.array([self.time_index.get(int(t),-1) for t in target_times],np.int64)
        exists=rows>=0;safe=np.maximum(rows,0)
        # Check both bracketing source forecasts BEFORE reading embeddings.
        reference=self.right_source_times[safe,None]-np.arange(1,8)[None,:]*24*HOUR
        legal=(reference+lag_hours*HOUR<=origin)&exists[:,None]
        legal &= self.mask[safe]
        values=np.where(legal,self.values[safe],0).astype(np.float32)
        h,k=np.nonzero(legal);facts=rows[h]*7+k
        return values,legal,facts.astype(np.int64),rows
    def metadata(self,fact,origin,lag_hours):
        row,k=divmod(int(fact),7);lead=(k+1)*24
        valid=int(self.valid_times[row]);reference=valid-lead*HOUR
        available=int(self.right_source_times[row]-lead*HOUR+lag_hours*HOUR)
        assert available<=origin
        batch_ids=(int(self.left_batches[row]),int(self.right_batches[row]))
        sources=[self.meta['batches'][i] for i in sorted(set(batch_ids))]
        text=self.templates[int(self.template_indices[row,k])]
        return {'document_id':f'gfs:{valid}:{lead}','forecast_kind':'model_forecast_fixed_nominal_lead',
            'valid_time_utc':pd.Timestamp(valid,tz='UTC').isoformat(),'valid_time_ns':valid,'nominal_lead_hours':lead,
            'model_reference_time_ns':reference,'verified_run_time':None,'verified_publication_time':None,
            'assumed_available_at_ns':available,'origin_ns':int(origin),'asof_legal':True,
            'availability_policy':f'max_bracketing_nominal_reference_plus_{lag_hours}h_assumption',
            'temperature_celsius':float(self.values[row,k]),'model':self.meta['model'],
            'requested_coordinate':self.meta['requested_coordinate'],'response_grids':[s['response_grid'] for s in sources],
            'source_response_sha256':[s['sha256'] for s in sources],'api_urls':[s['url'] for s in sources],
            'bracketing_valid_times_ns':[int(self.left_source_times[row]),int(self.right_source_times[row])],
            'interpolation_weights':[5/6,1/6],'rounding_degree_C':0.1,
            'text':text+' Valid time (UTC): '+pd.Timestamp(valid,tz='UTC').isoformat(),
            'embedding_text':text,'embedding_text_policy':'semantic_weather_facts; absolute UTC time is a structured ranking/filtering field'}

@lru_cache(maxsize=2)
def shared_store(directory):return GFSStore(directory)

def query_bin(history,mask,channels):
    def mean(name):
        j=channels.index(name);v=history[:,j][mask[:,j]]
        return float(v.mean()) if len(v) else None
    t,rh,p=mean('T (degC)'),mean('rh (%)'),mean('p (mbar)')
    tb=0 if t is None or t<5 else 1 if t<=20 else 2
    rb=int(rh is not None and rh>=60);pb=int(p is not None and p>=1000)
    return tb*4+rb*2+pb,{'historical_temperature_c':t,'historical_relative_humidity_percent':rh,'historical_pressure_mbar':p,'query_uses_future_y':False}

class MIMODataset(Dataset):
    def __init__(self,directory,split,horizon,strategy='semantic_temporal',top_k=4,temporal_weight=.2,lag_hours=12,seed=2024,
                 check_hashes=True,gfs_directory=None,observation_latency_minutes=None,common_sensitivity=False):
        from pathlib import Path
        self.directory=Path(directory);self.bundle=verify(directory) if check_hashes else read_json(self.directory/'_SUCCESS.json')
        self.meta=read_json(self.directory/'manifest.json');self.channels=self.meta['channels'];self.lookback=self.meta['lookback']
        if split not in ('train','validation','test') or horizon not in self.meta['horizons']:raise ValueError('Unknown split/horizon')
        if strategy not in ('none','missing','mean','latest','semantic','semantic_temporal','random'):raise ValueError('Unknown retrieval strategy')
        if top_k<1 or lag_hours not in (3,6,12):raise ValueError('Invalid retrieval budget/policy')
        self.split,self.horizon=split,horizon;self.strategy,self.top_k,self.temporal_weight,self.lag_hours,self.seed=strategy,top_k,temporal_weight,lag_hours,seed
        self.observation_latency=self.meta['observation_latency_minutes'] if observation_latency_minutes is None else observation_latency_minutes
        if not 0<=self.observation_latency<60:raise ValueError('Observation delay must leave origin strictly before first target')
        z=np.load(self.directory/f'{split}.npz');self.values,self.mask,self.times,self.raw_values=z['values'],z['mask'],z['times'],z['raw_values']
        self.naive_times=z['naive_times'];self.starts=np.load(self.directory/f'{split}_H{horizon}_starts.npy')
        if common_sensitivity:
            allowed=None
            for tz in ('fixed_cet','berlin','utc'):
                p=self.directory.parent/('jena_'+tz)/f'{split}_H{horizon}_starts.npy'
                rows=np.load(p);allowed=rows if allowed is None else np.intersect1d(allowed,rows)
            self.starts=np.intersect1d(self.starts,allowed)
        self.store=shared_store(str(Path(gfs_directory or DATA/'gfs').resolve()))
        # Timezone sensitivity changes external timing, not numerical labels.
        # Pair by original N2 row identity, and compare only common row sets.
        self.sample_binding=digest([self.meta['source_n2_success_sha256'],self.meta['scaler'],split,horizon])
    def __len__(self):return len(self.starts)
    def __getitem__(self,item):
        start=int(self.starts[item]);stop=start+self.lookback
        history_end=int(self.times[stop-1]);origin=history_end+int(self.observation_latency*60*10**9)
        targets=self.times[stop:stop+self.horizon]
        assert history_end<=origin<int(targets[0])
        xmask=self.mask[start:stop];raw_x=self.raw_values[start:stop]
        external,external_mask,facts,rows=self.store.pool(origin,targets,self.lag_hours)
        bin_id,query_stats=query_bin(raw_x,xmask,self.channels)
        shape=(len(self.channels),self.top_k,384)
        vectors=np.zeros(shape,np.float32);tmask=np.zeros(shape[:-1],bool);traces=[[] for _ in self.channels]
        time_features=np.zeros((*shape[:-1],3),np.float32)
        pool_ids=[f'gfs:{int(self.store.valid_times[f//7])}:{(int(f)%7+1)*24}' for f in facts]
        if len(facts) and self.strategy not in ('none','missing'):
            rr,kk=facts//7,facts%7
            template_ids=self.store.template_indices[rr,kk]
            # This is the first access to any candidate embedding: facts were
            # selected by pointwise historical availability above.
            embeddings=self.store.embeddings[template_ids]
            available=self.store.right_source_times[rr]-(kk+1)*24*HOUR+self.lag_hours*HOUR
            ages=(origin-available)/HOUR
            # Keep target timing explicit in fusion rather than asking a
            # language embedding to interpret absolute date strings.
            timing=np.stack(((self.store.valid_times[rr]-origin)/HOUR/168,ages/168,(kk+1)*24/168),axis=-1).astype(np.float32)
            if self.strategy=='mean':
                vectors[:,0]=embeddings.mean(0);tmask[:,0]=True
                time_features[:,0]=timing.mean(0)
                trace={'document_id':'pooled_'+digest(pool_ids),'pool_ids_sha256':digest(pool_ids),'member_count':len(pool_ids),
                    'max_assumed_available_at_ns':int(available.max()),'origin_ns':origin,'asof_legal':True,'query':query_stats}
                traces=[[trace] for _ in self.channels]
            else:
                semantic=self.store.queries[:,bin_id]@embeddings.T if self.strategy in ('semantic','semantic_temporal') else None
                metadata={}
                for c in range(len(self.channels)):
                    if self.strategy=='latest':scores=-ages
                    elif self.strategy=='random':
                        rng=np.random.default_rng(int(digest([self.seed,origin,c])[:16],16));scores=rng.random(len(facts))
                    else:
                        scores=semantic[c].copy()
                        if self.strategy=='semantic_temporal':scores+=self.temporal_weight*np.exp(-ages/24)
                    selected=np.lexsort((facts,-scores))[:self.top_k]
                    vectors[c,:len(selected)]=embeddings[selected];tmask[c,:len(selected)]=True
                    time_features[c,:len(selected)]=timing[selected]
                    for j in selected:
                        fact=int(facts[j])
                        if fact not in metadata:metadata[fact]=self.store.metadata(fact,origin,self.lag_hours)
                        traces[c].append({**metadata[fact],'score':float(scores[j]),'channel':self.channels[c],'query':query_stats})
        ymask=self.mask[stop:stop+self.horizon];y=np.where(ymask,self.values[stop:stop+self.horizon],0).astype(np.float32)
        temp=self.channels.index('T (degC)');mean=self.meta['scaler']['mean'][temp];scale=self.meta['scaler']['scale'][temp]
        ext=np.where(external_mask,(external-mean)/scale,0).astype(np.float32)
        return {'x_num':torch.from_numpy(causal_fill(self.values[start:stop],xmask)),'x_mask':torch.from_numpy(xmask.copy()),
            'y':torch.from_numpy(y),'y_mask':torch.from_numpy(ymask.copy()),'origin_time':origin,'numeric_time':int(self.naive_times[stop-1]),'origin_utc_ns':origin,
            'history_end_time':history_end,'first_target_time':int(targets[0]),'target_times':torch.from_numpy(targets.copy()),
            'channel_ids':torch.arange(len(self.channels)),'text_embeddings':torch.from_numpy(vectors),'text_mask':torch.from_numpy(tmask),
            'text_time_features':torch.from_numpy(time_features),
            'text_metadata':traces,'gfs_num':torch.from_numpy(ext),'gfs_mask':torch.from_numpy(external_mask),
            'gfs_pool_ids':pool_ids,'gfs_pool_sha256':digest(pool_ids),'sample_id':digest([self.sample_binding,start]),
            'eligible_document_counts':torch.full((len(self.channels),),len(facts),dtype=torch.long)}
