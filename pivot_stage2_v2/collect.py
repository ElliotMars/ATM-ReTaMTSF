"""Immutable formal-only result tables; timezone sensitivity has a common scope."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import uuid,csv
import numpy as np
import yaml
from pivot_stage2.metrics import block_bootstrap
from .common import ART,read_json,write_json,verify,seal,new_directory

def collect(root=None):
    root=Path(root or ART/'experiments');records=[];seen=set();fingerprints={};groups=defaultdict(list)
    for p in sorted(root.rglob('summary.json')) if root.exists() else []:
        if not (p.parent/'_SUCCESS.json').exists():continue
        verify(p.parent);s=read_json(p)
        if s['smoke_only']:continue
        c=yaml.safe_load((p.parent/'config.yaml').read_text());scope=c['comparison_scope']
        key=(s['horizon'],scope,c['variant']);unique=(*key,s['seed'])
        if unique in seen:raise ValueError('Duplicate formal run; no choosing favorable repetitions')
        seen.add(unique);compare=key[:2];fingerprint=s['test']['sample_ids_sha256']
        if compare in fingerprints and fingerprints[compare]!=fingerprint:raise ValueError('Methods have different numerical evaluation samples')
        fingerprints[compare]=fingerprint
        row={'dataset':'jena','horizon':s['horizon'],'scope':scope,'variant':c['variant'],'seed':s['seed'],
          **{k:s['test'][k] for k in ('mse_standardized','mae_standardized','mse','mae','label_coverage')},
          'parameters':s['parameters'],'wall_seconds':s['wall_seconds'],'run':str(p.parent)}
        records.append(row);groups[key].append(row)
    aggregates=[]
    for (h,scope,v),rows in groups.items():
        curves=[];origins=None
        for r in rows:
            e=np.load(Path(r['run'])/'test/origin_errors.npz')
            if origins is not None and not np.array_equal(origins,e['origins']):raise ValueError('Original station row origins differ')
            origins=e['origins'];curves.append(e['mse_standardized'])
        stats={k:{'mean':float(np.mean([r[k] for r in rows])),'seed_std':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None}
               for k in ('mse_standardized','mae_standardized','mse','mae')}
        base='timezone_cet_common' if scope=='common_timezone_rows' else 'numeric_only'
        baseline={r['seed']:r for r in groups.get((h,scope,base),[])};diff=[]
        for r in rows:
            if r['seed'] in baseline:
                b=np.load(Path(baseline[r['seed']]['run'])/'test/origin_errors.npz');e=np.load(Path(r['run'])/'test/origin_errors.npz')
                if not np.array_equal(b['origins'],e['origins']):raise ValueError('Unpaired source origins')
                diff.append(e['mse_standardized']-b['mse_standardized'])
        aggregates.append({'horizon':h,'scope':scope,'variant':v,'seeds':sorted(r['seed'] for r in rows),'statistics':stats,
          'temporal_block_ci':block_bootstrap(np.mean(curves,0),h),'paired_difference_vs_'+base:block_bootstrap(np.mean(diff,0),h) if diff else None})
    output=ART/'collections'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:6])
    with new_directory(output):
        write_json(output/'results.json',{'status':'formal_results_available' if records else 'no_formal_results_yet','runs':records,'aggregates':aggregates,'smoke_results_excluded':True})
        with (output/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['dataset','horizon','scope','variant','seed','mse_standardized','mae_standardized']);w.writeheader();w.writerows(records)
        seal(output,{'kind':'v2_formal_collection','runs':len(records)})
    write_json(ART/'latest_collection.json',{'directory':str(output)})
    return str(output)
