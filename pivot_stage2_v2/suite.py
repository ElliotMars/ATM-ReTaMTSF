"""Explicit execute gate; default is a side-effect-free plan."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import yaml
from .common import ROOT,ART,DATA,HORIZONS,digest,read_json,write_json,verify
from .configurations import MAIN,ABLATIONS,configuration,generate

def plans(action,include_h336=False):
    if action not in ('pilot','main','ablations'):return [{'action':action,'formal_training':False}]
    variants=ABLATIONS if action=='ablations' else MAIN
    hs=HORIZONS if include_h336 else HORIZONS[:3]
    return [{'dataset':'jena','horizon':h,'variant':v,'seed':s,'smoke':action=='pilot','formal_training':action!='pilot'}
      for h in hs for v in variants for s in ([2024] if action=='pilot' else [2024,2025,2026])]

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','pilot','main','ablations','collect'])
    g=p.add_mutually_exclusive_group();g.add_argument('--execute',action='store_true');g.add_argument('--dry-run',action='store_true')
    p.add_argument('--include-h336',action='store_true');p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');a=p.parse_args()
    selected=plans(a.action,a.include_h336)
    if not a.execute:
        print(__import__('json').dumps({'dry_run':True,'network':False,'action':a.action,'jobs':selected,'formal_jobs':sum(x.get('formal_training',False) for x in selected)},indent=2));return
    if a.action=='prepare':
        for module in ('acquire','timezone','prepare','configurations','audit'):
            subprocess.run([sys.executable,'-m','pivot_stage2_v2.'+module],cwd=ROOT,check=True)
        return
    if a.action=='validate':
        subprocess.run([sys.executable,'-m','pivot_stage2_v2.audit'],cwd=ROOT,check=True)
        log=ART/'logs/validation_tests.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:
            subprocess.run([sys.executable,'-m','pytest','-q','tests/fidel_ts','tests/pivot_sources','tests/pivot_stage2','tests/pivot_stage2_v2',
                            '--junitxml='+str(ART/'logs/validation_tests.xml')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(log);return
    if a.action=='collect':
        from .collect import collect
        print(collect());return
    admission=read_json(ART/'admission.json')
    if not admission.get('main_enabled'):raise SystemExit('v2 data audit has not admitted Jena; no training started')
    from .train import run,CODE_BINDING
    summaries=[]
    for job in selected:
        config=yaml.safe_load((ROOT/f"configs/pivot_stage2_v2/jena/H{job['horizon']}_{job['variant']}.yaml").read_text())
        config.update(seed=job['seed'],device=a.device)
        directory=ART/('smoke' if job['smoke'] else 'experiments')/f"H{job['horizon']}"/job['variant']/f"seed{job['seed']}"
        if (directory/'_SUCCESS.json').exists():
            record=verify(directory)
            if record['binding']['code_sha256']==CODE_BINDING and record['config_digest']==digest(config):
                summaries.append(read_json(directory/'summary.json'));continue
            if not job['smoke']:raise ValueError('Successful formal result differs; preserve and create a new experiment version')
            retired=ART/'retired_smoke'/str(time.time_ns())/f"H{job['horizon']}"/job['variant'];retired.parent.mkdir(parents=True,exist_ok=True);directory.rename(retired)
        resume=None
        if directory.exists():
            old=directory.with_name(directory.name+'_failed_'+str(time.time_ns()));directory.rename(old)
            if (old/'latest.pt').exists():resume=old/'latest.pt'
        print('Running',job,flush=True)
        summaries.append(run(config,directory,smoke=job['smoke'],resume=resume))
    if a.action=='pilot':write_json(ART/'smoke_summary.json',{'runs':summaries,'formal_training':False,'seeds':[2024],
      'each_run_optimizer_steps':1,'seasonal_examples':{'train':'winter_2022-01-15','validation':'summer_2024-07-15','test':'DST_2025-03-30'},'not_research_results':True})
if __name__=='__main__':main()
