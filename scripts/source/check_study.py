"""Offline entry point for the November 19 replication supplement."""
import argparse, hashlib, json, subprocess, sys, tempfile, shutil
from pathlib import Path
root=Path(__file__).resolve().parent
main=root/'main'
sys.path.insert(0,str(main))
from analyze_main import analyze_prediction, reliability, run_monetary
from runtime import parse
read=lambda p:json.loads(p.read_text())
args=argparse.ArgumentParser()
args.add_argument('--monetary',action='store_true',help='also re-estimate both monetary specifications and 1,000 pair bootstraps')
opts=args.parse_args()
records=read(main/'analysis/observations.json')
episodes={e['episode_id']:e for e in map(json.loads,(main/'inputs/episodes.jsonl').read_text().splitlines())}
for r in records:
 rawpath=main/'runs'/r['model']/'responses'/(r['episode_id']+'.json')
 assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==r['response_sha256']
 raw=read(rawpath)
 assert raw['request']['messages']==episodes[r['episode_id']]['messages']
 status,choice=parse(raw,episodes[r['episode_id']])
 if r['status']=='valid':assert status=='valid' and choice==r['physical_choice']
 else:assert r['status']=='service_error' and raw['state']!='completed'
assert len(records)==1602
assert reliability(main,records)==read(main/'analysis/run_results.json')['cells']
replay=analyze_prediction(main,records)
archive=read(main/'analysis/prediction_results.json')
import numpy as np
for c,d in zip(replay['cells'],archive['cells']):
 for rule in c['scores']:
  for metric in c['scores'][rule]:
   for k in ['estimate','interval95']:np.testing.assert_allclose(c['scores'][rule][metric][k],d['scores'][rule][metric][k],rtol=1e-8,atol=1e-10)
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp)
 subprocess.run([sys.executable,str(root/'diagnostics.py'),str(root),str(tmp)],check=True)
 import csv
 for f in (root/'revision').glob('*.csv'):
  a=list(csv.DictReader(f.open()));b=list(csv.DictReader((tmp/f.name).open()))
  assert len(a)==len(b)
  for x,y in zip(a,b):
   assert x.keys()==y.keys()
   for k in x:
    try:v,w=float(x[k]),float(y[k])
    except ValueError:assert x[k]==y[k]
    else:np.testing.assert_allclose(v,w,atol=1e-5,rtol=1e-6)
 if opts.monetary:
  (tmp/'inputs').mkdir();shutil.copyfile(main/'inputs/physical_pairs.jsonl',tmp/'inputs/physical_pairs.jsonl')
  calc=run_monetary(tmp,records)
  for a,b in zip(calc['cells'],read(main/'analysis/monetary_results.json')['cells']):
   assert a['monetary_status']==b['monetary_status']
   for seg in a['tradeoffs']:
    for k,item in a['tradeoffs'][seg]['contrasts'].items():
     ref=b['tradeoffs'][seg]['contrasts'][k]
     np.testing.assert_allclose(item['estimate_usd'],ref['estimate_usd'],rtol=1e-5,atol=1e-7)
print('Verified 1,602 first attempts, response consistency, frozen test scores, and all revision diagnostics.')
