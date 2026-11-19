"""Reproduce the November 19 sensitivity checks from archived first attempts. Offline."""
from pathlib import Path
import csv, json, sys
import numpy as np
root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root/'scenario'))
import scenario_analysis as sc
M = root/'main'
read=lambda p: json.loads(p.read_text())
pairs={r['case_id']:r for r in map(json.loads,(M/'inputs/physical_pairs.jsonl').read_text().splitlines())}
obs=read(M/'analysis/observations.json')
frozen=read(M/'predictions.json')
preds={(r['model'],r['episode_id']):r for r in frozen['predictions']}
summary=[]; spec=[]; smoothing=[]
comp=sc.comparator(.02)
s1={**sc.FAMILIES[0],'fee':.01}
s3={**sc.FAMILIES[2],'fee':.01}
matched={**s3,'fee':.02}
def write(name,rows):
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for fit in frozen['fits']:
 m,a=fit['model'],fit['mandate']
 selected=fit['selected']
 valid=[r for r in obs if r['status']=='valid' and (r['model'],r['mandate'])==(m,a)]
 train=[r for r in valid if r['stage']=='estimation']
 test=[r for r in valid if r['stage']=='holdout']
 y=np.array([['A','B','DECLINE'].index(r['display_choice']) for r in train])
 for hinge in (False,True):
  x=np.array([sc.options(pairs[r['case_id']],r['order'],hinge) for r in train])
  for penalty in (.01,.1,1,10):
   beta=sc.fit(x,y,penalty)
   assert beta is not None
   _,probs=sc.balanced_probs([s1,s3,matched],comp,beta,hinge)
   q=probs[0]
   bd=sc.boundary(s1,comp,beta,hinge)
   spec.append(dict(model=m,mandate=a,hinge=hinge,penalty=penalty,same_fee_gain_pp=100*(q[1]-q[0]),matched_gain_pp=100*(q[2]-q[0]),boundary_status=bd['status'],s1_boundary_cents=None if bd['bounded_fee'] is None else 100*bd['bounded_fee']))
   if (hinge,penalty)==(selected['nonlinear'],selected['penalty']):
    coeff=np.array(fit['coefficients'])
    bs=np.load(root/'scenario/analysis'/f'bootstrap_{m}_{a}.npy')
    _,bq=sc.balanced_probs([s1,s3,matched],comp,bs,hinge)
    _,ref=sc.balanced_probs([s1,s3,matched],comp,coeff,hinge)
    assert np.max(np.abs(ref[0]-q))<1e-5 and len(bs)==1000
    lo,hi=np.quantile(100*(bq[:,2]-bq[:,0]),[.025,.975])
    summary.append(dict(model=m,mandate=a,matched_gain_pp=100*(ref[0,2]-ref[0,0]),lo=lo,hi=hi))
 for mass in (.90,.95,.98,.99):
  draw=np.random.default_rng(2026111311).integers(0,16,size=(2000,16))
  for rule in ['fee','expenditure','protection','modal_display']:
   losses=np.zeros((16,2));count=np.zeros(16)
   for r in test:
    dist=preds[m,r['episode_id']]['probabilities']
    yobs=['A','B','DECLINE'].index(r['display_choice'])
    prob=np.array(dist[rule])
    if rule!='modal_display':
     wins=prob==prob.max()
     prob=wins*mass/wins.sum()+(1-mass)/3
    i=int(r['case_id'][1:])-1
    losses[i]-=np.log([dist['logit'][yobs],prob[yobs]]);count[i]+=1
   totals=losses.sum(axis=0)/count.sum()
   delta=losses[:,0]-losses[:,1]
   lo,hi=np.quantile(delta[draw].sum(axis=1)/count[draw].sum(axis=1),[.025,.975])
   smoothing.append(dict(model=m,mandate=a,winner_mass=mass,rule=rule,delta_log_loss=totals[0]-totals[1],lo=lo,hi=hi))
write('matched.csv',summary);write('specifications.csv',spec);write('smoothing.csv',smoothing)
