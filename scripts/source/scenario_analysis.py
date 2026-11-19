"""Post-estimation payment-arrangement scenarios. Offline; no API clients or credentials.

Reproduce: OPENBLAS_NUM_THREADS=1 python scenario_analysis.py --replicates 1000
The original main experiment and its frozen predictions remain unchanged.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, csv, hashlib, itertools, json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parent
MODELS = ['sol', 'qwen', 'claude']
MANDATES = ['reference', 'cost', 'protection']
SEED = 2026111317
FEE_MIN, FEE_MAX, KNOT = .001, .250, .020
ANCHORS = [.010, .020, .051]
FAMILIES = [
 {'id':'S1','name':'Operator keys without recovery','custody':'operator','identity':'persistent','refund':0.,'loss_probability':.001},
 {'id':'S2','name':'Provider keys without recovery','custody':'provider','identity':'persistent','refund':0.,'loss_probability':.001},
 {'id':'S3','name':'Operator keys with full recovery','custody':'operator','identity':'persistent','refund':1.,'loss_probability':.001},
 {'id':'S4','name':'Full recovery with higher reserve risk','custody':'operator','identity':'persistent','refund':1.,'loss_probability':.01},
 {'id':'S5','name':'Full recovery with lower reserve risk','custody':'operator','identity':'persistent','refund':1.,'loss_probability':.0001},
 {'id':'S6','name':'Full recovery with transaction pseudonym','custody':'operator','identity':'transaction','refund':1.,'loss_probability':.001},
]

def load(name): return json.loads((ROOT/'inputs'/name).read_text())
def write(path, obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def features(p, hinge):
 f=float(p['fee'])
 x=[f/.1,float(p['loss_probability'])==.001,float(p['loss_probability'])==.0001,
    float(p['refund'])==.5,float(p['refund'])==1,p['custody']=='operator',
    p['identity']=='persistent',p['identity']=='transaction']
 if hinge:x.append(max(f-KNOT,0)/.1)
 return np.asarray(x,dtype=float)
def options(pair,order,hinge):
 x=np.zeros((3,len(features(pair['profiles']['X'],hinge))+2))
 for j,key in enumerate(order):x[j]=np.r_[1.,features(pair['profiles'][key],hinge),float(j==0)]
 return x
def objective(b,x,y,penalty):
 v=x@b;p=softmax(v,axis=1)
 value=np.sum(logsumexp(v,axis=1)-v[np.arange(len(y)),y])+.5*penalty*np.dot(b,b)
 p[np.arange(len(y)),y]-=1
 return float(value),np.einsum('na,nak->k',p,x)+penalty*b
def fit(x,y,penalty,start=None):
 if start is None:start=np.zeros(x.shape[2])
 r=minimize(objective,start,args=(x,y,penalty),jac=True,method='L-BFGS-B',
            options={'maxiter':2000,'gtol':1e-7,'ftol':1e-12})
 if not r.success or not np.all(np.isfinite(r.x)):return None
 return r.x
def comparator(fee):return {'fee':fee,'loss_probability':.001,'refund':1.,'custody':'provider','identity':'legal'}
def balanced_probs(profiles,comp,betas,hinge):
 """Average unconditional probabilities over display order, then condition on purchase."""
 betas=np.atleast_2d(betas)
 fp=np.asarray([np.r_[1.,features(p,hinge)] for p in profiles])
 cp=np.r_[1.,features(comp,hinge)]
 a=betas[:,:-1]@fp.T;b=betas[:,:-1]@cp;d=betas[:,-1,None]
 b=np.broadcast_to(b[:,None],a.shape);z=np.zeros_like(a)
 p1=softmax(np.stack([a+d,b,z],axis=-1),axis=-1)
 p2=softmax(np.stack([a,b+d,z],axis=-1),axis=-1)
 p=(p1+p2)/2
 conditional=p[:,:,0]/(p[:,:,0]+p[:,:,1])
 return p,conditional
def boundary(profile,comp,beta,hinge):
 """Equal-profile utility implies equal order-balanced purchase probabilities.
 Return domain-censored limits and monotonicity status, never a WTP estimate.
 """
 slope_low=beta[1]/.1;slope_high=(beta[1]+(beta[-2] if hinge else 0))/.1
 if slope_low>=0 or slope_high>=0:return {'status':'nondecreasing_segment','bounded_fee':None}
 def delta(f):return float((features({**profile,'fee':f},hinge)-features(comp,hinge))@beta[1:-1])
 lo,hi=delta(FEE_MIN),delta(FEE_MAX)
 if lo<0:return {'status':'below_range','bounded_fee':FEE_MIN}
 if hi>0:return {'status':'above_range','bounded_fee':FEE_MAX}
 middle=delta(KNOT)
 if not hinge:root=FEE_MIN-lo/slope_low
 elif middle<=0:root=FEE_MIN-lo/slope_low
 else:root=KNOT-middle/slope_high
 return {'status':'within_range','bounded_fee':float(root)}

def bootstrap_cell(task):
 model,mandate,reps=task
 fitted=next(v for v in load('prediction_results.json')['fitted_models'] if (v['model'],v['mandate'])==(model,mandate))
 hinge=fitted['selected']['nonlinear'];penalty=fitted['selected']['penalty'];point=np.array(fitted['coefficients'])
 pairs={p['case_id']:p for p in map(json.loads,(ROOT/'inputs/physical_pairs.jsonl').read_text().splitlines())}
 observations=[r for r in load('observations.json') if r['stage']=='estimation' and r['status']=='valid' and r['model']==model and r['mandate']==mandate]
 groups=[]
 for case in [f'E{i:03}' for i in range(1,65)]:
  group=[r for r in observations if r['case_id']==case]
  x=np.array([options(pairs[case],r['order'],hinge) for r in group])
  y=np.array([2 if r['physical_choice']=='DECLINE' else r['order'].index(r['physical_choice']) for r in group])
  groups.append((x,y))
 x=np.concatenate([g[0] for g in groups]);y=np.concatenate([g[1] for g in groups])
 reproduced=fit(x,y,penalty)
 assert reproduced is not None
 diff=float(np.max(np.abs(softmax(x@reproduced,axis=1)-softmax(x@point,axis=1))))
 assert diff<1e-5,(model,mandate,diff)
 draws=np.random.default_rng(SEED).integers(0,64,size=(reps,64))
 bs=np.full((reps,len(point)),np.nan)
 for i,draw in enumerate(draws):
  xx=np.concatenate([groups[j][0] for j in draw]);yy=np.concatenate([groups[j][1] for j in draw])
  b=fit(xx,yy,penalty,point)
  if b is not None:bs[i]=b
  if (i+1)%200==0:print(json.dumps({'cell':model+'/'+mandate,'bootstrap':i+1,'of':reps}),flush=True)
 np.save(ROOT/'analysis'/f'bootstrap_{model}_{mandate}.npy',bs)
 return {'model':model,'mandate':mandate,'bootstrap_valid':int(np.isfinite(bs).all(axis=1).sum()),
         'replicates':reps,'reproduced_probability_max_error':diff,'hinge':hinge,'penalty':penalty}

def summarize(reps):
 fits=load('prediction_results.json')['fitted_models'];domain=load('attributes.json')['levels']
 profiles=[dict(zip(domain,vals)) for vals in itertools.product(*domain.values())]
 rows=[];boundaries=[];contrasts=[];curves=[]
 for fitted in fits:
  model,mandate=fitted['model'],fitted['mandate'];hinge=fitted['selected']['nonlinear']
  point=np.asarray(fitted['coefficients']);bs=np.load(ROOT/'analysis'/f'bootstrap_{model}_{mandate}.npy')
  good=np.isfinite(bs).all(axis=1);bs=bs[good]
  for anchor in ANCHORS:
   comp=comparator(anchor);p,q=balanced_probs(profiles,comp,point,hinge);bp,bq=balanced_probs(profiles,comp,bs,hinge)
   qci=np.quantile(bq,[.025,.975],axis=0);pci=np.quantile(bp,[.025,.975],axis=0)
   for j,profile in enumerate(profiles):
    rows.append({'model':model,'mandate':mandate,'comparator_fee':anchor,**profile,
     'conditional_share':float(q[0,j]),'conditional_lo':float(qci[0,j]),'conditional_hi':float(qci[1,j]),
     'purchase_probability':float(p[0,j,0]),'comparator_probability':float(p[0,j,1]),'decline_probability':float(p[0,j,2]),
     'decline_lo':float(pci[0,j,2]),'decline_hi':float(pci[1,j,2]),'bootstrap_valid':len(bs)})
   for family in FAMILIES:
    bd=boundary(family,comp,point,hinge);bb=[boundary(family,comp,b,hinge) for b in bs]
    statuses={s:sum(x['status']==s for x in bb) for s in ['within_range','below_range','above_range','nondecreasing_segment']}
    vals=[x['bounded_fee'] for x in bb if x['bounded_fee'] is not None]
    ci=np.quantile(vals,[.025,.975]).tolist() if vals else [None,None]
    boundaries.append({'model':model,'mandate':mandate,'comparator_fee':anchor,'scenario':family['id'],**bd,
                       'domain_censored_interval95':ci,'bootstrap_statuses':statuses})
   # Paired gains from guarantee, custody, and lower/higher reserve risk at selected fees.
   for fee in [.001,.010,.020,.051,.111,.250]:
    fam=[{**f,'fee':fee} for f in FAMILIES];_,pq=balanced_probs(fam,comp,point,hinge);_,bq=balanced_probs(fam,comp,bs,hinge)
    for name,a,b in [('full_recovery',0,2),('operator_control',1,0),('lower_reserve_risk',2,4),('higher_reserve_risk',2,3),('transaction_pseudonym',2,5)]:
     interval=np.quantile(bq[:,b]-bq[:,a],[.025,.975])
     contrasts.append({'model':model,'mandate':mandate,'comparator_fee':anchor,'fee':fee,'contrast':name,
                       'probability_change':float(pq[0,b]-pq[0,a]),'lo':float(interval[0]),'hi':float(interval[1])})
  # Plot interpolation is confined to the observed fee interval.
  fees=np.unique(np.r_[np.geomspace(FEE_MIN,FEE_MAX,160),[float(f) for f in domain['fee']]])
  for family in FAMILIES:
   ps=[{**family,'fee':float(f)} for f in fees];_,q=balanced_probs(ps,comparator(.020),point,hinge);_,bq=balanced_probs(ps,comparator(.020),bs,hinge)
   ci=np.quantile(bq,[.025,.975],axis=0)
   for j,fee in enumerate(fees):curves.append({'model':model,'mandate':mandate,'scenario':family['id'],'fee':float(fee),'share':float(q[0,j]),'lo':float(ci[0,j]),'hi':float(ci[1,j])})
 for name,data in [('scenario_grid',rows),('scenario_contrasts',contrasts),('scenario_curves',curves)]:
  with (ROOT/'analysis'/f'{name}.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 write(ROOT/'analysis/scenario_boundaries.json',boundaries)
 table=[]
 for fit in fits:
  for family in FAMILIES:
   row=next(r for r in rows if r['model']==fit['model'] and r['mandate']==fit['mandate'] and r['comparator_fee']==.02 and float(r['fee'])==.01 and all(str(r[k])==str(family[k]) or (k in ['refund','loss_probability'] and float(r[k])==float(family[k])) for k in ['custody','identity','refund','loss_probability']))
   table.append({**row,'scenario':family['id']})
 write(ROOT/'analysis/scenario_table.json',table)
 return rows,contrasts,boundaries

def validate(rows,contrasts,boundaries):
 checks={}
 checks['grid_rows']=len(rows)==9*3*486
 checks['probability_sums']=all(abs(r['purchase_probability']+r['comparator_probability']+r['decline_probability']-1)<1e-12 for r in rows)
 same=[r for r in rows if float(r['fee'])==r['comparator_fee'] and float(r['loss_probability'])==.001 and float(r['refund'])==1 and r['custody']=='provider' and r['identity']=='legal']
 checks['identical_profiles_equal_share']=len(same)==27 and all(abs(r['conditional_share']-.5)<1e-12 for r in same)
 fits=load('prediction_results.json')['fitted_models'];root_errors=[]
 for bd in boundaries:
  if bd['status']!='within_range':continue
  fit=next(f for f in fits if (f['model'],f['mandate'])==(bd['model'],bd['mandate']))
  family=next(f for f in FAMILIES if f['id']==bd['scenario'])
  _,q=balanced_probs([{**family,'fee':bd['bounded_fee']}],comparator(bd['comparator_fee']),np.array(fit['coefficients']),fit['selected']['nonlinear'])
  root_errors.append(abs(float(q[0,0])-.5))
 checks['all_threshold_roots_equal_share']=max(root_errors,default=0)<1e-10
 checks['maximum_threshold_error']=max(root_errors,default=0)
 checks['probability_intervals_in_range']=all(0<=r['conditional_lo']<=r['conditional_hi']<=1 for r in rows)
 checks['all_point_fee_slopes_negative']=all(f['coefficients'][1]<0 and (not f['selected']['nonlinear'] or f['coefficients'][1]+f['coefficients'][-2]<0) for f in fits)
 checks['source_hashes_match']=all(sha(ROOT/'inputs'/Path(p).name)==h for p,h in load('main_source_hashes.json').items())
 assert all(v for k,v in checks.items() if k!='maximum_threshold_error'),checks
 write(ROOT/'analysis/scenario_validation.json',{'status':'passed','checks':checks})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=1000);ap.add_argument('--summarize-only',action='store_true');args=ap.parse_args()
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'post-estimation extension; specified after main results',
  'seed':SEED,'replicates':args.replicates,'bootstrap_unit':'64 physical estimation pairs; common draws across all nine cells',
  'selection':'Frozen predictive fee forms and penalties; selection is not repeated within bootstrap',
  'comparator':'Provider-controlled reserve, verified legal identity, full guaranteed principal recovery, 0.1% reserve loss probability',
  'comparator_fees':ANCHORS,'primary_comparator_fee':.020,'focal_domain':load('attributes.json')['levels'],'families':FAMILIES,
  'scope':'Common $1 purchase and $20 reserve scenario. No empirical risk rates, fee quotes, market shares, network-label effects or new LLM calls.',
  'prediction':'Average unconditional probabilities over both positions, then condition the focal choice on either purchase.',
  'threshold':'Equal order-balanced predicted choice, within $0.001 to $0.250; descriptive predictive boundary, not WTP.',
  'intervals':'95% paired percentile intervals conditional on fixed penalty/form and stated profiles. Outside-range boundaries censored at limits, with counts retained.'}
 if not args.summarize_only:
  write(ROOT/'inputs/scenario_plan.json',plan)
  summaries=[]
  with ProcessPoolExecutor(max_workers=3) as ex:
   futures=[ex.submit(bootstrap_cell,(m,d,args.replicates)) for m,d in itertools.product(MODELS,MANDATES)]
   for future in as_completed(futures):
    result=future.result();summaries.append(result);print(json.dumps(result),flush=True)
  write(ROOT/'analysis/scenario_bootstrap_summary.json',summaries)
 rows,contrasts,boundaries=summarize(args.replicates);validate(rows,contrasts,boundaries)
 print(json.dumps({'status':'complete','grid_rows':len(rows),'paired_contrasts':len(contrasts),'thresholds':len(boundaries)}),flush=True)
if __name__=='__main__':main()
