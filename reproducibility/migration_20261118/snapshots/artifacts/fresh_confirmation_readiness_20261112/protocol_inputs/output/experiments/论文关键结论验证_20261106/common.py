"""Frozen inputs and prediction-blind selectors for the bounded paper-claim extension."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,warnings
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix,hstack,vstack,eye,diags
ROOT=Path('/home/hcf/project/QE');sys.path.insert(0,str(ROOT/'scripts'))
import conditional_coverage_common as cc
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/paper_claim_validation_20261106'
BASE=ROOT/'artifacts/coco_five_dimension_fine_selection_20261015'
SPEC=ROOT/'artifacts/other_classes_specific_experiment_20261018'
EXT=ROOT/'artifacts/coverage_diagnostic_validation_20261018'
DROOT=ROOT/'artifacts/objects365_independent_confirmation_20261005'
CLASSES=['dog','cat','horse','sheep','person','car']
METHODS=['R','C20','CS','P20','G','SR']
MODELS=cc.old.MODELS;PSEEDS=cc.old.SEEDS
SEEDS=list(range(2026110600,2026110650))
BUDGETS=[0,10,20,40,60]
SOLVER={'time_limit':20.,'mip_rel_gap':0.,'threads':1,'random_seed':2026110600}

def dump(name,value):cc.dump(OUT/name,value)
def write(name,rows,fields=None):cc.write(OUT/name,rows,fields)
def key(c,m,s):return f'{c}__{m}__{s}'

def encode(rows,features,states):
    X=np.zeros((len(rows),len(states)),bool)
    for i,r in enumerate(rows):
        for j,(_,field,value) in enumerate(states):
            if field=='target_scale':
                a=float(r['bbox_area_ratio']);v='tiny' if a<.01 else 'small' if a<.03 else 'medium' if a<.1 else 'large'
            else:v=features[r['sample_id']].get(field,'unknown')
            X[i,j]=value in v if isinstance(v,list) else value==v
    return X

def contexts():
    rows=cc.read(BASE/'manifest.csv');risk={r['sample_id']:r for r in cc.read(SPEC/'risk_scores_and_tiers.csv')}
    dictionary=json.loads((BASE/'state_dictionary.json').read_text())
    native={r['sample_id']:r['features'] for r in map(json.loads,(BASE/'native_features.jsonl').open())}
    result={}
    for c in CLASSES:
        cr=[r for r in rows if r['class_name']==c];ids=np.array([r['sample_id'] for r in cr])
        if c in ['dog','cat']:
            states=[('shared',*s) for s in dictionary['main_states']]+[('specific',*s) for s in dictionary['local_states'][c]]
            X=encode(cr,native,states)
        else:
            states=[tuple(s) for s in json.loads((SPEC/f'{c}_state_dictionary.json').read_text())['states']]
            with np.load(SPEC/f'{c}_state_matrix.npz',allow_pickle=False) as z:
                assert np.array_equal(ids,z['sample_ids'].astype(str));X=z['matrix'].astype(bool)
        rawkeys=[(int(risk[s]['Q']),round(float(risk[s]['score_midpoint'])*1e8),round((float(risk[s]['score_upper'])-float(risk[s]['score_lower']))*1e8)) for s in ids]
        keys=sorted(set(rawkeys));lookup={k:i for i,k in enumerate(keys)}
        result[c]={'ids':ids,'rows':cr,'matrix':X,'states':states,'tiers':np.array([k[0] for k in rawkeys]),
            'cell':np.array([lookup[k] for k in rawkeys]),'keys':keys,'embeddings':np.zeros((400,1)),
            'score':np.array([float(risk[s]['score_midpoint']) for s in ids])}
        assert len(cr)==400 and len({r['image_sha256'] for r in cr})==400
    return result

def external_contexts(ctx):
    existing=cc.read(EXT/'external_manifest.csv');drows=cc.read(DROOT/'manifests/candidate_pool.csv');result={};files=[]
    for c in CLASSES:
        if c in ['dog','cat']:
            rows=sorted((r for r in drows if r['class_name']==c),key=lambda r:r['sample_id'])[:200]
            features={}
            for r in rows:
                p=DROOT/f"annotations/local/old/parsed/{r['sample_id']}.json";j=json.loads(p.read_text())
                assert j['image_id']==r['sample_id'];features[r['sample_id']]=j['features'];files.append(p)
            X=encode(rows,features,ctx[c]['states'])
        else:
            rows=[r for r in existing if r['class_name']==c]
            p=EXT/f'external_{c}_matrix.npz';files.append(p)
            with np.load(p,allow_pickle=False) as z:
                assert [r['sample_id'] for r in rows]==z['sample_ids'].astype(str).tolist();X=z['matrix'].astype(bool)
        assert X.shape==(200,len(ctx[c]['states']))
        result[c]={'ids':np.array([r['sample_id'] for r in rows]),'rows':rows,'matrix':X,'states':ctx[c]['states']}
    Bhash={r['image_sha256'] for c in ctx.values() for r in c['rows']}
    Dhash={r['image_sha256'] for c in result.values() for r in c['rows']}
    assert not Bhash.intersection(Dhash),'Manifest image-hash overlap between discovery and reference'
    return result,files

def pair_context(ctx):
    c=ctx.copy();order=sorted(range(len(c['ids'])),key=lambda i:c['ids'][i]);pairs=[(i,j) for i in order for j in order if i!=j and c['cell'][i]==c['cell'][j]]
    c['pair_out']=np.array([a for a,b in pairs]);c['pair_in']=np.array([b for a,b in pairs]);c['pair_delta']=c['matrix'][c['pair_in']].astype(np.int8)-c['matrix'][c['pair_out']].astype(np.int8)
    return c

def random_change(ctx,s0,seed,ci,limit=20):
    rng=np.random.default_rng(np.random.SeedSequence([seed,ci,991]));selected=s0.copy();original=set(s0);added=set();n=0
    for pos in rng.permutation(len(s0)):
        outgoing=int(s0[pos]);possible=[int(i) for i in np.flatnonzero(ctx['cell']==ctx['cell'][outgoing]) if int(i) not in original and int(i) not in added]
        if possible:
            incoming=int(rng.choice(possible));selected[pos]=incoming;added.add(incoming);n+=1
            if n==limit:break
    return selected,n

def priority(ctx,s0,threshold,limit):
    X=ctx['matrix'];n0=X[s0].sum(0);before=cc.count_metrics(n0,20)
    if cc.passes(before,threshold):return s0.copy(),dict(stop_reason='initial_pass',triggered=False,optimal_certified=False,stage1_status=-1,stage2_status=-1)
    if limit==0:return s0.copy(),dict(stop_reason='zero_change_limit',triggered=True,optimal_certified=False,stage1_status=-1,stage2_status=-1)
    N,S=X.shape;q=cc.old.quotas(ctx,200);Xt=csr_matrix(X.T.astype(float));Z=csr_matrix((S,S))
    Aq=csr_matrix((np.ones(N),(ctx['cell'],np.arange(N))),shape=(len(q),N))
    retain=np.zeros(N+2*S);retain[s0]=1;sumy=np.r_[np.zeros(N),np.ones(S),np.zeros(S)];sumz=np.r_[np.zeros(N+S),np.ones(S)]
    minocc=max(int((n0>0).sum()),int(np.ceil(threshold['occurrence']*S-1e-9)))
    minunits=max(int(np.minimum(n0,20).sum()),int(np.ceil(threshold['support_completion']*S*20-1e-9)))
    A=vstack([hstack([Aq,csr_matrix((len(q),2*S))]),hstack([Xt,-eye(S),Z]),hstack([Xt,Z,-eye(S)]),
        hstack([Xt,Z,-diags(X.sum(0).astype(float))]),csr_matrix(retain[None]),csr_matrix(sumy[None]),csr_matrix(sumz[None])],format='csr')
    lo=np.r_[q,np.zeros(S),np.zeros(S),np.full(S,-np.inf),200-limit,minunits,minocc]
    hi=np.r_[q,np.full(S,np.inf),np.full(S,np.inf),np.zeros(S),200,np.inf,np.inf]
    bounds=Bounds(np.zeros(N+2*S),np.r_[np.ones(N),np.full(S,20),np.ones(S)]);integer=np.r_[np.ones(N),np.zeros(S),np.ones(S)]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*',category=RuntimeWarning)
        a=milp(-sumy,integrality=integer,bounds=bounds,constraints=LinearConstraint(A,lo,hi),options=SOLVER)
        if a.x is None:return s0.copy(),dict(stop_reason='infeasible' if a.status==2 else 'no_feasible_incumbent',triggered=True,optimal_certified=False,stage1_status=int(a.status),stage2_status=-1)
        s1=np.flatnonzero(a.x[:N]>.5);units=int(np.minimum(X[s1].sum(0),20).sum());lo2=lo.copy();lo2[-2]=units
        b=milp(-sumz,integrality=integer,bounds=bounds,constraints=LinearConstraint(A,lo2,hi),options=SOLVER)
    selected=np.flatnonzero(b.x[:N]>.5) if b.x is not None else s1
    metrics=cc.count_metrics(X[selected].sum(0),20)
    assert cc.passes(metrics,threshold) and metrics['occurrence']+1e-12>=before['occurrence'] and metrics['support_completion']+1e-12>=before['support_completion']
    assert len(set(s0)-set(selected))<=limit
    return selected,dict(stop_reason='lexicographic_optimal' if a.status==b.status==0 else 'feasible_solver_limit',triggered=True,optimal_certified=bool(a.status==b.status==0),stage1_status=int(a.status),stage2_status=int(b.status))

def run_method(ctx,method,seed,ci,threshold,limit=20):
    start=time.perf_counter();s0=cc.initial_selection(ctx,200,seed,ci);passed=cc.passes(cc.count_metrics(ctx['matrix'][s0].sum(0),20),threshold)
    work={'triggered':not passed,'optimal_certified':False,'stage1_status':'','stage2_status':''}
    if method=='R':s=s0.copy();work.update(stop_reason='untouched_initial',triggered=False)
    elif method=='C20':
        s,n=(s0.copy(),0) if passed else random_change(ctx,s0,seed,ci,20);work['stop_reason']='initial_pass' if passed else 'random_changes_complete'
    elif method=='CS':
        if passed:s=s0.copy();work['stop_reason']='initial_pass'
        else:s,detail,trace=cc.swap_select(pair_context(ctx),s0,threshold,cap=20,conditional=True);work.update(detail)
    elif method in ['P20','Pbudget']:s,work=priority(ctx,s0,threshold,limit)
    elif method in ['G','SR']:s,detail=cc.baseline(ctx,200,method,20);work.update(detail,triggered=False)
    else:raise ValueError(method)
    metrics=cc.count_metrics(ctx['matrix'][s].sum(0),20);elapsed=time.perf_counter()-start;cc.old.validate_selection(ctx,s,200)
    n=ctx['matrix'][s].sum(0)
    record={'class_name':CLASSES[ci],'method':method,'selection_seed':seed,'change_limit':limit if method in ['P20','Pbudget','C20'] else '',
        'initial_pass':passed,'final_pass':cc.passes(metrics,threshold),'seconds':elapsed,'removed_targets':len(set(s0)-set(s)),
        'occurrence_pct':100*metrics['occurrence'],'support_completion_pct':100*metrics['support_completion'],'supported_states':int((n>=20).sum()),
        'stop_reason':work['stop_reason'],'triggered':work['triggered'],'optimal_certified':work['optimal_certified'],
        'stage1_status':work.get('stage1_status',''),'stage2_status':work.get('stage2_status','')}
    return s,s0,record
