#!/usr/bin/env python3
"""Conditional mixed-model supplements, never replacements for main tests."""
import json
import itertools
import math
import warnings
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from patsy import dmatrix, build_design_matrices
from automatic_strengthening_common import *

OUT=BASE/'statistics'


def observations(pools,source):
    frames=[]
    for pool,path in pools:
        rows=read_csv(path/'manifests/candidate_pool.csv')
        risks=pd.read_csv(path/'selection/risk_scores.csv')
        scores=risks[risks.source==source].set_index('sample_id')
        for model,seed in itertools.product(MODELS,SEEDS):
            with np.load(path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as saved:
                assert saved['sample_ids'].tolist()==[r['sample_id'] for r in rows]
                pred=saved['logits'].argmax(1)
            frames.append(pd.DataFrame(dict(dataset=pool,architecture=model,run_id=f'{model}-{seed}',
                image_id=[f'{pool}:{r["image_id"]}' for r in rows],class_name=[r['class_name'] for r in rows],
                risk_tier=[int(scores.loc[r['sample_id'],'tier']) for r in rows],
                error=pred!=np.asarray([LABELS[r['class_name']] for r in rows]))))
    frame=pd.concat(frames,ignore_index=True)
    assert not frame.duplicated(['image_id','run_id']).any()
    frame.error=frame.error.astype(int);frame.risk_tier=pd.Categorical(frame.risk_tier,categories=[1,2,3,4,5])
    return frame


def fit_risk(pools,source):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    frame=observations(pools,source)
    matrix=dmatrix('C(risk_tier) * C(dataset) + C(class_name) + C(architecture)',frame,return_type='dataframe',eval_env=-1)
    runs,run_names=pd.factorize(frame.run_id,sort=True);images,image_names=pd.factorize(frame.image_id,sort=True)
    n=len(frame);nrun=len(run_names);nimg=len(image_names)
    # Each observation has two nonzero entries; avoid an enormous dense image design matrix.
    random=sparse.coo_matrix((np.ones(2*n),(np.concatenate([np.arange(n),np.arange(n)]),np.concatenate([runs,nrun+images]))),
                            shape=(n,nrun+nimg)).tocsr()
    model=BinomialBayesMixedGLM(frame.error.values,np.asarray(matrix),random,np.r_[np.zeros(nrun,dtype=int),np.ones(nimg,dtype=int)],
                              fe_p=2,vcp_p=1,fep_names=list(matrix.columns),vcp_names=['run_sd','image_sd'])
    rng=np.random.default_rng(20261005)
    note('statistics',source=source,operation='variational_GLMM',observations=n,images=nimg,runs=nrun,optimizer='L-BFGS-B')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted=model.fit_vb(fit_method='L-BFGS-B',minim_opts={'maxiter':300,'ftol':1e-9,'gtol':1e-5},rng=rng)
    converged=bool(fitted.optim_retvals.success)
    path=OUT/source;path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'posterior.npz',fe_mean=fitted.fe_mean,fe_sd=fitted.fe_sd,vcp_mean=fitted.vcp_mean,vcp_sd=fitted.vcp_sd,
                        parameter_names=np.asarray(matrix.columns))
    write_csv(path/'fixed_effects.csv',[dict(parameter=name,posterior_mean=float(mean),posterior_sd=float(sd),
        approximate_cr95_low=float(mean-1.96*sd),approximate_cr95_high=float(mean+1.96*sd)) for name,mean,sd in zip(matrix.columns,fitted.fe_mean,fitted.fe_sd)])
    diag=dict(status='converged' if converged else 'not_converged',source=source,n=n,image_count=nimg,run_count=nrun,
              architecture_fixed=True,message=str(fitted.optim_retvals.message),iterations=int(fitted.optim_retvals.nit),
              warnings=[str(w.message) for w in caught],run_sd=float(np.exp(fitted.vcp_mean[0])),image_sd=float(np.exp(fitted.vcp_mean[1])),
              interval='mean-field variational approximate posterior; may underestimate uncertainty; no frequentist p-values',
              inference_allowed=converged and bool(np.isfinite(fitted.fe_mean).all()))
    write_json(path/'fit_audit.json',diag)
    results=[];grid=pd.DataFrame([dict(dataset=pool,architecture=m,class_name=c,risk_tier=1) for pool,_ in pools for m in MODELS for c in CLASSES])
    grid.risk_tier=pd.Categorical(grid.risk_tier,categories=[1,2,3,4,5])
    # Average each class/architecture equally and integrate new run/image random intercepts.
    nodes,weights=np.polynomial.hermite.hermgauss(15)
    draws=2000;beta=rng.normal(fitted.fe_mean,fitted.fe_sd,size=(draws,len(fitted.fe_mean)))
    logsd=rng.normal(fitted.vcp_mean,fitted.vcp_sd,size=(draws,2));variance=(np.exp(2*logsd)).sum(1)
    for pool,_ in pools:
        low=grid[grid.dataset==pool].copy();high=low.copy();high.risk_tier=pd.Categorical(np.full(len(high),5),categories=[1,2,3,4,5])
        design_low=np.asarray(build_design_matrices([matrix.design_info],low)[0]);design_high=np.asarray(build_design_matrices([matrix.design_info],high)[0])
        offset=design_high[0]-design_low[0]
        odds=np.exp(beta@offset);deltas=[]
        for start in range(0,draws,100):
            bb=beta[start:start+100];vv=variance[start:start+100]
            def marginal(xx):
                eta=bb@xx.T
                probs=expit(eta[:,:,None]+np.sqrt(2*vv)[:,None,None]*nodes[None,None,:])
                return (probs@weights/np.sqrt(np.pi)).mean(1)
            deltas.extend((marginal(design_high)-marginal(design_low)).tolist())
        results.append(dict(source=source,dataset=pool,metric='Q5_vs_Q1_error',conditional_odds_ratio_median=float(np.median(odds)),
            approximate_cr95_or_low=float(np.quantile(odds,.025)),approximate_cr95_or_high=float(np.quantile(odds,.975)),
            marginal_error_difference_median=float(np.median(deltas)),approximate_cr95_delta_low=float(np.quantile(deltas,.025)),
            approximate_cr95_delta_high=float(np.quantile(deltas,.975)),fit_converged=converged,
            scope='fixed eight architectures, balanced class/architecture standardization; approximate Bayesian sensitivity'))
    write_csv(path/'risk_effects.csv',results)
    return diag


def fit_auc(pools):
    import statsmodels.formula.api as smf
    from scipy.stats import shapiro
    frames=[]
    for pool,path in pools:
        table=pd.read_csv(path/'comparison/tables/equal_error_metrics.csv')
        table=table[table.representation=='classifier']
        for (source,model,seed,baseline),group in table.groupby(['source','model','seed','baseline']):
            group=group.sort_values('budget');x=np.r_[0,group.budget.values];maximum=x[-1]
            for metric in ('confusion','clusters'):
                y=np.r_[0,group['delta_'+metric].values]
                frames.append(dict(dataset=pool,source=source,architecture=model,seed=int(seed),run_id=f'{model}-{seed}',
                                   baseline=baseline,metric=metric,delta_auc=float(np.trapezoid(y,x=x)/maximum)))
    frame=pd.DataFrame(frames);write_csv(OUT/'diagnostic_auc_observations.csv',frame.to_dict('records'));diags=[]
    for metric in ('confusion','clusters'):
        part=frame[frame.metric==metric]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result=smf.mixedlm('delta_auc ~ C(dataset) * C(source) + C(baseline) + C(architecture)',part,groups=part.run_id,eval_env=-1).fit(
                    reml=True,method='lbfgs',maxiter=300,disp=False)
                covariance=float(np.asarray(result.cov_re)[0,0]);boundary=covariance<1e-8
                residual=np.asarray(result.resid);test=shapiro(residual)
                structural_warning=any('singular' in str(w.message).lower() or 'boundary' in str(w.message).lower() for w in caught)
                valid=bool(result.converged) and not boundary and not structural_warning and np.isfinite(result.fe_params).all()
                write_csv(OUT/f'auc_{metric}_coefficients.csv',[dict(parameter=k,estimate=float(v),standard_error=float(result.bse_fe[k]),
                    ci95_low=float(result.conf_int().loc[k,0]),ci95_high=float(result.conf_int().loc[k,1]),usable_sensitivity=valid)
                    for k,v in result.fe_params.items()])
                diags.append(dict(metric=metric,status='converged' if valid else 'fit_with_limitations',converged=bool(result.converged),
                    random_intercept_variance=covariance,boundary=boundary,residual_shapiro_p=float(test.pvalue),
                    structural_warning=structural_warning,usable_for_confirmatory_claim=False,
                    warnings=[str(w.message) for w in caught],scope='conditional sensitivity; repeated-source/baseline residual assumptions remain'))
            except Exception as error:
                diags.append(dict(metric=metric,status='fit_failed',error=f'{type(error).__name__}: {error}',warnings=[str(w.message) for w in caught]))
    write_json(OUT/'auc_fit_audits.json',diags)
    return diags


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.smoke:
        OUT=BASE/'statistics_smoke'
        original=observations
        def small(pools,source):
            frame=original(pools,source)
            unique=frame.drop_duplicates('image_id').sort_values('image_id')
            ids=unique.groupby(['dataset','class_name','risk_tier'],observed=True).head(1).image_id
            return frame[frame.image_id.isin(ids)].copy()
        globals()['observations']=small
    OUT.mkdir(parents=True,exist_ok=True)
    pools=[('B',B),('C',C)]+([('D',D)] if (D/'final_audit.json').exists() else [])
    specification=dict(saved_before_fit=now(),pools=[p for p,_ in pools],models=list(MODELS),seeds=list(SEEDS),
        formula='error ~ C(risk_tier)*C(dataset)+C(class_name)+C(architecture)+(1|run_id)+(1|image_id)',
        implementation='statsmodels variational Bayes; sparse two-entry random design',optimizer='L-BFGS-B',maxiter=300,
        reason_for_optimizer='bounded-memory deterministic implementation, selected before fitting',fe_p=2,vcp_p=1,
        interval='approximate posterior credible interval, not original frequentist CI or p-value',
        source='https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.html',
        reused_data_is_supplementary=True,retry_after_nonconvergence=False)
    write_json(OUT/'specification.json',specification)
    write_json(OUT/'status.json',dict(status='running',started_at=now(),pools=[p for p,_ in pools]))
    # Capture packages so numerical implementation is fully identifiable.
    import statsmodels,scipy
    write_json(OUT/'environment.json',dict(statsmodels=statsmodels.__version__,scipy=scipy.__version__,numpy=np.__version__))
    audits=[]
    for source in ('local','online'):
        try:audits.append(fit_risk(pools,source))
        except Exception as error:
            write_json(OUT/source/'fit_audit.json',dict(status='failed',error=f'{type(error).__name__}: {error}'))
            audits.append(dict(source=source,status='failed',error=str(error)))
    auc=fit_auc(pools)
    write_json(OUT/'status.json',dict(status='complete' if all(a.get('inference_allowed') for a in audits) and all(a['status']=='converged' for a in auc) else 'complete_with_fit_limitations',
        finished_at=now(),risk_audits=audits,auc_audits=auc,primary_tests_replaced=False,manual_review=False))


if __name__=='__main__':main()
