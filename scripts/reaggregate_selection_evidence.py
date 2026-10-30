#!/usr/bin/env python3
"""Reaggregate historical equal-error metrics; no embedding/cluster rebuild."""
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd


def holm(p):
    order=np.argsort(p);out=np.zeros(len(p));running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(p)-rank)*p[i]));out[i]=running
    return out


def main(inputs,out):
    out.mkdir(parents=True,exist_ok=True)
    for pool in ('B','C','D'):
        source=pd.read_csv(inputs/pool/'equal_error_metrics.csv')
        source=source[source.representation=='classifier']
        records=[]
        for (annotator,baseline),data in source.groupby(['source','baseline']):
            for metric in ('confusion','clusters'):
                group=[]
                for (model,seed),d in data.groupby(['model','seed']):
                    d=d.sort_values('budget')
                    x=np.r_[0,d.budget.to_numpy()];y=np.r_[0,d['delta_'+metric].to_numpy()]
                    value=float(np.trapezoid(y,x)/x[-1])
                    group.append(dict(model=model,seed=seed,value=value))
                group=pd.DataFrame(group);means=group.groupby('model').value.mean().to_numpy()
                assert len(means)==8 and len(group)==24
                null=np.abs((np.array(list(itertools.product([-1,1],repeat=8)))*means).mean(1))
                records.append(dict(source=annotator,baseline=baseline,metric=metric,
                    mean_delta_auc=float(group.value.mean()),
                    permutation_p=float((null>=abs(means.mean())-1e-15).mean())))
        result=pd.DataFrame(records)
        for family,column in [(None,'holm_all20_p'),(['random'],'holm_primary4_p'),
                              (['deepgini','dsa','kcenter'],'holm_strong12_p'),(['coverage_without_H'],'holm_ablation4_p')]:
            idx=result.index if family is None else result[result.baseline.isin(family)].index
            result.loc[idx,column]=holm(result.loc[idx,'permutation_p'].to_numpy())
        old=pd.read_csv(inputs/pool/'comparisons.csv')
        merge=result.merge(old,on=['source','baseline','metric'],suffixes=('_new','_old'))
        assert len(merge)==20
        for field in ['mean_delta_auc','permutation_p','holm_all20_p','holm_primary4_p','holm_strong12_p','holm_ablation4_p']:
            assert np.allclose(merge[field+'_new'],merge[field+'_old'],equal_nan=True,atol=1e-12),field
        result.to_csv(out/f'{pool}_comparisons.csv',index=False)
        print(pool,'20 effect estimates and all four correction families match retained outputs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.input,a.output)
