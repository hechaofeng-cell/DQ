"""Recheck frozen conditions on historical Objects365; no fresh-confirmation claim."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import conditional_coverage_common as cc


def external_contexts(out,contexts):
    original=cc.old.OUT/'external_manifest.csv'
    rows=cc.read(original)
    hashes={str(original):cc.sha(original)}
    external={}
    for c,ctx in contexts.items():
        cr=[r for r in rows if r['class_name']==c]
        states=ctx['states']
        X=np.zeros((len(cr),len(states)),bool)
        dictionary=cc.old.OUT/'dictionaries'/f'{c}.json'
        hashes[str(dictionary)]=cc.sha(dictionary)
        for i,row in enumerate(cr):
            sid=row['sample_id']
            domain='car' if c=='car' else 'old'
            native_path=cc.old.EXTERNAL/f'annotations/local/{domain}/parsed/{sid}.json'
            specific_path=cc.old.OUT/f'annotations/external/{c}/{sid}_parsed.json'
            native=json.loads(native_path.read_text())
            specific=json.loads(specific_path.read_text())
            assert native['image_id']==sid
            assert specific['original_sha256']==row['image_sha256']
            assert specific['dictionary_sha256']==hashes[str(dictionary)]
            hashes[str(native_path)]=cc.sha(native_path)
            hashes[str(specific_path)]=cc.sha(specific_path)
            for j,(ns,field,value) in enumerate(states):
                if ns=='shared' and field=='target_scale':
                    area=float(row['bbox_area_ratio'])
                    v='tiny' if area<.01 else 'small' if area<.03 else 'medium' if area<.1 else 'large'
                else:
                    v=(native['features'] if ns=='shared' else specific['features'])[field]
                X[i,j]=value in v if isinstance(v,list) else value==v
        # Independently reconstruct rather than blindly trust the historical matrix.
        old_matrix=cc.old.OUT/f'external_{c}_matrix.npz'
        with np.load(old_matrix,allow_pickle=False) as z:
            assert np.array_equal(z['matrix'].astype(bool),X)
            assert z['sample_ids'].astype(str).tolist()==[r['sample_id'] for r in cr]
        hashes[str(old_matrix)]=cc.sha(old_matrix)
        external[c]=dict(rows=cr,matrix=X,states=states)
        np.savez_compressed(out/f'external_{c}_matrix.npz',
                            sample_ids=np.array([r['sample_id'] for r in cr]),matrix=X)
    cc.write(out/'external_manifest.csv',rows)
    cc.dump(out/'external_annotation_audit.json',dict(status='pass',
        reference_role='historical_reanalysis_not_new_confirmation',files=hashes,
        independent_matrix_reconstructions=4,automatic_annotations_semantic_accuracy_unknown=True))
    return rows,external


def confirm(out):
    cc.verify_freeze(out/'conditions_before_external_prediction_freeze.json')
    assert not (out/'external_condition_results.csv').exists(),'Reference outputs already exist.'
    contexts=cc.load_contexts()
    rows,external=external_contexts(out,contexts)
    conditions=cc.read(out/'discovered_conditions.csv')
    groups=cc.read(out/'discovery_groups.csv')
    # Only after every discovery condition has been frozen do we open D outputs.
    pred,predhash=cc.old.predictions('D',rows)
    plan=json.loads((out/'protocol.json').read_text())
    B=plan['statistics']['bootstrap_replicates']
    rng=np.random.default_rng(plan['statistics']['bootstrap_seed'])
    cache={};bootstrap={};bootstrap_evaluable={};bootstrap_effect={};unique=[]
    bootstrap_records=[]
    for ci,(c,ctx) in enumerate(external.items()):
        X,control=cc.condition_masks(ctx)
        cr=ctx['rows']
        images=sorted({r['image_id'] for r in cr})
        image_index={im:i for i,im in enumerate(images)}
        image_for_row=np.array([image_index[r['image_id']] for r in cr])
        weights=rng.multinomial(len(images),np.full(len(images),1/len(images)),size=B)
        row_weights=weights[:,image_for_row]
        row_presence=(weights>0).astype(np.int16)
        image_X=np.zeros((len(images),X.shape[1]),bool)
        image_control=np.zeros_like(image_X)
        for i,im in enumerate(image_for_row):
            image_X[im] |= X[i]
            image_control[im] |= control[i]
        n1=X.sum(0);n0=control.sum(0)
        u1=image_X.sum(0);u0=image_control.sum(0)
        b1=row_weights@X;b0=row_weights@control
        bu1=row_presence@image_X;bu0=row_presence@image_control
        eligibility=(n1>=20)&(n0>=20)&(u1>=20)&(u0>=20)
        beligibility=(b1>=20)&(b0>=20)&(bu1>=20)&(bu0>=20)
        positions=[i for i,r in enumerate(rows) if r['class_name']==c]
        for model in cc.old.MODELS:
            correct_by_seed=(pred[model][:,positions]==cc.old.ALL_CLASSES.index(c))
            correct=correct_by_seed.mean(0)
            r1=np.divide(X.T@correct,n1,out=np.full(len(n1),np.nan),where=n1>0)
            r0=np.divide(control.T@correct,n0,out=np.full(len(n0),np.nan),where=n0>0)
            delta=100*(r1-r0)
            br1=np.divide(row_weights@(X*correct[:,None]),b1,
                          out=np.full(b1.shape,np.nan),where=b1>0)
            br0=np.divide(row_weights@(control*correct[:,None]),b0,
                          out=np.full(b0.shape,np.nan),where=b0>0)
            effects=100*(br1-br0)
            indices=sorted({int(r['state_index']) for r in conditions if r['class_name']==c and r['model']==model})
            for j in indices:
                key=(c,model,j)
                status=('insufficient_support' if not eligibility[j] else
                        'confirmed' if delta[j]<=-10+cc.TOL else 'not_confirmed')
                per_seed=[]
                for correct_seed in correct_by_seed:
                    if min(n1[j],n0[j])>0:
                        per_seed.append(float(100*(correct_seed[X[:,j]].mean()-correct_seed[control[:,j]].mean())))
                valid_effects=effects[:,j][beligibility[:,j]&np.isfinite(effects[:,j])]
                low,high=np.quantile(valid_effects,[.025,.975]) if len(valid_effects) else [None,None]
                cache[key]=dict(reference_state_targets=int(n1[j]),reference_control_targets=int(n0[j]),
                    reference_state_images=int(u1[j]),reference_control_images=int(u0[j]),
                    reference_state_recall=float(r1[j]) if n1[j] else '',
                    reference_control_recall=float(r0[j]) if n0[j] else '',
                    reference_delta_pp=float(delta[j]) if np.isfinite(delta[j]) else '',
                    status=status,all_prediction_seeds_negative=bool(len(per_seed)==3 and max(per_seed)<0),
                    all_prediction_seeds_gap_10pp=bool(len(per_seed)==3 and max(per_seed)<=-10+cc.TOL),
                    eligible_bootstrap_replicates=len(valid_effects),
                    bootstrap_effect_p025_pp=float(low) if low is not None else '',
                    bootstrap_effect_p975_pp=float(high) if high is not None else '')
                bootstrap[key]=beligibility[:,j]&(effects[:,j]<=-10+cc.TOL)
                bootstrap_evaluable[key]=beligibility[:,j]
                bootstrap_effect[key]=np.where(beligibility[:,j],effects[:,j],np.nan)
                unique.append(key)
        cc.log(out,'historical_reference_class_complete',class_name=c,images=len(images),
               bootstrap_replicates=B)
    details=[dict(**r,**cache[(r['class_name'],r['model'],int(r['state_index']))]) for r in conditions]
    detailfields=list(dict.fromkeys(k for r in details for k in r))
    cc.write(out/'external_condition_results.csv',details,detailfields)
    grouped=defaultdict(list)
    for r in details:
        grouped[(r['method'],int(r['seed']),r['scope'])].append(r)
    run_rows=[];boot_runs={};effect_sensitivity=[];dedup_rows=[]
    setups=[(m,-1) for m in ['G','SR','KC']]
    setups += [(m,s) for s in plan['phases']['retrospective']['selection_seeds'] for m in ['R','CS','AS']]
    for scope in ['all','specific']:
        for method,seed in setups:
            rr=grouped[method,seed,scope]
            proposed=len(rr)
            confirmed=[r for r in rr if r['status']=='confirmed']
            evaluable=[r for r in rr if r['status']!='insufficient_support']
            dedup={(r['class_name'],r['namespace'],r['field'],r['value']) for r in confirmed}
            run_rows.append(dict(method=method,seed=seed,scope=scope,proposed=proposed,
                evaluable=len(evaluable),confirmed=len(confirmed),not_confirmed=len(evaluable)-len(confirmed),
                insufficient=proposed-len(evaluable),deduplicated_confirmed=len(dedup),
                evaluable_classes=len({r['class_name'] for r in evaluable}),
                confirmation_rate=len(confirmed)/len(evaluable) if evaluable else '',
                insufficient_rate=(proposed-len(evaluable))/proposed if proposed else ''))
            for c,ns,field,value in sorted(dedup):
                dedup_rows.append(dict(method=method,seed=seed,scope=scope,class_name=c,
                                       namespace=ns,field=field,value=value))
            keys=[(r['class_name'],r['model'],int(r['state_index'])) for r in rr]
            boot_runs[f'{method}__s{seed}__{scope}']=(np.stack([bootstrap[k] for k in keys]).sum(0)
                                                    if keys else np.zeros(B,int))
            for gap in [5,10,20]:
                effect_sensitivity.append(dict(method=method,seed=seed,scope=scope,reference_gap_pp=gap,
                    proposed=proposed,evaluable=len(evaluable),confirmed=sum(
                        float(r['reference_delta_pp'])<=-gap+cc.TOL for r in evaluable)))
    cc.write(out/'external_diagnostic_runs.csv',run_rows)
    cc.write(out/'deduplicated_conditions.csv',dedup_rows,
             ['method','seed','scope','class_name','namespace','field','value'])
    cc.write(out/'reference_gap_sensitivity.csv',effect_sensitivity)
    np.savez_compressed(out/'reference_bootstrap_confirmed_counts.npz',**boot_runs)
    effect_arrays={f'{c}__{model}__j{j}':bootstrap_effect[c,model,j] for c,model,j in unique}
    np.savez_compressed(out/'reference_bootstrap_condition_effects.npz',**effect_arrays)
    cc.dump(out/'bootstrap_configuration.json',dict(replicates=B,
        seed=plan['statistics']['bootstrap_seed'],unit='original_image',stratified_by='class',
        paired_across_methods=True,recompute_target_and_unique_image_support=True,
        resampled_duplicate_images_do_not_count_as_distinct=True,
        discovery_conditions_frozen=True,scope='historical_reference_only',
        missing_effect='NaN_not_zero',nested_selection_seed_resampling=False,
        inference_boundary='conditional_on_historical_discovery_pool_dictionary_and_conditions'))
    cc.dump(out/'external_confirmation_audit.json',dict(status='pass',
        historical_reference=True,new_independent_confirmation=False,
        reference_targets=len(rows),source_matrices_independently_rebuilt=True,
        reference_prediction_hashes=predhash,implementation={str(Path(__file__)):cc.sha(__file__)},
        no_new_model_inference=True,condition_freeze_verified=True))
    cc.log(out,'historical_confirmation_complete',conditions=len(details),new_independent_confirmation=False)


if __name__=='__main__':
    args=cc.parsed_args(__doc__).parse_args()
    confirm(args.out)
