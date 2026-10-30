"""External-source parameters and audited reuse of immutable COCO algorithms."""
from __future__ import annotations
import inspect
import textwrap
from pathlib import Path
import independent_confirmation_common as common

ROOT=common.ROOT
COCO_B=common.BASE
BASE=ROOT/'artifacts/sixclass_voc_external_confirmation_20261005'
EVIDENCE=ROOT/'artifacts/empirical_evidence_strengthening_20261005'
PROTOCOL=ROOT/'docs/sixclass_voc_external_confirmation_protocol_20261005.md'
ARCHIVE_DIR=common.PROJECT/'data/pascal_voc2007_external_confirmation_v1/raw'
ARCHIVES=(ARCHIVE_DIR/'VOCtrainval_06-Nov-2007.tar',ARCHIVE_DIR/'VOCtest_06-Nov-2007.tar')
N=100
TIER_N=20
STEP=10
TOTAL=600
BUDGETS=(10,20,30,40,50)

def adapt(module,name,replacements):
    """Exact guarded substitutions of sample-size constants, never scientific scores."""
    source=textwrap.dedent(inspect.getsource(getattr(module,name)))
    for before,after in replacements:
        if before not in source:raise ValueError(f'Adapter source mismatch: {module.__name__}.{name}: {before}')
        source=source.replace(before,after)
    module.__dict__.update(N=N,TIER_N=TIER_N,STEP=STEP,TOTAL=TOTAL,BUDGETS=BUDGETS)
    exec(compile(source,f'{__file__}:{module.__name__}.{name}','exec'),module.__dict__)

def activate():
    common.BASE=BASE;common.PROTOCOL=PROTOCOL
    import analyze_dual_vlm_study as a
    adapt(a,'native_labels',[('!= 2400','!= TOTAL'),('requires 2400','requires the registered target count')])
    adapt(a,'contexts',[('!= 2400','!= TOTAL')])
    import freeze_independent_sixclass_selection as selection
    selection.BASE=BASE;selection.PROTOCOL=PROTOCOL
    adapt(selection,'largest_remainder',[('total=20','total=STEP'),('/80','/TIER_N')])
    adapt(selection,'constrained',[('range(20)','range(STEP)')])
    adapt(selection,'build',[('range(400)','range(N)'),('t*80:(t+1)*80','t*TIER_N:(t+1)*TIER_N'),
          ('i//20','i//STEP'),('targets=2400','targets=TOTAL')])
    import analyze_sixclass_difficulty as supplement
    adapt(supplement,'build_orders',[('range(400)','range(N)'),('i*80:(i+1)*80','i*TIER_N:(i+1)*TIER_N'),
          ('range(20)','range(STEP)'),('len(set(picked))==20','len(set(picked))==STEP'),
          ('candidates=80,selected=20','candidates=TIER_N,selected=STEP'),('i//20','i//STEP')])
    adapt(supplement,'run',[('n=480','n=6*TIER_N'),('budget=tier*20','budget=tier*STEP'),
          ('budget-20:budget','budget-STEP:budget'),('n=120','n=6*STEP'),('targets=2400','targets=TOTAL'),
          ("r['budget']==100","r['budget']==BUDGETS[-1]"),
          ('selected_endpoint_n=600','selected_endpoint_n=6*BUDGETS[-1]')])
    import analyze_independent_sixclass as analysis
    analysis.BASE=BASE;analysis.PROTOCOL=PROTOCOL
    adapt(analysis,'prediction',[('(2400,6)','(TOTAL,6)')])
    adapt(analysis,'inference',[('(2400,6)','(TOTAL,6)'),('targets=2400','targets=TOTAL')])
    adapt(analysis,'confirm',[('prospective_independent_sample_confirmation','prospective_external_source_confirmation'),
          ('new samples; same COCO dataset source','unused VOC2007 samples; source distinct from COCO training')])
    adapt(analysis,'source_bootstrap',[('(2000,80)','(2000,TIER_N)'),('np.arange(6),80','np.arange(6),TIER_N')])
    adapt(analysis,'audit_order',[('==100','==BUDGETS[-1]'),('t*20:(t+1)*20','t*STEP:(t+1)*STEP')])
    adapt(analysis,'compare',[('(20,40,60,80,100)','BUDGETS')])
    adapt(analysis,'strong_image_bootstrap',[('multinomial(400,np.full(400,1/400)','multinomial(N,np.full(N,1/N)'),
          ('(20,40,60,80,100)','BUDGETS'),('np.arange(0,101,20)','np.array([0,*BUDGETS])'),('/100)','/BUDGETS[-1])')])
    adapt(analysis,'transfer',[('np.zeros(400','np.zeros(N'),('A_B_kendall_tau','A_external_kendall_tau')])
    adapt(analysis,'robust_risks',[('range(400)','range(N)'),('t*80:(t+1)*80','t*TIER_N:(t+1)*TIER_N')])
    a.stats.BUDGETS=BUDGETS;a.b.BUDGETS=BUDGETS;a.control.BUDGETS=BUDGETS
    return a,selection,analysis

def code_paths():
    names=('voc_external_common.py','download_sixclass_voc_sources.py','prepare_sixclass_voc_external.py','run_sixclass_voc_external_confirmation.py',
           'analyze_sixclass_voc_external.py','strengthen_empirical_evidence.py','independent_confirmation_common.py',
           'prepare_independent_sixclass_pool.py','annotate_independent_sixclass_pool.py','freeze_independent_sixclass_selection.py',
           'analyze_independent_sixclass.py','analyze_sixclass_difficulty.py','run_sixclass_independent_confirmation.py',
           'annotate_dual_vlm.py','annotate_multiclass_online_sensenova.py','run_dog_local_pipeline.py',
           'analyze_dual_vlm_study.py','run_class_stratified_error_controls.py','run_coverage_incremental_value.py')
    return [PROTOCOL,*[ROOT/'scripts'/n for n in names]]
