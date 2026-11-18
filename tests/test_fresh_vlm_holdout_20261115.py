from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_vlm_holdout_20261115 as held


class HeldoutValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=held.read(held.DATA/'qualified_manifest.csv')
        cls.targets=held.select_targets(cls.rows)

    def test_selection_is_deterministic_balanced_and_excludes_debug(self):
        again=held.select_targets(list(reversed(self.rows)))
        self.assertEqual([r['sample_id'] for r in self.targets],[r['sample_id'] for r in again])
        self.assertEqual(len(self.targets),60)
        self.assertFalse({r['sample_id'] for r in self.targets}&held.debug_sample_ids())
        counts={}
        for r in self.targets:counts[r['class_name'],r['role']]=counts.get((r['class_name'],r['role']),0)+1
        self.assertEqual(set(counts.values()),{5})

    def test_only_native_evidence_string_arrays_are_serialized_reversibly(self):
        job=next(j for j in held.format_pilot_jobs() if j['kind']=='native')
        value=dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],target_instance_id='target_1',
            features={},evidence={'x':['a','b']},untouched={'v':[1]})
        job={**job,'response_schema':{'properties':{'evidence':{'properties':{'x':{'type':'string'}}}}}}
        normalized,changes=held.normalize_evidence_arrays(value,job)
        self.assertEqual(json.loads(normalized['evidence']['x']),['a','b'])
        self.assertEqual(normalized['untouched'],value['untouched'])
        self.assertEqual(normalized['features'],value['features'])
        self.assertEqual(changes[0]['path'],'/evidence/x')

    def test_normalizer_rejects_non_string_or_empty_arrays(self):
        job={'kind':'native','response_schema':{'properties':{'evidence':{'properties':{'x':{'type':'string'}}}}}}
        for value in ([],[1],['a',1]):
            normalized,changes=held.normalize_evidence_arrays({'evidence':{'x':value}},job)
            self.assertEqual(normalized['evidence']['x'],value);self.assertEqual(changes,[])

    def test_feature_values_are_never_normalized(self):
        job={'kind':'native','response_schema':{'properties':{'evidence':{'properties':{}}}}}
        value={'features':{'field':['wrong-shape']},'evidence':{}}
        normalized,changes=held.normalize_evidence_arrays(value,job)
        self.assertEqual(normalized,value);self.assertEqual(changes,[])

    def test_existing_two_failures_become_valid_offline(self):
        jobs=held.format_pilot_jobs();fixed=0
        for job in jobs:
            folder=held.FORMAT_PILOT/'jobs'/f"{job['sample_id']}_{job['kind']}_B"
            if not (folder/'error.json').exists():continue
            value=held.response_value(json.loads((folder/'raw.json').read_text()),'B')
            outcome=held.validate_raw_or_normalized(value,job)
            self.assertEqual(outcome['status'],'normalized_valid');fixed+=1
        self.assertEqual(fixed,2)

    def test_call_budget_blocks_before_transport(self):
        job=held.format_pilot_jobs()[0]
        with tempfile.TemporaryDirectory() as d,patch.object(held,'OUT',Path(d)),patch.object(held,'post_json') as transport:
            for i in range(200):
                p=Path(d)/'jobs'/str(i)/'call_started.json';p.parent.mkdir(parents=True);p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'200-call budget'):held.execute(job,'A',dict(model='frozen',url='unused'))
            transport.assert_not_called()

    def test_incomplete_started_call_is_recorded_without_retry(self):
        job=held.format_pilot_jobs()[0];cfg=dict(model='frozen',url='unused')
        with tempfile.TemporaryDirectory() as d,patch.object(held,'OUT',Path(d)),patch.object(held,'post_json') as transport:
            folder=Path(d)/'jobs'/f"{job['sample_id']}_{job['kind']}_A";folder.mkdir(parents=True)
            binding=held.fingerprint(dict(job=job,source='A',model=cfg['model']))
            (folder/'call_started.json').write_text(json.dumps(dict(input_binding=binding)))
            result=held.execute(job,'A',cfg)
            self.assertEqual(result,dict(status='invalid',reason='started_without_complete_response'))
            self.assertEqual(json.loads((folder/'outcome.json').read_text()),result)
            transport.assert_not_called()


if __name__=='__main__':unittest.main()
