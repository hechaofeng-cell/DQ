from pathlib import Path
import json
import sys
import unittest

from jsonschema import Draft7Validator

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_vlm_flat_strict_completion_20261115 as run


class FlatStrictCompletionTests(unittest.TestCase):
    def test_two_native_pilot_jobs_are_frozen(self):
        jobs=run.pilot_jobs()
        self.assertEqual([(j['sample_id'],j['kind']) for j in jobs],[
            ('fresh-obj365-train-1291311-dog','native'),('fresh-obj365-train-1330653-car','native')])

    def test_flat_schema_has_no_nested_objects(self):
        job=run.pilot_jobs()[0];schema=run.flat_schema(job)
        self.assertNotIn('features',schema['properties']);self.assertNotIn('evidence',schema['properties'])
        self.assertEqual(schema['properties']['evidence_json']['type'],'string')
        self.assertTrue(all(v.get('type')!='object' for v in schema['properties'].values()))

    def test_flat_reconstruction_is_valid_and_reversible(self):
        job=run.pilot_jobs()[0];feature_props=job['response_schema']['properties']['features']['properties']
        value=dict(schema_version=job['semantic_schema']['schema_version'],image_id=job['sample_id'],target_instance_id='target_1')
        for key,schema in feature_props.items():
            value[key]=[schema['items']['enum'][0]] if schema['type']=='array' else schema['enum'][0]
        evidence={k:'visible evidence' for k in job['response_schema']['properties']['evidence']['properties']}
        value['evidence_json']=json.dumps(evidence,ensure_ascii=False,separators=(',',':'))
        Draft7Validator(run.flat_schema(job)).validate(value)
        standard=run.reconstruct_flat(value,job)
        self.assertEqual(standard['evidence'],evidence)
        self.assertEqual(standard['features'],{k:value[k] for k in feature_props})

    def test_fresh_targets_are_balanced_and_unseen(self):
        targets=run.select_targets(run.read(run.DATA/'qualified_manifest.csv'))
        self.assertEqual(len(targets),60);self.assertFalse({r['sample_id'] for r in targets}&run.excluded_ids())
        counts={}
        for row in targets:counts[row['class_name'],row['role']]=counts.get((row['class_name'],row['role']),0)+1
        self.assertEqual(set(counts.values()),{5})


if __name__=='__main__':unittest.main()
