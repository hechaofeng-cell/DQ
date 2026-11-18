from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_vlm_strict_schema_repair_20261115 as repair


class StrictSchemaRepairTests(unittest.TestCase):
    def test_exactly_four_parent_failures_are_frozen(self):
        failures=repair.parent_failures()
        self.assertEqual(len(failures),4)
        self.assertEqual({x['job']['sample_id'] for x in failures},{
            'fresh-obj365-train-1080301-car','fresh-obj365-train-1291311-dog',
            'fresh-obj365-train-1330653-car','fresh-obj365-train-1434540-car'})

    def test_strict_body_uses_server_schema_and_system_mapping(self):
        job=repair.parent_failures()[0]['job']
        body=repair.strict_body(job,dict(model='frozen',request_parameters={'top_p':0.8}))
        self.assertEqual(body['response_format']['type'],'json_schema')
        self.assertTrue(body['response_format']['json_schema']['strict'])
        self.assertEqual(body['response_format']['json_schema']['schema'],job['response_schema'])
        self.assertEqual(body['messages'][0]['role'],'system')
        self.assertIn('mixed_or_other',body['messages'][0]['content'])
        self.assertIn('interacting',body['messages'][0]['content'])
        self.assertEqual(body['temperature'],0)

    def test_all_evidence_hashes_still_match(self):
        for item in repair.parent_failures():
            job=item['job']
            self.assertEqual([repair.sha(path) for path in job['evidence_paths']],job['evidence_sha256'])


if __name__=='__main__':unittest.main()
