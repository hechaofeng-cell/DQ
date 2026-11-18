from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import validate_fresh_vlm_format_repair_20261114 as repair


class FormatRepairTests(unittest.TestCase):
    def test_schema_is_explicit_and_identical_for_both_backends(self):
        old=json.loads((repair.original.OUT/'inputs.json').read_text())['jobs'][0]
        job={**old,'prompt':repair.explicit_schema_prompt(old)}
        prefix=old['prompt']
        self.assertTrue(job['prompt'].startswith(prefix))
        self.assertIn(json.dumps(old['response_schema'],ensure_ascii=False,separators=(',',':')),job['prompt'])
        a=repair.original.request_body(job,'A','A');b=repair.original.request_body(job,'B','B')
        self.assertEqual(a['messages'][0]['content'],b['messages'][0]['content'][0]['text'])
        self.assertEqual(a['messages'][0]['images'],[v['image_url']['url'].split(',',1)[1] for v in b['messages'][0]['content'][1:]])

    def test_actual_sheep413_payload_fits_without_changing_target_geometry(self):
        job=next(j for j in json.loads((repair.original.OUT/'inputs.json').read_text())['jobs']
            if j['class_name']=='sheep' and j['kind']=='native')
        job={**job,'prompt':repair.explicit_schema_prompt(job)}
        with tempfile.TemporaryDirectory() as d,patch.object(repair,'OUT',Path(d)):
            evidence=repair.prepare_evidence(job)
            self.assertLessEqual(evidence['serialized_body_bytes'],repair.MAX_BODY_BYTES)
            self.assertEqual(evidence['source_sha256'],job['evidence_sha256'])
            self.assertEqual(evidence['mode'],'bounded_JPEG_transport')
            for before,after in zip(job['evidence_paths'],evidence['paths']):
                with Image.open(before) as old,Image.open(after) as new:
                    self.assertLessEqual(max(new.size),2048)
                    self.assertLess(abs(old.width/old.height-new.width/new.height),.002)
            again=repair.prepare_evidence(job)
            self.assertEqual(again,evidence)

    def test_small_evidence_bytes_remain_original(self):
        job=json.loads((repair.original.OUT/'inputs.json').read_text())['jobs'][0]
        with tempfile.TemporaryDirectory() as d,patch.object(repair,'OUT',Path(d)):
            evidence=repair.prepare_evidence(job)
            self.assertEqual(evidence['mode'],'unchanged')
            self.assertEqual(evidence['paths'],job['evidence_paths'])
            self.assertEqual(evidence['sha256'],job['evidence_sha256'])

    def test_exhausted_or_interrupted_new_job_cannot_issue_another_call(self):
        job=json.loads((repair.original.OUT/'inputs.json').read_text())['jobs'][0]
        cfg=dict(model='frozen',url='never-called')
        with tempfile.TemporaryDirectory() as d,patch.object(repair,'OUT',Path(d)),patch.object(repair,'post_json') as transport:
            folder=Path(d)/'jobs'/f"{job['sample_id']}_{job['kind']}_B";folder.mkdir(parents=True)
            (folder/'call_started.json').write_text(json.dumps(dict(input_binding=repair.fingerprint(dict(job=job,source='B',model='frozen')))))
            result=repair.execute_one(job,'B',cfg)
            self.assertFalse(result['valid']);transport.assert_not_called()

    def test_invalid_type_is_rejected_without_array_string_coercion(self):
        job=json.loads((repair.original.OUT/'inputs.json').read_text())['jobs'][0]
        old=repair.original.OUT/'jobs'/f"{job['sample_id']}_{job['kind']}_B"/'attempt1_raw.json'
        with self.assertRaises(Exception):repair.validate_returned(json.loads(old.read_text()),job,'B')

    def test_new_pilot_limit_is_enforced_before_transport(self):
        job=json.loads((repair.original.OUT/'inputs.json').read_text())['jobs'][0]
        with tempfile.TemporaryDirectory() as d,patch.object(repair,'OUT',Path(d)),patch.object(repair,'post_json') as transport:
            for i in range(20):
                p=Path(d)/'jobs'/f'other{i}'/'call_started.json';p.parent.mkdir(parents=True);p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'20-call budget'):repair.execute_one(job,'A',dict(model='frozen',url='unused'))
            transport.assert_not_called()


if __name__=='__main__':unittest.main()
