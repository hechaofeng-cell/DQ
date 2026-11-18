from pathlib import Path
import base64
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import smoke_fresh_dual_vlm_20261112 as smoke
from continue_fresh_pipeline_20261112 import alive


class ContinuationTests(unittest.TestCase):
    def test_both_backends_receive_identical_native_task_and_image_bytes(self):
        # Test actual original schemas/templates and real evidence already source-verified locally.
        rows=smoke.read(smoke.DATA/'materialized_targets.csv')
        row=next(r for r in rows if r['class_name']=='dog')
        job=smoke.definition(row,'native')
        a=smoke.request_body(job,'A','frozen-A')
        b=smoke.request_body(job,'B','frozen-B',{'temperature':.7,'max_tokens':1})
        am=a['messages'][0];bm=b['messages'][0]['content']
        self.assertEqual(am['content'],bm[0]['text'])
        self.assertEqual(am['images'],[i['image_url']['url'].split(',',1)[1] for i in bm[1:]])
        self.assertEqual([base64.b64decode(x) for x in am['images']],
            [Path(p).read_bytes() for p in job['evidence_paths']])
        self.assertEqual(a['format'],job['response_schema'])
        self.assertEqual(b['temperature'],0)
        self.assertEqual(b['max_tokens'],2600)
        self.assertNotIn('assistant',[m['role'] for m in b['messages']])

    def test_data_gate_blocks_before_any_transport_call(self):
        with tempfile.TemporaryDirectory() as d,patch.object(smoke,'DATA',Path(d)),patch.object(smoke,'post_json') as transport:
            (Path(d)/'qualification_summary.json').write_text(json.dumps(dict(status='pending',data_qualified=False)))
            with self.assertRaises((ValueError,FileNotFoundError)):smoke.main()
            transport.assert_not_called()

    def test_fine_renderer_and_backend_contract_on_real_official_target(self):
        rows=smoke.read(smoke.DATA/'materialized_targets.csv')
        row=next(r for r in rows if r['class_name']=='person')
        with tempfile.TemporaryDirectory() as d,patch.object(smoke,'OUT',Path(d)):
            job=smoke.definition(row,'fine')
            self.assertEqual(Path(job['evidence_paths'][0]).read_bytes(),smoke.contact(row,0))
            a=smoke.request_body(job,'A','A');b=smoke.request_body(job,'B','B')
            self.assertEqual(a['messages'][0]['content'],b['messages'][0]['content'][0]['text'])
            self.assertEqual(a['options']['num_ctx'],16384)
            self.assertEqual(b['max_tokens'],2000)
            synthetic={f['feature_id']:('unknown' if f['value_type']=='enum' else ['unknown']) for f in job['semantic_schema']['features']}
            smoke.validate_output(synthetic,job)
            with self.assertRaises(Exception):smoke.validate_output({**synthetic,'invented_field':'unknown'},job)

    def test_lost_response_consumes_attempt_and_does_not_reissue(self):
        with tempfile.TemporaryDirectory() as d,patch.object(smoke,'OUT',Path(d)),patch.object(smoke,'post_json') as transport:
            image=Path(d)/'image.jpg';image.write_bytes(b'not-sent')
            job=dict(sample_id='target',class_name='dog',kind='native',prompt='frozen',response_schema={},
                evidence_paths=[str(image)],evidence_sha256=['bound'],options=dict(num_predict=2600))
            cfg=dict(model='frozen-A',url='http://never-called')
            binding=smoke.fingerprint(dict(job=job,source='A',model=cfg['model']))
            folder=Path(d)/'jobs/target_native_A';folder.mkdir(parents=True)
            for i in range(1,4):(folder/f'attempt{i}_started.json').write_text(json.dumps(dict(input_binding=binding)))
            with self.assertRaisesRegex(RuntimeError,'at most3 attempts'):smoke.run_job(job,'A',cfg,[0])
            transport.assert_not_called()

    def test_auth_error_stops_without_retry_and_redacts_secret(self):
        with tempfile.TemporaryDirectory() as d,patch.object(smoke,'OUT',Path(d)),patch.object(smoke,'post_json',return_value=(401,'secret-test-key',.1,{})) as transport:
            image=Path(d)/'image.jpg';image.write_bytes(b'fake-evidence')
            job=dict(sample_id='target',kind='fine',prompt='frozen',response_schema={},
                evidence_paths=[str(image)],evidence_sha256=['bound'],options=dict(num_predict=2000))
            with self.assertRaises(smoke.AccessFailure):
                smoke.run_job(job,'B',dict(model='frozen-B',url='fake',api_key='secret-test-key'),[0])
            self.assertEqual(transport.call_count,1)
            raw=Path(d)/'jobs/target_fine_B/attempt1_raw.json'
            self.assertEqual(raw.read_text(),'[REDACTED]')

    def test_smoke_targets_are_first_qualified_selection_ranks_only(self):
        rows=[dict(class_name=c,role=role,rank=rank) for c in smoke.CLASSES for role,rank in
            [('reference','1'),('selection','3'),('selection','2')]]
        chosen=smoke.choose_targets(rows)
        self.assertEqual([r['class_name'] for r in chosen],smoke.CLASSES)
        self.assertTrue(all(r['role']=='selection' and r['rank']=='2' for r in chosen))

    def test_pid_reuse_and_other_commands_are_not_running_acquisition(self):
        with patch('continue_fresh_pipeline_20261112.identity',return_value=dict(start_ticks='123',command=[b'other.py'])):
            self.assertFalse(alive(1,'123'))
        with patch('continue_fresh_pipeline_20261112.identity',return_value=dict(start_ticks='124',command=[b'scripts/acquire_fresh_images_20261112.py'])):
            self.assertFalse(alive(1,'123'))


if __name__=='__main__':unittest.main()
