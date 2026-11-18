from collections import Counter
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_dual_vlm_full_annotation_20261115 as run


class FullAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=run.read(run.MANIFEST)

    def test_manifest_and_job_counts_are_frozen(self):
        self.assertEqual(len(self.rows),4800)
        self.assertEqual(Counter((row['class_name'],row['role']) for row in self.rows),
            Counter({(class_name,role):400 for class_name in run.CLASSES for role in run.ROLES}))
        self.assertEqual(sum(len(run.kinds(row)) for row in self.rows),8000)

    def test_representative_jobs_have_bounded_two_part_transport(self):
        for class_name in run.CLASSES:
            row=next(row for row in self.rows if row['class_name']==class_name)
            for kind in run.kinds(row):
                job=run.build_job(row,kind);chunks=run.transport.field_chunks(job)
                self.assertEqual(len(chunks),2);self.assertLessEqual(max(map(len,chunks)),10)
                for path,digest in zip(job['evidence_paths'],job['evidence_sha256']):
                    self.assertEqual(run.sha(path),digest)

    def test_physical_call_caps_cover_three_semantic_attempts(self):
        self.assertEqual(run.MAX_CALLS['A'],run.SEMANTIC_JOBS*run.MAX_ATTEMPTS)
        self.assertEqual(run.MAX_CALLS['B'],run.SEMANTIC_JOBS*run.MAX_ATTEMPTS*2)

    def test_final_gate_is_complete(self):
        gate=json.loads((run.FINAL_GATE/'independent_audit.json').read_text())
        self.assertTrue(gate['success_criterion_met']);self.assertEqual((gate['valid'],gate['invalid']),(100,0))


if __name__=='__main__':unittest.main()
