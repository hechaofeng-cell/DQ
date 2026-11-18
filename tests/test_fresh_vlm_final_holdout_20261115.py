from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_vlm_final_holdout_20261115 as run


class FinalHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=json.loads((run.PARENT/'inputs.json').read_text())

    def test_parent_replays_100_of_100_with_one_self_reference_drop(self):
        audit=run.offline_parent_audit()
        self.assertEqual((audit['valid'],audit['invalid']),(100,0));self.assertEqual(audit['self_reference_drops'],1)

    def test_self_reference_rule_rejects_other_extras_and_missing_keys(self):
        job=next(j for j in self.parent['fresh_jobs'] if j['kind']=='native');fields=run.transport.field_chunks(job)[0]
        raw=run.PARENT/'fresh/components'/f"{job['sample_id']}_{job['kind']}_B_part1"/'raw.json'
        response=json.loads(raw.read_text());wire=run.held.response_value(response,'B')
        for mutate in ('other_extra','missing'):
            changed=json.loads(json.dumps(wire));evidence=json.loads(changed['evidence_json'])
            if mutate=='other_extra':evidence['bad_key']='x'
            else:evidence.pop(fields[0])
            changed['evidence_json']=json.dumps(evidence)
            outer={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(changed)}}]}
            with self.assertRaises(ValueError):run.parse_component(outer,job,fields)

    def test_fourth_targets_are_balanced_and_disjoint(self):
        targets=run.select_targets(run.read(run.DATA/'qualified_manifest.csv'))
        self.assertEqual(len(targets),60);self.assertFalse({r['sample_id'] for r in targets}&run.excluded_ids())
        counts={}
        for row in targets:counts[row['class_name'],row['role']]=counts.get((row['class_name'],row['role']),0)+1
        self.assertEqual(set(counts.values()),{5})

    def test_every_job_uses_two_bounded_components(self):
        targets=run.select_targets(run.read(run.DATA/'qualified_manifest.csv'));jobs=run.make_jobs(targets)
        for job in jobs:
            parts=run.transport.field_chunks(job);self.assertEqual(len(parts),2);self.assertLessEqual(max(map(len,parts)),10)


if __name__=='__main__':unittest.main()
