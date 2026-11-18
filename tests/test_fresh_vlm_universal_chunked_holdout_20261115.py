from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_vlm_universal_chunked_holdout_20261115 as run


class UniversalChunkedTests(unittest.TestCase):
    def test_pilot_contains_horse_and_person_http400_fine_jobs(self):
        jobs=run.prior_pilot_jobs();self.assertEqual([(j['class_name'],j['kind']) for j in jobs],[('horse','fine'),('person','fine')])

    def test_every_schema_splits_into_two_bounded_disjoint_parts(self):
        source=json.loads((run.PARENT/'inputs.json').read_text())
        for job in source['fresh_jobs']:
            parts=run.field_chunks(job);self.assertEqual(len(parts),2);self.assertLessEqual(max(map(len,parts)),10)
            self.assertFalse(set(parts[0])&set(parts[1]))

    def test_fine_component_parse_and_merge_preserves_values(self):
        job=run.prior_pilot_jobs()[0];values={}
        for fields in run.field_chunks(job):
            props=job['response_schema']['properties'];wire={k:props[k]['enum'][0] if 'enum' in props[k] else [props[k]['items']['enum'][0]] for k in fields}
            response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(wire)}}]}
            parsed,returned,changes=run.parse_component(response,job,fields);values.update(parsed['values'])
            self.assertEqual(returned,wire);self.assertEqual(changes,[])
        run.held.validate_result(values,job)

    def test_third_holdout_is_balanced_and_disjoint(self):
        targets=run.select_targets(run.read(run.DATA/'qualified_manifest.csv'))
        self.assertEqual(len(targets),60);self.assertFalse({r['sample_id'] for r in targets}&run.excluded_ids())
        counts={}
        for row in targets:counts[row['class_name'],row['role']]=counts.get((row['class_name'],row['role']),0)+1
        self.assertEqual(set(counts.values()),{5})


if __name__=='__main__':unittest.main()
