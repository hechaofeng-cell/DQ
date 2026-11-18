from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_vlm_chunked_normalized_holdout_20261115 as run


class ChunkedNormalizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=json.loads((run.SOURCE/'inputs.json').read_text());cls.job=cls.source['pilot_jobs'][0]

    def test_prior_two_pilots_replay_valid(self):
        with self.subTest('offline replay'):
            audit=run.offline_pilot_audit(self.source['pilot_jobs'])
            self.assertEqual((audit['valid'],audit['invalid']),(2,0));self.assertGreaterEqual(audit['normalization_fields'],2)

    def test_only_nonempty_string_arrays_are_serialized(self):
        fields=run.chunk.feature_chunks(self.job)[0];props=self.job['response_schema']['properties']['features']['properties']
        wire={}
        for key in fields:wire[key]=[props[key]['items']['enum'][0]] if props[key]['type']=='array' else props[key]['enum'][0]
        evidence={k:'visible' for k in fields};evidence[fields[0]]=['a','b'];wire['evidence_json']=json.dumps(evidence)
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(wire)}}]}
        _,parsed,_,changes=run.parse_component(response,self.job,fields)
        self.assertEqual(json.loads(parsed[fields[0]]),['a','b']);self.assertEqual(len(changes),1)

    def test_empty_or_mixed_evidence_arrays_fail(self):
        fields=run.chunk.feature_chunks(self.job)[0];props=self.job['response_schema']['properties']['features']['properties']
        for bad in ([],['a',1]):
            wire={k:([props[k]['items']['enum'][0]] if props[k]['type']=='array' else props[k]['enum'][0]) for k in fields}
            evidence={k:'visible' for k in fields};evidence[fields[0]]=bad;wire['evidence_json']=json.dumps(evidence)
            response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(wire)}}]}
            with self.assertRaises(ValueError):run.parse_component(response,self.job,fields)

    def test_fresh_jobs_are_still_unqueried(self):
        self.assertEqual(len(self.source['fresh_jobs']),100)
        self.assertFalse((run.PARENT/'fresh').exists())


if __name__=='__main__':unittest.main()
