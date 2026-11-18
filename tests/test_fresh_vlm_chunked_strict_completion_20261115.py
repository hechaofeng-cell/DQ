from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_fresh_vlm_chunked_strict_completion_20261115 as run


class ChunkedStrictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=json.loads((run.PARENT/'inputs.json').read_text());cls.job=cls.inputs['pilot_jobs'][0]

    def test_native_chunks_are_disjoint_complete_and_bounded(self):
        chunks=run.feature_chunks(self.job);expected=list(self.job['response_schema']['properties']['features']['properties'])
        self.assertEqual(chunks[0]+chunks[1],expected);self.assertFalse(set(chunks[0])&set(chunks[1]))
        self.assertLessEqual(max(map(len,chunks)),10)

    def test_component_schema_is_flat_and_at_most_eleven_properties(self):
        for fields in run.feature_chunks(self.job):
            schema=run.component_schema(self.job,fields)
            self.assertLessEqual(len(schema['properties']),11)
            self.assertTrue(all(v.get('type')!='object' for v in schema['properties'].values()))

    def test_component_parser_preserves_feature_and_evidence_values(self):
        fields=run.feature_chunks(self.job)[0];props=self.job['response_schema']['properties']['features']['properties']
        value={}
        for key in fields:
            value[key]=[props[key]['items']['enum'][0]] if props[key]['type']=='array' else props[key]['enum'][0]
        evidence={k:'visible' for k in fields};value['evidence_json']=json.dumps(evidence,separators=(',',':'))
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}
        features,parsed_evidence,wire=run.parse_component(response,self.job,fields)
        self.assertEqual(features,{k:value[k] for k in fields});self.assertEqual(parsed_evidence,evidence);self.assertEqual(wire,value)

    def test_parent_fresh_set_remains_unexposed(self):
        summary=json.loads((run.PARENT/'summary.json').read_text())
        self.assertFalse(summary['fresh_started']);self.assertEqual(len(self.inputs['fresh_targets']),60)
        self.assertEqual(len(self.inputs['fresh_jobs']),100)


if __name__=='__main__':unittest.main()
