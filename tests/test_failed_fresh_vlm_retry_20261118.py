"""Verify that constrained decoding preserves the existing set semantics."""
import importlib.util
from pathlib import Path
import unittest

from jsonschema import Draft7Validator

SPEC = importlib.util.spec_from_file_location('retry_failed', Path(__file__).resolve().parents[1] /
                                            'scripts/retry_failed_fresh_vlm_20261118.py')
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)


class ConstraintsTest(unittest.TestCase):
    def test_encoding_agrees_with_existing_semantic_rule(self):
        import itertools
        values = ['bridle', 'saddle', 'none_observed', 'unknown']
        job = dict(kind='fine', options={}, response_schema=dict(type='object', properties={
            'tack': dict(type='array', items=dict(type='string', enum=values))}),
            semantic_schema=dict(features=[dict(feature_id='tack', value_type='set',
                                                exclusive_values=['none_observed', 'unknown'])]))
        changed = retry.constrained_job(job)
        validator = Draft7Validator(changed['response_schema'])
        for n in range(5):
            for value in itertools.product(values, repeat=n):
                expected = bool(value) and len(value) == len(set(value)) and (
                    len(value) == 1 or not set(value) & {'none_observed', 'unknown'})
                self.assertEqual(validator.is_valid(dict(tack=list(value))), expected, value)
        self.assertNotIn('oneOf', job['response_schema']['properties']['tack'])

    def test_context_change_preserves_image_prompt_and_output_budget(self):
        job = dict(kind='native', options=dict(num_ctx=8192, num_predict=2600),
                   evidence_paths=['unchanged.jpg'], prompt='same')
        changed = retry.constrained_job(job)
        self.assertEqual(changed, {**job, 'options': dict(num_ctx=16384, num_predict=2600)})
        self.assertEqual(job['options']['num_ctx'], 8192)


if __name__ == '__main__':
    unittest.main()
