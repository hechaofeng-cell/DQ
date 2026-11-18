"""Checks for paired estimands and semantic-flag boundaries in the VLM pilot."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_label_protocol_pilot_20261112 as analysis


class PilotAnalysisTests(unittest.TestCase):
    def test_unknown_filter_cannot_hide_loss_of_known_support(self):
        records=[]
        for i in range(40):
            records.append(dict(class_name='dog',phase='B',
                old={'A':'clear','B':'clear' if i<20 else 'partial'},
                revised={'A':'clear' if i<20 else 'unknown',
                         'B':'clear' if i<20 else 'unknown'}))
        r=analysis.summarize_field(records,'native','head_visibility')
        self.assertEqual(r['old_known_agreement_pct'],50)
        self.assertEqual(r['revised_known_agreement_pct'],100)
        self.assertEqual(r['both_known_pct_delta_pp'],-50)
        self.assertEqual(r['engineering_screen'],'screen_not_passed')
        self.assertEqual(r['all_four_known_n'],20)

    def test_identical_paired_protocols_have_zero_delta_and_interval(self):
        records=[dict(class_name='cat',phase='D',
                 old={'A':'left','B':'left' if i%2 else 'right'},
                 revised={'A':'left','B':'left' if i%2 else 'right'}) for i in range(40)]
        r=analysis.summarize_field(records,'native','head_orientation')
        self.assertEqual(r['known_agreement_pct_delta_pp'],0)
        self.assertEqual(r['known_agreement_pct_delta_ci_low'],0)
        self.assertEqual(r['known_agreement_pct_delta_ci_high'],0)

    def test_bootstrap_keeps_class_pool_sizes_and_pairing(self):
        records=[dict(class_name=c,phase=p) for c in ['dog','cat'] for p in ['B','D'] for _ in range(3)]
        w=analysis.bootstrap_weights(records,100)
        for start in range(0,12,3):
            np.testing.assert_array_equal(w[:,start:start+3].sum(1),np.full(100,3))

    def test_coexisting_blur_word_does_not_veto_entity_occlusion(self):
        answer=dict(native=dict(occlusion_level='moderate',occluder_type=['person'],
                               truncation_level='none',truncation_parts=['none']),
                    native_evidence=dict(occlusion_level='模糊画面中前景人明确挡住躯干'))
        self.assertEqual(analysis.consistency_flags(answer),[])
        answer['native']['occluder_type']=['image_boundary']
        self.assertIn('image_boundary_used_as_occluder',analysis.consistency_flags(answer))


if __name__=='__main__':unittest.main()
