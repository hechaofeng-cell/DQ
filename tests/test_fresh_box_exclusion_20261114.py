from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qualify_fresh_images_v2_20261114 import partition_official_boxes,valid_official_box,choose_qualified


class BoxExclusionTests(unittest.TestCase):
    def test_frozen_bounds_are_not_clipped_or_relaxed(self):
        self.assertTrue(valid_official_box([0,0,1,1]))
        self.assertFalse(valid_official_box([0,0,1.0000001803,1]))
        self.assertFalse(valid_official_box([0,0,1,1.0000006782]))
        self.assertFalse(valid_official_box([.5,0,.5,1]))

    def test_box_failure_replaced_only_from_its_fixed_role(self):
        bad=dict(source_image_id='bad',class_name='dog',role='reference',rank='751',bbox_norm_xyxy='[0,0,1.01,1]')
        ok=dict(source_image_id='good',class_name='dog',role='selection',rank='1',bbox_norm_xyxy='[0,0,1,1]')
        valid,failed=partition_official_boxes([bad,ok])
        self.assertEqual(valid,[ok]);self.assertEqual(bad['bbox_norm_xyxy'],'[0,0,1.01,1]')
        self.assertEqual(failed[0]['role'],'reference')
        self.assertEqual(failed[0]['reason'],'invalid_official_normalized_box')
        rows=[dict(sample_id='s',class_name='dog',planned_role='selection'),
              dict(sample_id='reserve',class_name='dog',planned_role='technical_reserve',reserve_for='reference')]
        accepted,_,counts=choose_qualified(rows,{}, {})
        self.assertEqual(counts['dog','reference'],1)
        self.assertEqual(accepted[1]['role'],'reference')

    def test_actual_frozen_plan_has_two_accounted_box_failures(self):
        from prepare_fresh_images_20261112 import READY,read
        plan=read(READY/'alternative_provisional_plan.csv')
        valid,failed=partition_official_boxes(plan)
        self.assertEqual(len(valid)+len(failed),5400)
        self.assertEqual({r['sample_id'] for r in failed},{'fresh-obj365-train-1064760-dog','fresh-obj365-train-1300935-person'})
        self.assertTrue(all(r['role']=='reference' for r in failed))


if __name__=='__main__':unittest.main()
