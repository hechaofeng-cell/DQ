from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from qualify_fresh_images_20261112 import choose_qualified,distances,fixed_near


def target(sid,role,reserve_for=""):
    return dict(sample_id=sid,class_name="dog",planned_role=role,reserve_for=reserve_for)


class FreshQualificationTests(unittest.TestCase):
    def test_hamming_distance_matches_scalar_for_full_64_bits(self):
        a=np.array([0,1,2**64-1,0x123456789ABCDEF0],dtype=np.uint64)
        b=np.array([2**64-1,0xFEDCBA9876543210],dtype=np.uint64)
        expected=np.array([[(int(x)^int(y)).bit_count() for y in b] for x in a])
        np.testing.assert_array_equal(distances(a,b),expected)

    def test_frozen_joint_threshold_requires_both_conditions(self):
        self.assertTrue(fixed_near(4,.980))
        self.assertFalse(fixed_near(4,.9799))
        self.assertTrue(fixed_near(8,.995))
        self.assertFalse(fixed_near(9,1.0))
        self.assertFalse(fixed_near(5,.9949))

    def test_primary_has_priority_over_earlier_listed_technical_reserve(self):
        rows=[target("reserve","technical_reserve","selection"),target("primary","reference")]
        kept,excluded,_=choose_qualified(rows,{}, {0:{1},1:{0}})
        self.assertEqual([r["sample_id"] for r in kept],["primary"])
        self.assertEqual(excluded[0]["sample_id"],"reserve")

    def test_reserve_cannot_fill_another_role_or_replace_full_pool(self):
        rows=[target(f"s{i}","selection") for i in range(400)]
        rows += [target("failed_reference","reference"),target("selection_reserve","technical_reserve","selection")]
        kept,_,counts=choose_qualified(rows,{400:["historical_duplicate"]},{})
        self.assertEqual(len(kept),400)
        self.assertEqual(counts["dog","reference"],0)
        self.assertFalse(any(r["sample_id"]=="selection_reserve" for r in kept))


if __name__=="__main__":unittest.main()
