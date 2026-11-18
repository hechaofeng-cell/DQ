"""Counterexamples for risk mixtures and outcome-conditioned matching."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_risk_error_control_20261112 as r


class RiskErrorControlTests(unittest.TestCase):
    def test_pooled_risk_mixture_can_create_a_gap_without_within_Q_gap(self):
        q=np.repeat([1,2],40)
        x=np.zeros((80,1),bool);x[:5]=True;x[40:75]=True;y=~x
        correct=(q==1).astype(float)
        st=r.structure(x,y,q,r.strata_counts(x,y,q)[0])
        g=r.gap_bundle(x,y,q,correct,st)
        self.assertTrue(st['eligible'][0])
        self.assertAlmostEqual(g['pooled'][0],-75)
        self.assertAlmostEqual(g['retained'][0],-75)
        self.assertAlmostEqual(g['standardized'][0],0)

    def test_minimum_counts_do_not_hide_losing_most_state_support(self):
        q=np.array([1]*40+[2]*10)
        x=np.zeros((50,1),bool);x[:20]=True;x[40:]=True;y=~x
        st=r.structure(x,y,q,r.strata_counts(x,y,q)[0])
        self.assertEqual(int((st['a']*st['keep']).sum()),20)
        self.assertFalse(st['eligible'][0])

    def test_exact_error_and_correct_matching_within_each_Q(self):
        q=np.repeat(np.arange(1,6),8)
        correct=np.tile([0,0,1,1,1,1,1,1],5).astype(bool)
        original={m:np.concatenate([np.arange(offset,offset+4)+8*i for i in range(5)])
                  for m,offset in zip(r.METHODS,[0,1,2])}
        chosen,cells=r.match_outcomes(original,q,correct,np.arange(40,dtype=float))
        for m,ix in chosen.items():
            self.assertEqual(len(ix),10)
            self.assertTrue(correct[ix].all())
            self.assertEqual(np.bincount(q[ix],minlength=6)[1:].tolist(),[2]*5)
            self.assertTrue(set(ix)<=set(original[m]))

    def test_already_balanced_sets_keep_all_original_members(self):
        q=np.repeat(np.arange(1,6),8);correct=np.tile([0,1]*4,5).astype(bool)
        selected={m:np.arange(40) for m in r.METHODS}
        matched,_=r.match_outcomes(selected,q,correct,np.arange(40)[::-1])
        for ix in matched.values():
            self.assertEqual(set(ix),set(range(40)))

    def test_zero_support_is_ineligible_and_gap_values_remain_finite(self):
        x=np.zeros((10,2),bool);y=x.copy();q=np.ones(10,int)
        st=r.structure(x,y,q,np.zeros((5,2)))
        result=r.gap_bundle(x,y,q,np.zeros(10),st)
        self.assertFalse(st['eligible'].any())
        for values in result.values():
            self.assertTrue(np.isfinite(values).all())


if __name__=='__main__':
    unittest.main()
