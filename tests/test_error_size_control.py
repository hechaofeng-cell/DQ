import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_error_size_control_20261112 import take_sizes


class SizeControlTests(unittest.TestCase):
    def test_keeps_requested_Q_counts_without_using_member_outcomes(self):
        q=np.repeat(np.arange(1,6),4)
        original={m:np.arange(20) for m in ['R','Greedy20','P20']}
        sizes={tier:2 for tier in range(1,6)}
        chosen=take_sizes(original,q,sizes,np.arange(20)[::-1])
        expected=set(np.concatenate([np.array([2,3])+4*i for i in range(5)]))
        for ix in chosen.values():self.assertEqual(set(ix),expected)


if __name__=='__main__':unittest.main()
