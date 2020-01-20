import numpy as np
import matplotlib.pyplot as pl

import pytest

from neat import PhysTree, PhysNode


class TestPhysTree():
    def loadTree(self, reinitialize=0):
        '''
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        '''
        if not hasattr(self, 'tree') or reinitialize:
            print '>>> loading T-tree <<<'
            fname = 'test_morphologies/Ttree.swc'
            self.tree = PhysTree(fname, types=[1,3,4])

    def testLeakDistr(self):
        self.loadTree(reinitialize=1)
        with pytest.raises(AssertionError):
            self.tree.fitLeakCurrent(e_eq_target=-75., tau_m_target=-10.)
        # test simple distribution
        self.tree.fitLeakCurrent(e_eq_target=-75., tau_m_target=10.)
        for node in self.tree:
            assert np.abs(node.c_m - 1.0) < 1e-9
            assert np.abs(node.currents['L'][0] - 1. / (10.*1e-3)) < 1e-9
            assert np.abs(node.e_eq + 75.) < 1e-9
        # create complex distribution
        tau_distr = lambda x: x + 100.
        for node in self.tree:
            d2s = self.tree.pathLength({'node': node.index, 'x': 1.}, (1., 0.5))
            node.fitLeakCurrent(e_eq_target=-75., tau_m_target=tau_distr(d2s))
            assert np.abs(node.c_m - 1.0) < 1e-9
            assert np.abs(node.currents['L'][0] - 1. / (tau_distr(d2s)*1e-3)) < \
                   1e-9
            assert np.abs(node.e_eq + 75.) < 1e-9


if __name__ == '__main__':
    tphys = TestPhysTree()
    tphys.testLeakDistr()
