import numpy as np
import matplotlib.pyplot as pl

import pytest
from neat import SOVTree, SOVNode, Kernel
import neat.tools.kernelextraction as ke


class TestCompartmentTree():
    def loadTTree(self):
        '''
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        '''
        print '>>> loading T-tree <<<'
        fname = 'test_morphologies/Tsovtree.swc'
        self.tree = SOVTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(e_eq_target=-75., tau_m_target=10.)
        self.tree.setCompTree()
        # do SOV calculation
        self.tree.calcSOVEquations()

    def testFitting(self):
        self.loadTTree()
        # locations
        locs_soma         = [(1, 0.5)]
        locs_prox         = [(4, 0.2)]
        locs_bifur        = [(4, 1.0)]
        locs_dist_nobifur = [(6., 0.5), (8., 0.5)]
        locs_dist_bifur   = [(4, 1.0), (6., 0.5), (8., 0.5)]
        # store the locations
        self.tree.storeLocs(locs_soma+locs_prox, 'prox')
        self.tree.storeLocs(locs_soma+locs_bifur, 'bifur')
        self.tree.storeLocs(locs_soma+locs_dist_nobifur, 'dist_nobifur')
        self.tree.storeLocs(locs_soma+locs_dist_bifur, 'dist_bifur')
        # derive steady state impedance matrices
        z_mat_prox         = self.tree.calcImpedanceMatrix(name='prox')
        z_mat_bifur        = self.tree.calcImpedanceMatrix(name='bifur')
        z_mat_dist_nobifur = self.tree.calcImpedanceMatrix(name='dist_nobifur')
        z_mat_dist_bifur   = self.tree.calcImpedanceMatrix(name='dist_bifur')
        # create the tree structures
        ctree_prox         = self.tree.createNewTree('prox').createCompartmentTree()
        ctree_bifur        = self.tree.createNewTree('bifur').createCompartmentTree()
        ctree_dist_nobifur = self.tree.createNewTree('dist_nobifur').createCompartmentTree()
        ctree_dist_bifur   = self.tree.createNewTree('dist_bifur').createCompartmentTree()
        # test the tree structures
        assert len(ctree_prox) == len(locs_prox) + 1
        assert len(ctree_bifur) == len(locs_bifur) + 1
        assert len(ctree_dist_nobifur) == len(locs_dist_nobifur) + 1
        assert len(ctree_dist_bifur) == len(locs_dist_bifur) + 1
        # fit the steady state models
        ctree_prox.computeG(z_mat_prox)
        ctree_bifur.computeG(z_mat_bifur)
        ctree_dist_nobifur.computeG(z_mat_dist_nobifur)
        ctree_dist_bifur.computeG(z_mat_dist_bifur)
        # compute the fitted impedance matrices
        z_fit_prox         = ctree_prox.calcImpedanceMatrix()
        z_fit_bifur        = ctree_bifur.calcImpedanceMatrix()
        z_fit_dist_nobifur = ctree_dist_nobifur.calcImpedanceMatrix()
        z_fit_dist_bifur   = ctree_dist_bifur.calcImpedanceMatrix()
        # test correctness
        assert np.allclose(z_fit_prox, z_mat_prox, atol=0.5)
        assert np.allclose(z_fit_bifur, z_mat_bifur, atol=0.5)
        assert not np.allclose(z_fit_dist_nobifur, z_mat_dist_nobifur, atol=0.5)
        assert np.allclose(z_fit_dist_bifur, z_mat_dist_bifur, atol=0.5)
        # TODO: test with capacitances





if __name__ == '__main__':
    tcomp = TestCompartmentTree()
    tcomp.testFitting()
