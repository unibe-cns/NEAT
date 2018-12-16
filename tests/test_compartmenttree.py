import numpy as np
import matplotlib.pyplot as pl

import pytest
import random
import copy
from neat import SOVTree, SOVNode, Kernel, GreensTree
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

    def testTreeDerivation(self):
        self.loadTTree()
        # locations
        locs_soma         = [(1, 0.5)]
        locs_prox         = [(4, 0.2)]
        locs_bifur        = [(4, 1.0)]
        locs_dist_nobifur = [(6., 0.5), (8., 0.5)]
        locs_dist_bifur   = [(4, 1.0), (6., 0.5), (8., 0.5)]
        locs_dist_nroot   = [(4, 1.0), (4, 0.5), (6., 0.5), (8., 0.5)]
        # test structures
        with pytest.raises(KeyError):
            self.tree.createCompartmentTree('set0')
        # test root (is soma) in set
        self.tree.storeLocs(locs_dist_bifur+locs_soma, 'set0')
        ctree = self.tree.createCompartmentTree('set0')
        assert ctree[0].loc_ind == 3
        assert ctree[1].loc_ind == 0
        cloc_inds = [cn.loc_ind for cn in ctree[1].child_nodes]
        assert 1 in cloc_inds and 2 in cloc_inds
        # test soma not in set (but common root)
        self.tree.storeLocs(locs_dist_bifur, 'set1')
        ctree = self.tree.createCompartmentTree('set1')
        assert ctree[0].loc_ind == 0
        cloc_inds = [cn.loc_ind for cn in ctree[0].child_nodes]
        assert 1 in cloc_inds and 2 in cloc_inds
        # test soma not in set and no common root
        self.tree.storeLocs(locs_dist_nobifur, 'set2')
        with pytest.warns(UserWarning):
            ctree = self.tree.createCompartmentTree('set2')
        assert self.tree.getLocs('set2')[0] == (4, 1.)
        cloc_inds = [cn.loc_ind for cn in ctree[0].child_nodes]
        assert 1 in cloc_inds and 2 in cloc_inds
        # test 2 locs on common root
        self.tree.storeLocs(locs_dist_nroot, 'set3')
        ctree = self.tree.createCompartmentTree('set3')
        assert ctree[0].loc_ind == 1
        assert ctree[1].loc_ind == 0

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
        ctree_prox         = self.tree.createCompartmentTree('prox')
        ctree_bifur        = self.tree.createCompartmentTree('bifur')
        ctree_dist_nobifur = self.tree.createCompartmentTree('dist_nobifur')
        ctree_dist_bifur   = self.tree.createCompartmentTree('dist_bifur')
        # test the tree structures
        assert len(ctree_prox) == len(locs_prox) + 1
        assert len(ctree_bifur) == len(locs_bifur) + 1
        assert len(ctree_dist_nobifur) == len(locs_dist_nobifur) + 1
        assert len(ctree_dist_bifur) == len(locs_dist_bifur) + 1
        # fit the steady state models
        ctree_prox.computeGMC(z_mat_prox)
        ctree_bifur.computeGMC(z_mat_bifur)
        ctree_dist_nobifur.computeGMC(z_mat_dist_nobifur)
        ctree_dist_bifur.computeGMC(z_mat_dist_bifur)
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

    def testReordering(self):
        self.loadTTree()
        # test reordering
        locs_dist_badorder = [(1., 0.5), (8., 0.5), (4, 1.0)]
        self.tree.storeLocs(locs_dist_badorder, 'badorder')
        z_mat_badorder = self.tree.calcImpedanceMatrix(name='badorder')
        ctree_badorder = self.tree.createCompartmentTree('badorder')
        # check if location indices are assigned correctly
        assert [node.loc_ind for node in ctree_badorder] == [0, 2, 1]
        # check if reordering works
        z_mat_reordered = ctree_badorder._preprocessZMatArg(z_mat_badorder)
        assert np.allclose(z_mat_reordered, z_mat_badorder[:,[0,2,1]][[0,2,1],:])
        # check if fitting is correct
        ctree_badorder.computeGMC(z_mat_badorder)
        z_fit_badorder = ctree_badorder.calcImpedanceMatrix()
        assert np.allclose(z_mat_badorder, z_fit_badorder, atol=0.5)
        assert not np.allclose(z_mat_reordered, z_fit_badorder)
        # test if equivalent locs are returned correctly
        locs_equiv = ctree_badorder.getEquivalentLocs()
        assert all([loc == loc_ for loc, loc_ in zip(locs_equiv, [(0, .5), (2, .5), (1, .5)])])

    def loadBallAndStick(self):
        self.greens_tree = GreensTree(file_n='test_morphologies/ball_and_stick.swc')
        for node in self.greens_tree:
            node.setPhysiology(0.8,      # Cm [uF/cm^2]
                               100./1e6, # Ra [MOhm*cm]
                              )
            node.addCurrent('L',  # leak current
                            100., # g_max [uS/cm^2]
                            -75., # e_rev [mV]
                           )
        self.greens_tree.setCompTree()
        # set the impedances
        self.freqs = np.array([0.,1.,10.,100.,1000]) * 1j
        self.greens_tree.setImpedance(self.freqs)

    def testLocationMapping(self, n_loc=20):
        self.loadBallAndStick()
        # define locations
        xvals = np.linspace(0., 1., n_loc+1)[1:]
        locs_1 = [(1, 0.5)] + [(4, x) for x in xvals]
        locs_2 = [(1, 0.5)] + [(4, x) for x in xvals][::-1]
        locs_3 = [(4, x) for x in xvals] + [(1, 0.5)]
        # create compartment trees
        ctree_1 = self.greens_tree.createCompartmentTree(locs_1)
        ctree_2 = self.greens_tree.createCompartmentTree(locs_2)
        ctree_3 = self.greens_tree.createCompartmentTree(locs_3)
        # test location indices
        locinds_1 = np.array([node.loc_ind for node in ctree_1])
        locinds_2 = np.array([node.loc_ind for node in ctree_2])
        locinds_3 = np.array([node.loc_ind for node in ctree_3])
        # check consecutive
        assert np.allclose(locinds_1[:-1], locinds_1[1:] - 1)
        # check permutation
        assert np.allclose(locinds_1[1:], locinds_2[1:][::-1])
        assert np.allclose(locinds_1[:-1], locinds_3[1:])

    def testGCFit(self, n_loc=20):
        self.loadBallAndStick()
        # define locations
        xvals = np.linspace(0., 1., n_loc+1)[1:]
        locs_1 = [(1, 0.5)] + [(4, x) for x in xvals]
        locs_2 = [(1, 0.5)] + [(4, x) for x in xvals][::-1]
        locs_3 = [(4, x) for x in xvals] + [(1, 0.5)]
        locs_4 = random.sample(locs_1, k=len(locs_1))
        # calculate impedance matrices
        z_mat_1 = self.greens_tree.calcImpedanceMatrix(locs_1)
        z_mat_2 = self.greens_tree.calcImpedanceMatrix(locs_2)
        z_mat_3 = self.greens_tree.calcImpedanceMatrix(locs_3)
        z_mat_4 = self.greens_tree.calcImpedanceMatrix(locs_4)
        # create compartment trees
        ctree_1 = self.greens_tree.createCompartmentTree(locs_1)
        ctree_2 = self.greens_tree.createCompartmentTree(locs_2)
        ctree_3 = self.greens_tree.createCompartmentTree(locs_3)
        ctree_4 = self.greens_tree.createCompartmentTree(locs_4)
        # fit g_m and g_c
        ctree_1.computeGMC(z_mat_1[0,:,:], channel_names=['L'])
        ctree_2.computeGMC(z_mat_2[0,:,:], channel_names=['L'])
        ctree_3.computeGMC(z_mat_3[0,:,:], channel_names=['L'])
        ctree_4.computeGMC(z_mat_4[0,:,:], channel_names=['L'])
        # fit c_m
        ctree_1.computeC(self.freqs, z_mat_1)
        ctree_2.computeC(self.freqs, z_mat_2)
        ctree_3.computeC(self.freqs, z_mat_3)
        ctree_4.computeC(self.freqs, z_mat_4)
        # compare both models
        assert str(ctree_1) == str(ctree_2)
        assert str(ctree_1) == str(ctree_3)
        assert str(ctree_1) == str(ctree_4)
        # compare impedance matrices
        z_fit_1 = ctree_1.calcImpedanceMatrix(self.freqs)
        z_fit_2 = ctree_2.calcImpedanceMatrix(self.freqs)
        z_fit_3 = ctree_3.calcImpedanceMatrix(self.freqs)
        z_fit_4 = ctree_4.calcImpedanceMatrix(self.freqs)
        assert np.allclose(z_fit_1, z_mat_1, atol=0.1)
        assert np.allclose(z_fit_2, z_mat_2, atol=0.1)
        assert np.allclose(z_fit_3, z_mat_3, atol=0.1)
        assert np.allclose(z_fit_4, z_mat_4, atol=0.1)
        assert np.allclose(z_fit_1, ctree_2.calcImpedanceMatrix(self.freqs, indexing='tree'))
        assert np.allclose(z_fit_1, ctree_3.calcImpedanceMatrix(self.freqs, indexing='tree'))
        assert np.allclose(z_fit_1, ctree_4.calcImpedanceMatrix(self.freqs, indexing='tree'))


if __name__ == '__main__':
    # tcomp = TestCompartmentTree()
    # tcomp.testTreeDerivation()
    # tcomp.testFitting()
    # tcomp.testReordering()

    tcomp.testLocationMapping()
    tcomp.testGCFit()

