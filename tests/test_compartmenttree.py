import numpy as np
import matplotlib.pyplot as pl
import os

import pytest
import random
import copy

from neat import SOVTree, SOVNode, Kernel, GreensTree, CompartmentTree, CompartmentNode
import neat.tools.kernelextraction as ke
# from neat.channels.channelcollection import channelcollection

import channelcollection_for_tests as channelcollection


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_morphologies'))


class TestCompartmentTree():
    def loadTTree(self):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Tsovtree.swc')
        self.tree = SOVTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(-75., 10.)
        self.tree.setCompTree()
        # do SOV calculation
        self.tree.calcSOVEquations()

    def testStringRepresentation(self):
        # create simple compartment tree
        self.loadTTree()
        locs = [(1, 0.5), (4, 0.5)]
        ctree = self.tree.createCompartmentTree(locs)
        assert str(ctree) == ">>> CompartmentTree\n" \
            "    CompartmentNode 0, Parent: None --- loc_ind = 0, g_c = 0.0 uS, ca = 1.0 uF/cm^2, e_eq = -75.0 mV, (g_L = 0.01 uS/cm^2, e_L = -75.0 mV)\n" \
            "    CompartmentNode 1, Parent: 0 --- loc_ind = 1, g_c = 0.0 uS, ca = 1.0 uF/cm^2, e_eq = -75.0 mV, (g_L = 0.01 uS/cm^2, e_L = -75.0 mV)"

        assert repr(ctree) == "[" \
            "\"{'node index': 0, 'parent index': -1, 'content': '{}', 'loc_ind': 0, 'ca': 1.0, 'g_c': 0.0, 'e_eq': -75.0, 'conc_eqs': {}, 'currents': {'L': [0.01, -75.0]}, 'concmechs': {}, 'expansion_points': {}}\", " \
            "\"{'node index': 1, 'parent index': 0, 'content': '{}', 'loc_ind': 1, 'ca': 1.0, 'g_c': 0.0, 'e_eq': -75.0, 'conc_eqs': {}, 'currents': {'L': [0.01, -75.0]}, 'concmechs': {}, 'expansion_points': {}}\"" \
            "]{'channel_storage': []}"

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
        z_mat_prox         = self.tree.calcImpedanceMatrix(locarg='prox')
        z_mat_bifur        = self.tree.calcImpedanceMatrix(locarg='bifur')
        z_mat_dist_nobifur = self.tree.calcImpedanceMatrix(locarg='dist_nobifur')
        z_mat_dist_bifur   = self.tree.calcImpedanceMatrix(locarg='dist_bifur')
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

    def testReordering(self):
        self.loadTTree()
        # test reordering
        locs_dist_badorder = [(1., 0.5), (8., 0.5), (4, 1.0)]
        self.tree.storeLocs(locs_dist_badorder, 'badorder')
        z_mat_badorder = self.tree.calcImpedanceMatrix(locarg='badorder')
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
        self.greens_tree = GreensTree(file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball_and_stick.swc'))
        self.greens_tree.setPhysiology(0.8, 100./1e6)
        self.greens_tree.setLeakCurrent(100., -75.)
        self.greens_tree.setCompTree()
        # set the impedances
        self.freqs = np.array([0.]) * 1j
        self.greens_tree.setImpedance(self.freqs)
        # create sov tree
        self.sov_tree = self.greens_tree.__copy__(new_tree=SOVTree())
        self.sov_tree.calcSOVEquations(maxspace_freq=50.)

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

    def testGSSFit(self, n_loc=20):
        self.loadBallAndStick()
        # define locations
        xvals = np.linspace(0., 1., n_loc+1)[1:]
        locs_1 = [(1, 0.5)] + [(4, x) for x in xvals]
        locs_2 = [(1, 0.5)] + [(4, x) for x in xvals][::-1]
        locs_3 = [(4, x) for x in xvals] + [(1, 0.5)]
        locs_4 = random.sample(locs_1, k=len(locs_1))
        # calculate impedance matrices
        z_mat_1 = self.greens_tree.calcImpedanceMatrix(locs_1)[0].real
        z_mat_2 = self.greens_tree.calcImpedanceMatrix(locs_2)[0].real
        z_mat_3 = self.greens_tree.calcImpedanceMatrix(locs_3)[0].real
        z_mat_4 = self.greens_tree.calcImpedanceMatrix(locs_4)[0].real
        # create compartment trees
        ctree_1 = self.greens_tree.createCompartmentTree(locs_1)
        ctree_2 = self.greens_tree.createCompartmentTree(locs_2)
        ctree_3 = self.greens_tree.createCompartmentTree(locs_3)
        ctree_4 = self.greens_tree.createCompartmentTree(locs_4)
        # fit g_m and g_c
        ctree_1.computeGMC(z_mat_1, channel_names=['L'])
        ctree_2.computeGMC(z_mat_2, channel_names=['L'])
        ctree_3.computeGMC(z_mat_3, channel_names=['L'])
        ctree_4.computeGMC(z_mat_4, channel_names=['L'])
        # compare impedance matrices
        z_fit_1 = ctree_1.calcImpedanceMatrix(self.freqs)
        z_fit_2 = ctree_2.calcImpedanceMatrix(self.freqs)
        z_fit_3 = ctree_3.calcImpedanceMatrix(self.freqs)
        z_fit_4 = ctree_4.calcImpedanceMatrix(self.freqs)
        assert np.allclose(z_fit_1, z_mat_1, atol=1e-8)
        assert np.allclose(z_fit_2, z_mat_2, atol=1e-8)
        assert np.allclose(z_fit_3, z_mat_3, atol=1e-8)
        assert np.allclose(z_fit_4, z_mat_4, atol=1e-8)
        assert np.allclose(z_fit_1, ctree_2.calcImpedanceMatrix(indexing='tree'))
        assert np.allclose(z_fit_1, ctree_3.calcImpedanceMatrix(indexing='tree'))
        assert np.allclose(z_fit_1, ctree_4.calcImpedanceMatrix(indexing='tree'))

    def testCFit(self, n_loc=20):
        self.loadBallAndStick()
        # define locations
        xvals = np.linspace(0., 1., n_loc+1)[1:]
        locs = [(1, 0.5)] + [(4, x) for x in xvals]
        # create compartment tree
        ctree = self.greens_tree.createCompartmentTree(locs)
        # steady state fit
        z_mat = self.greens_tree.calcImpedanceMatrix(locs)[0].real
        ctree.computeGMC(z_mat)
        # get SOV constants for capacitance fit
        alphas, phimat, importance = self.sov_tree.getImportantModes(locarg=locs,
                                            sort_type='importance', eps=1e-12,
                                            return_importance=True)
        # fit the capacitances from SOV time-scales
        ctree.computeC(-alphas[0:1].real*1e3, phimat[0:1,:].real, weights=importance[0:1])
        # check if equal to membrane time scale
        nds = [self.greens_tree[loc[0]] for loc in locs]
        taus_orig = np.array([n.c_m / n.currents['L'][0] for n in nds])
        taus_fit = np.array([n.ca / n.currents['L'][0] for n in ctree])
        assert np.allclose(taus_orig, taus_fit)

        # fit capacitances with experimental vector fit
        for n in ctree: n.ca = 1.
        self.greens_tree.setImpedance(freqs=ke.create_logspace_freqarray())
        z_mat = self.greens_tree.calcImpedanceMatrix(locs)
        # run the vector fit
        ctree.computeCVF(self.greens_tree.freqs, z_mat)
        taus_fit2 = np.array([n.ca / n.currents['L'][0] for n in ctree])
        assert np.allclose(taus_orig, taus_fit2, atol=.3)

    def testCFitFromZ(self):
        pass

    def fitBallAndStick(self, n_loc=20):
        self.loadBallAndStick()
        # define locations
        xvals = np.linspace(0., 1., n_loc+1)[1:]
        np.random.shuffle(xvals)
        locs = [(1, 0.5)] + [(4, x) for x in xvals]
        # create compartment tree
        ctree = self.greens_tree.createCompartmentTree(locs)
        # steady state fit
        z_mat = self.greens_tree.calcImpedanceMatrix(locs)[0].real
        ctree.computeGMC(z_mat)
        # get SOV constants for capacitance fit
        alphas, phimat, importance = self.sov_tree.getImportantModes(locarg=locs,
                                            sort_type='importance', eps=1e-12,
                                            return_importance=True)
        # fit the capacitances from SOV time-scales
        ctree.computeC(-alphas[0:1].real*1e3, phimat[0:1,:].real, weights=importance[0:1])
        self.ctree = ctree

    def testPasFunctionality(self, n_loc=10):
        self.fitBallAndStick(n_loc=n_loc)

        # test equilibrium potential setting
        e_eq = -75. + np.random.randint(10, size=n_loc+1)
        # with tree indexing
        self.ctree.setEEq(e_eq, indexing='tree')
        assert np.allclose(e_eq, np.array([n.e_eq for n in self.ctree]))
        assert np.allclose(e_eq, self.ctree.getEEq(indexing='tree'))
        assert not np.allclose(e_eq, self.ctree.getEEq(indexing='locs'))
        # with loc indexing
        self.ctree.setEEq(e_eq, indexing='locs')
        assert not np.allclose(e_eq, np.array([n.e_eq for n in self.ctree]))
        assert not np.allclose(e_eq, self.ctree.getEEq(indexing='tree'))
        assert np.allclose(e_eq, self.ctree.getEEq(indexing='locs'))

        # conductance matrices
        gm1 = self.ctree.calcConductanceMatrix(indexing='locs')
        gm2 = self.ctree.calcSystemMatrix(indexing='locs', channel_names=['L'], with_ca=True, use_conc=False)
        gm3 = self.ctree.calcSystemMatrix(indexing='locs', channel_names=['L'], with_ca=False, use_conc=False)
        gm4 = self.ctree.calcSystemMatrix(indexing='locs', channel_names=['L'], with_ca=False, use_conc=True)
        gm5 = self.ctree.calcSystemMatrix(indexing='locs', with_ca=False, use_conc=True)
        gm6 = self.ctree.calcSystemMatrix(indexing='tree', with_ca=False, use_conc=True)
        assert np.allclose(gm1, gm2)
        assert np.allclose(gm1, gm3)
        assert np.allclose(gm1, gm4)
        assert np.allclose(gm1, gm5)
        assert not np.allclose(gm1, gm6)

        # eigenvalues
        alphas, phimat, phimat_inv = self.ctree.calcEigenvalues()
        ca_vec = np.array([1./node.ca for node in self.ctree]) * 1e-3
        assert np.allclose(np.dot(phimat, phimat_inv), np.diag(ca_vec))
        assert np.allclose(np.array([n.ca / n.currents['L'][0] for n in self.ctree]),
                           np.ones(len(self.ctree)) * np.max(1e-3/np.abs(alphas)))

    def loadBall(self):
        self.greens_tree = GreensTree(file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball.swc'))
        # capacitance and axial resistance
        self.greens_tree.setPhysiology(0.8, 100./1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()
        self.greens_tree.addCurrent(k_chan, 0.766*1e6, -85.)
        na_chan = channelcollection.Na_Ta()
        self.greens_tree.addCurrent(na_chan, 1.71*1e6, 50.)
        # fit leak current
        self.greens_tree.fitLeakCurrent(-75., 10.)
        # set computational tree
        self.greens_tree.setCompTree()
        # set the impedances
        self.freqs = np.array([0.])
        self.greens_tree.setImpedance(self.freqs)
        # create sov tree
        self.sov_tree = self.greens_tree.__copy__(new_tree=SOVTree())
        self.sov_tree.calcSOVEquations(maxspace_freq=100.)

    def testChannelFit(self):
        self.loadBall()
        locs = [(1, 0.5)]
        e_eqs = [-75., -55., -35., -15.]
        # create compartment tree
        ctree = self.greens_tree.createCompartmentTree(locs)
        ctree.addCurrent(channelcollection.Na_Ta(), 50.)
        ctree.addCurrent(channelcollection.Kv3_1(), -85.)

        # create tree with only leak
        greens_tree_pas = self.greens_tree.__copy__()
        greens_tree_pas[1].currents = {'L': greens_tree_pas[1].currents['L']}
        greens_tree_pas.setCompTree()
        greens_tree_pas.setImpedance(self.freqs)
        # compute the passive impedance matrix
        z_mat_pas = greens_tree_pas.calcImpedanceMatrix(locs)[0]

        # create tree with only potassium
        greens_tree_k = self.greens_tree.__copy__()
        greens_tree_k[1].currents = {key: val for key, val in greens_tree_k[1].currents.items() \
                                               if key != 'Na_Ta'}
        # compute potassium impedance matrices
        z_mats_k = []
        for e_eq in e_eqs:
            greens_tree_k.setEEq(e_eq)
            greens_tree_k.setCompTree()
            greens_tree_k.setImpedance(self.freqs)
            z_mats_k.append(greens_tree_k.calcImpedanceMatrix(locs))

        # create tree with only sodium
        greens_tree_na = self.greens_tree.__copy__()
        greens_tree_na[1].currents = {key: val for key, val in greens_tree_na[1].currents.items() \
                                               if key != 'Kv3_1'}
        # create state variable expansion points
        svs = []; e_eqs_ = []
        na_chan = greens_tree_na.channel_storage['Na_Ta']
        for e_eq1 in e_eqs:
            sv1 = na_chan.computeVarinf(e_eq1)
            for e_eq2 in e_eqs:
                e_eqs_.append(e_eq2)
                sv2 = na_chan.computeVarinf(e_eq2)
                svs.append({'m': sv2['m'], 'h': sv1['h']})
        # compute sodium impedance matrices
        z_mats_na = []
        for ii, sv in enumerate(svs):
            greens_tree_na.setEEq(e_eqs[ii%len(e_eqs)])
            greens_tree_na[1].setExpansionPoint('Na_Ta', sv)
            greens_tree_na.setCompTree()
            greens_tree_na.setImpedance(self.freqs)
            z_mats_na.append(greens_tree_na.calcImpedanceMatrix(locs))

        # compute combined impedance matrices
        z_mats_comb = []
        for e_eq in e_eqs:
            self.greens_tree.setEEq(e_eq)
            self.greens_tree.setCompTree()
            self.greens_tree.setImpedance(self.freqs)
            z_mats_comb.append(self.greens_tree.calcImpedanceMatrix(locs))

        # passive fit
        ctree.computeGMC(z_mat_pas)
        # get SOV constants for capacitance fit
        sov_tree = greens_tree_pas.__copy__(new_tree=SOVTree())
        sov_tree.setCompTree()
        sov_tree.calcSOVEquations()
        alphas, phimat, importance = sov_tree.getImportantModes(locarg=locs,
                                            sort_type='importance', eps=1e-12,
                                            return_importance=True)
        # fit the capacitances from SOV time-scales
        ctree.computeC(-alphas[0:1].real*1e3, phimat[0:1,:].real, weights=importance[0:1])

        ctree1 = copy.deepcopy(ctree)
        ctree2 = copy.deepcopy(ctree)
        ctree3 = copy.deepcopy(ctree)
        ctree4 = copy.deepcopy(ctree)

        # fit paradigm 1 --> separate impedance matrices and separate fits
        # potassium channel fit
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            ctree1.computeGSingleChanFromImpedance('Kv3_1', z_mat_k, e_eq, self.freqs,
                                                    other_channel_names=['L'])
        ctree1.runFit()
        # sodium channel fit
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree1.computeGSingleChanFromImpedance('Na_Ta', z_mat_na, e_eq, self.freqs,
                                                    sv=sv, other_channel_names=['L'])
        ctree1.runFit()

        # fit paradigm 2 --> separate impedance matrices, same fit
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            ctree2.computeGSingleChanFromImpedance('Kv3_1', z_mat_k, e_eq, self.freqs,
                            all_channel_names=['Kv3_1', 'Na_Ta'])
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree2.computeGSingleChanFromImpedance('Na_Ta', z_mat_na, e_eq, self.freqs, sv=sv,
                            all_channel_names=['Kv3_1', 'Na_Ta'])
        ctree2.runFit()

        # fit paradigm 3 --> same impedance matrices
        for z_mat_comb, e_eq in zip(z_mats_comb, e_eqs):
            ctree3.computeGChanFromImpedance(['Kv3_1', 'Na_Ta'], z_mat_comb, e_eq, self.freqs)
        ctree3.runFit()

        # fit paradigm 4 --> fit incrementally
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree4.computeGSingleChanFromImpedance('Na_Ta', z_mat_na, e_eq, self.freqs, sv=sv)
        ctree4.runFit()
        for z_mat_comb, e_eq in zip(z_mats_comb, e_eqs):
            ctree4.computeGSingleChanFromImpedance('Kv3_1', z_mat_comb, e_eq, self.freqs,
                                other_channel_names=['Na_Ta', 'L'])
        ctree4.runFit()

        # test if correct
        keys = ['L', 'Na_Ta', 'Kv3_1']
        # soma surface (cm) for total conductance calculation
        a_soma = 4. * np.pi * (self.greens_tree[1].R*1e-4)**2
        conds = np.array([self.greens_tree[1].currents[key][0]*a_soma for key in keys])
        # compartment models conductances
        cconds1 = np.array([ctree1[0].currents[key][0] for key in keys])
        cconds2 = np.array([ctree2[0].currents[key][0] for key in keys])
        cconds3 = np.array([ctree3[0].currents[key][0] for key in keys])
        cconds4 = np.array([ctree4[0].currents[key][0] for key in keys])
        assert np.allclose(conds, cconds1)
        assert np.allclose(conds, cconds2)
        assert np.allclose(conds, cconds3)
        assert np.allclose(conds, cconds4)

        # rename for further testing
        ctree = ctree1
        # frequency array
        ft = ke.FourrierTools(np.linspace(0.,50.,100))
        freqs = ft.s
        # compute impedance matrix
        v_h = -42.
        # original
        self.greens_tree.setEEq(v_h)
        self.greens_tree.setCompTree()
        self.greens_tree.setImpedance(freqs)
        z_mat_orig = self.greens_tree.calcImpedanceMatrix([(1.,.5)])
        # potassium
        greens_tree_k.setEEq(v_h)
        greens_tree_k.setCompTree()
        greens_tree_k.setImpedance(freqs)
        z_mat_k = greens_tree_k.calcImpedanceMatrix([(1,.5)])
        # sodium
        greens_tree_na.removeExpansionPoints()
        greens_tree_na.setEEq(v_h)
        greens_tree_na.setCompTree()
        greens_tree_na.setImpedance(freqs)
        z_mat_na = greens_tree_na.calcImpedanceMatrix([(1,.5)])
        # passive
        greens_tree_pas.setCompTree()
        greens_tree_pas.setImpedance(freqs)
        z_mat_pas = greens_tree_pas.calcImpedanceMatrix([(1,.5)])

        # reduced impedance matrices
        ctree.removeExpansionPoints()
        ctree.setEEq(v_h)
        z_mat_fit = ctree.calcImpedanceMatrix(freqs=freqs)
        z_mat_fit_k = ctree.calcImpedanceMatrix(channel_names=['L', 'Kv3_1'], freqs=freqs)
        z_mat_fit_na = ctree.calcImpedanceMatrix(channel_names=['L', 'Na_Ta'], freqs=freqs)
        z_mat_fit_pas = ctree.calcImpedanceMatrix(channel_names=['L'], freqs=freqs)

        assert np.allclose(z_mat_orig, z_mat_fit)
        assert np.allclose(z_mat_k, z_mat_fit_k)
        assert np.allclose(z_mat_na, z_mat_fit_na)
        assert np.allclose(z_mat_pas, z_mat_fit_pas)

        # test total current, conductance
        sv = svs[-1]
        p_open = sv['m']**3 * sv['h']
        # with p_open given
        g1 = ctree[0].getGTot(ctree.channel_storage, channel_names=['L', 'Na_Ta'], p_open_channels={'Na_Ta': p_open})
        i1 = ctree[0].getGTot(ctree.channel_storage, channel_names=['L', 'Na_Ta'], p_open_channels={'Na_Ta': p_open})
        # with expansion point given
        ctree.setExpansionPoints({'Na_Ta': sv})
        g2 = ctree[0].getGTot(ctree.channel_storage, channel_names=['L', 'Na_Ta'])
        i2 = ctree[0].getGTot(ctree.channel_storage, channel_names=['L', 'Na_Ta'])
        # with e_eq given
        g3 = ctree[0].getGTot(ctree.channel_storage, v=e_eqs[-1], channel_names=['L', 'Na_Ta'])
        i3 = ctree[0].getGTot(ctree.channel_storage, v=e_eqs[-1], channel_names=['L', 'Na_Ta'])
        # with e_eq stored
        ctree.setEEq(e_eqs[-1])
        g4 = ctree[0].getGTot(ctree.channel_storage, channel_names=['L', 'Na_Ta'])
        i4 = ctree[0].getGTot(ctree.channel_storage, channel_names=['L', 'Na_Ta'])
        # check if correct
        assert np.abs(g1 - g2) < 1e-10
        assert np.abs(g1 - g3) < 1e-10
        assert np.abs(g1 - g4) < 1e-10
        assert np.abs(i1 - i2) < 1e-10
        assert np.abs(i1 - i3) < 1e-10
        assert np.abs(i1 - i4) < 1e-10
        # compare current, conductance
        g_ = ctree[0].getGTot(ctree.channel_storage, channel_names=['Na_Ta'])
        i_ = ctree[0].getITot(ctree.channel_storage, channel_names=['Na_Ta'])
        assert np.abs(g_ * (e_eqs[-1] - ctree[0].currents['Na_Ta'][1]) - i_) < 1e-10

        # test leak fitting
        self.greens_tree.setEEq(-75.)
        self.greens_tree.setCompTree()
        ctree.setEEq(-75.)
        ctree.removeExpansionPoints()
        ctree.fitEL()
        assert np.abs(ctree[0].currents['L'][1] - self.greens_tree[1].currents['L'][1]) < 1e-10


class TestCompartmentTreePlotting():
    def _initTree1(self):
        # 1   2
        #  \ /
        #   0
        croot = CompartmentNode(0, loc_ind=0)
        cnode1 = CompartmentNode(1, loc_ind=1)
        cnode2 = CompartmentNode(2, loc_ind=2)

        ctree = CompartmentTree(root=croot)
        ctree.addNodeWithParent(cnode1, croot)
        ctree.addNodeWithParent(cnode2, croot)

        self.ctree = ctree

    def _initTree2(self):
        # 3
        # |
        # 2
        # |
        # 1
        # |
        # 0
        croot = CompartmentNode(0, loc_ind=0)
        cnode1 = CompartmentNode(1, loc_ind=1)
        cnode2 = CompartmentNode(2, loc_ind=2)
        cnode3 = CompartmentNode(3, loc_ind=3)

        ctree = CompartmentTree(root=croot)
        ctree.addNodeWithParent(cnode1, croot)
        ctree.addNodeWithParent(cnode2, cnode1)
        ctree.addNodeWithParent(cnode3, cnode2)

        self.ctree = ctree

    def _initTree3(self):
        # 4 5 6 7   8
        #  \|/   \ /
        #   1  2  3
        #    \ | /
        #     \|/
        #      0
        cns = [CompartmentNode(ii, loc_ind=ii) for ii in range(9)]

        ctree = CompartmentTree(root=cns[0])
        # first order children
        ctree.addNodeWithParent(cns[1], cns[0])
        ctree.addNodeWithParent(cns[2], cns[0])
        ctree.addNodeWithParent(cns[3], cns[0])
        # second order children
        ctree.addNodeWithParent(cns[4], cns[1])
        ctree.addNodeWithParent(cns[5], cns[1])
        ctree.addNodeWithParent(cns[6], cns[1])
        ctree.addNodeWithParent(cns[7], cns[3])
        ctree.addNodeWithParent(cns[8], cns[3])

        self.ctree = ctree

    def testPlot(self, pshow=False):
        pl.figure('trees', figsize=(9,4))
        ax1, ax2, ax3 = pl.subplot(131), pl.subplot(132), pl.subplot(133)

        self._initTree1()
        self.ctree.plotDendrogram(ax1, plotargs={'lw': 1, 'c': 'k'})

        self._initTree2()
        self.ctree.plotDendrogram(ax2, plotargs={'lw': 1, 'c': 'DarkGrey'},
                                       labelargs={'marker': 'o', 'ms': 6, 'mfc':'y', 'mec':'r'})

        self._initTree3()
        labelargs = {0: {'marker': 'o', 'ms': 6, 'mfc':'y', 'mec':'r'},
                     3: {'marker': 's', 'ms': 10, 'mfc':'c', 'mec':'g'},
                     5: {'marker': 'v', 'ms': 12, 'mfc':'c', 'mec':'k'}}
        nodelabels = {1: '1', 4: ':-o', 8: ':-)', 9: ':-('}
        textargs = {'fontsize': 10}
        self.ctree.plotDendrogram(ax3, plotargs={'lw': 1, 'c': 'k'},
                                       labelargs=labelargs, nodelabels=nodelabels, textargs=textargs)

        if pshow:
            pl.show()




if __name__ == '__main__':
    tcomp = TestCompartmentTree()
    # tcomp.testStringRepresentation()
    # tcomp.testTreeDerivation()
    # tcomp.testFitting()
    # tcomp.testReordering()
    # tcomp.testLocationMapping()
    tcomp.testGSSFit()
    # tcomp.testCFit()
    # tcomp.testPasFunctionality()
    # tcomp.testChannelFit()

    # tplot = TestCompartmentTreePlotting()
    # tplot.testPlot(pshow=True)


