import numpy as np
import matplotlib.pyplot as pl

import pytest
import random
import copy
from neat import SOVTree, SOVNode, Kernel, GreensTree, CompartmentTree, CompartmentNode
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
        print('>>> loading T-tree <<<')
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
        self.greens_tree = GreensTree(file_n='test_morphologies/ball_and_stick.swc')
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
        # compare both models
        assert str(ctree_1) == str(ctree_2)
        assert str(ctree_1) == str(ctree_3)
        assert str(ctree_1) == str(ctree_4)
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

    def load(self, morph_n='ball_and_stick.swc', channel=None):
        self.greens_tree = GreensTree(file_n='test_morphologies/' + morph_n)
        for node in self.greens_tree:
            node.setPhysiology(1.,      # Cm [uF/cm^2]
                               100./1e6, # Ra [MOhm*cm]
                              )
            node.addCurrent('L',  # leak current
                            100., # g_max [uS/cm^2]
                            -75., # e_rev [mV]
                           )
        self.greens_tree[1].addCurrent('L',      0.0000344 *1e6,   e_rev=-90., channel_storage=self.greens_tree.channel_storage)
        self.greens_tree.addCurrent('L',       0.0000447*1e6, e_rev=-90., node_arg='apical')
        self.greens_tree_pas = copy.deepcopy(self.greens_tree)
        self.sov_tree = self.greens_tree.__copy__(new_tree=SOVTree())
        # soma ion channels [uS/cm^2]
        if channel == 'Na_Ta':
            self.greens_tree[1].addCurrent('Na_Ta',  1.71      *1e6,   e_rev=50.,  channel_storage=self.greens_tree.channel_storage)
            self.greens_tree.addCurrent('Na_Ta',   0.0211   *1e6, e_rev=50.,  node_arg='apical')
            # self.greens_tree[1].addCurrent('Na_Ta',  0.      *1e6,   e_rev=50.,  channel_storage=self.greens_tree.channel_storage)
            # self.greens_tree.addCurrent('Na_Ta',   0.0   *1e6, e_rev=50.,  node_arg='apical')
        elif channel == 'Ca_LVA':
            self.greens_tree[1].addCurrent('Ca_LVA', 0.00432   *1e6,   e_rev=50.,  channel_storage=self.greens_tree.channel_storage)
            self.greens_tree.addCurrent('Ca_LVA',  lambda x: 0.0198*1e6   if (x>685. and x<885.) else 0.0198*1e-2*1e6,   e_rev=50.,  node_arg='apical')
        elif channel == 'TestChannel2':
            self.greens_tree[1].addCurrent('TestChannel2', 0.01*1e6,   e_rev=-23.,  channel_storage=self.greens_tree.channel_storage)
            self.greens_tree.addCurrent('TestChannel2',  0.001*1e6,   e_rev=-23.,  node_arg='apical')
        self.greens_tree.setCompTree()
        self.greens_tree.treetype = 'computational'
        self.greens_tree_pas.setCompTree()
        self.sov_tree.setCompTree()
        # set the impedances
        self.freqs = np.array([0.,1.,10.,100.,1000]) * 1j
        # self.freqs = np.array([0.]) * 1j
        self.greens_tree.setImpedance(self.freqs)
        # compute SOV factorisation
        self.sov_tree.calcSOVEquations(pprint=True)

    def loadSOV(self):
        self.sov_tree = self.greens_tree_pas.__copy__(new_tree=SOVTree())
        # set the computational tree
        sov_tree.setCompTree(eps=1.)
        # compute SOV factorisation
        sov_tree.calcSOVEquations(pprint=True)

    def testGChanFitSteadyState(self, n_loc=3, morph_name='ball_and_stick_long.swc'):
        t_max, dt = 500., 0.1
        channel_name = 'Ca_LVA'
        self.load(morph_n=morph_name, channel=channel_name)
        # define locations
        # locs = [(1.,0.5)]
        xvals = np.linspace(0., 1., n_loc+1)[1:]
        locs = [(1, 0.5)] + [(4, x) for x in xvals]
        self.greens_tree.storeLocs(locs, name='locs')
        # input current amplitudes
        # levels = [0.05, 0.1]
        levels = [0.05, 0.1, 0.3]
        # levels = [0.05]
        from neatneuron import neuronmodel
        i_in = np.zeros((len(locs),len(locs)*len(levels)))
        v_end = np.zeros((len(locs),len(locs)*len(levels)))
        p_open = np.zeros((len(locs),len(locs)*len(levels)))
        for ii, loc in enumerate(locs):
            for jj, i_amp in enumerate(levels):
                i_in[ii, len(levels)*ii+jj] = i_amp
                # temporary, should be replaced by native neat function
                sim_tree = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
                sim_tree.initModel(t_calibrate=500., factor_lambda=10.)
                sim_tree.addIClamp(loc, i_amp, 0., 500.)
                sim_tree.storeLocs(locs, name='rec locs')
                # run the simulation
                res = sim_tree.run(500., record_from_channels=True, record_from_iclamps=True)
                sim_tree.deleteModel()
                # store data for fit
                v_end[:,len(levels)*ii+jj] = res['v_m'][:,-2]
                p_open[:,len(levels)*ii+jj] = res['chan'][channel_name]['p_open'][:,-2]

        # print p_open

        # print 'i_clamp amplitude =', res['i_clamp'][:,-1]

        # derivative is zero for steady state
        dv_end = np.zeros_like(v_end)

        # do fit the passive model
        self.load(morph_n=morph_name, channel=channel_name)
        z_mat_pas = self.greens_tree.calcImpedanceMatrix(locs)
        ctree = self.greens_tree.createCompartmentTree(locs)
        ctree.computeGMC(z_mat_pas[0,:,:], channel_names=['L'])
        # ctree.computeC(self.freqs, z_mat_pas, channel_names=['L'])

        # compute c
        alphas, phimat = self.sov_tree.getImportantModes(locarg=locs, sort_type='importance', eps=1e-9)
        n_mode = len(locs)
        alphas = alphas[:n_mode]
        phimat = phimat[:n_mode, :]
        importance = self.sovtree.getModeImportance(sov_data=(alphas, phimat), importance_type='simple')
        ctree.computeC(-alphas*1e3, phimat, weight=importance)
        # ctree.computeC(self.freqs, z_mat_pas, channel_names=['L'])

        # do the impedance matrix fit for the channel
        ctree_z = copy.deepcopy(ctree)
        ctree_z.addCurrent(channel_name)
        self.load(morph_n=morph_name, channel=channel_name)
        es = np.array([-75., -55., -35., -5.])
        z_mats = []
        for e in es:
            for node in self.greens_tree: node.setEEq(e)
            self.greens_tree.setImpedance(self.freqs)
            z_mats.append(self.greens_tree.calcImpedanceMatrix(locs))
        ctree_z.computeGM(z_mats, e_eqs=es, freqs=self.freqs, channel_names=[channel_name], other_channel_names=['L'])
        # create a biophysical simulation model
        sim_tree_biophys = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
        # compute equilibrium potentials
        sim_tree_biophys.initModel(t_calibrate=500., factor_lambda=10.)
        sim_tree_biophys.storeLocs(locs, 'rec locs')
        res_biophys = sim_tree_biophys.run(10.)
        sim_tree_biophys.deleteModel()
        v_eq = res_biophys['v_m'][:,-1]
        # fit the equilibirum potentials
        ctree_z.setEEq(v_eq)
        ctree_z.fitEL()

        # do the voltage fit for the channel
        for node in ctree: node.currents['L'][1] = -90.
        ctree.addCurrent(channel_name)
        ctree.computeGChan(dv_end, v_end, i_in, p_open_channels={channel_name: p_open})

        print('\n>>> currents old method:')
        for node in ctree_z:
            print(node.currents)

        print('\n>>> currents new method:')
        for node in ctree:
            print(node.currents)

        self.load(morph_n=morph_name, channel=channel_name)
        sim_tree = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
        sim_tree.initModel(t_calibrate=500., factor_lambda=10.)
        sim_tree.addIClamp(locs[-1], 0.1, 0., 500.)
        sim_tree.storeLocs(locs, name='rec locs')
        res = sim_tree.run(500.)
        sim_tree.deleteModel()

        sim_tree_ = neuronmodel.createReducedModel(ctree)
        locs_ = ctree.getEquivalentLocs()
        sim_tree_.initModel(t_calibrate=500.)
        sim_tree_.addIClamp(locs_[-1], 0.1, 0., 500.)
        sim_tree_.storeLocs(locs_, name='rec locs')
        res_ = sim_tree_.run(500.)
        sim_tree_.deleteModel()

        sim_tree_ = neuronmodel.createReducedModel(ctree_z)
        locs_ = ctree.getEquivalentLocs()
        sim_tree_.initModel(t_calibrate=500.)
        sim_tree_.addIClamp(locs_[-1], 0.1, 0., 500.)
        sim_tree_.storeLocs(locs_, name='rec locs')
        res__ = sim_tree_.run(500.)
        sim_tree_.deleteModel()

        pl.figure('v')
        pl.plot(res['t'], res['v_m'][0], 'b')
        pl.plot(res['t'], res['v_m'][-1], 'r')
        pl.plot(res_['t'], res_['v_m'][0], 'b--', lw=1.6)
        pl.plot(res_['t'], res_['v_m'][-1], 'r--', lw=1.6)
        pl.plot(res__['t'], res__['v_m'][0], 'b-.', lw=1.6)
        pl.plot(res__['t'], res__['v_m'][-1], 'r-.', lw=1.6)

        pl.show()


class TestCompartmentTreePlotting():
    def _initTree1(self):
        '''
        1   2
         \ /
          0
        '''
        croot = CompartmentNode(0, loc_ind=0)
        cnode1 = CompartmentNode(1, loc_ind=1)
        cnode2 = CompartmentNode(2, loc_ind=2)

        ctree = CompartmentTree(root=croot)
        ctree.addNodeWithParent(cnode1, croot)
        ctree.addNodeWithParent(cnode2, croot)

        self.ctree = ctree

    def _initTree2(self):
        '''
        3
        |
        2
        |
        1
        |
        0
        '''
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
        '''
        4 5 6 7   8
         \|/   \ /
          1  2  3
           \ | /
            \|/
             0
        '''
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
    # tcomp.testTreeDericvation()
    # tcomp.testFitting()
    # tcomp.testReordering()
    # tcomp.testLocationMapping()
    # tcomp.testGSSFit()
    tcomp.testCFit()
    # tcomp.testPasFunctionality()
    # tcomp.testGChanFitSteadyState()

    # tcomp.testGChanFitDynamic()
    # tcomp.testGChanFitDynamicComp()

    # tplot = TestCompartmentTreePlotting()
    # tplot.testPlot(pshow=True)


