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

    def load(self, morph_n='ball_and_stick.swc', add_channel=False):
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
        # soma ion channels [uS/cm^2]
        if add_channel:
            # self.greens_tree[1].addCurrent('Na_Ta',  1.71      *1e6,   e_rev=50.,  channel_storage=self.greens_tree.channel_storage)
            # self.greens_tree.addCurrent('Na_Ta',   0.0211   *1e6, e_rev=50.,  node_arg='apical')
            self.greens_tree[1].addCurrent('Ca_LVA', 0.00432   *1e6,   e_rev=50.,  channel_storage=self.greens_tree.channel_storage)
            self.greens_tree.addCurrent('Ca_LVA',  lambda x: 0.0198*1e6   if (x>685. and x<885.) else 0.0198*1e-2*1e6,   e_rev=50.,  node_arg='apical')
        self.greens_tree.setCompTree()
        self.greens_tree.treetype = 'computational'
        # set the impedances
        self.freqs = np.array([0.,1.,10.,100.,1000]) * 1j
        # self.freqs = np.array([0.]) * 1j
        self.greens_tree.setImpedance(self.freqs)

    def testGChanFit(self, n_loc=3, morph_name='ball_and_stick_long.swc'):
        channel_name = 'Ca_LVA'
        self.load(morph_n=morph_name, add_channel=True)
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
        for ii, loc in enumerate(locs):
            for jj, i_amp in enumerate(levels):
                i_in[ii, len(levels)*ii+jj] = i_amp
                # temporary, should be replaced by native neat function
                sim_tree = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
                sim_tree.initModel(t_calibrate=500., factor_lambda=10.)
                sim_tree.addIClamp(loc, i_amp, 0., 500.)
                sim_tree.storeLocs(locs, name='rec locs')
                res = sim_tree.run(500.)
                v_end[:,len(levels)*ii+jj] = res['v_m'][:,-2]

        print v_end
        for node in sim_tree:
            print node.currents


        # do the fit
        self.load(morph_n=morph_name, add_channel=False)
        z_mat_pas = self.greens_tree.calcImpedanceMatrix(locs)
        ctree = self.greens_tree.createCompartmentTree(locs)
        ctree.computeGMC(z_mat_pas[0,:,:], channel_names=['L'])
        ctree.computeC(self.freqs, z_mat_pas, channel_names=['L'])

        ctree_z = copy.deepcopy(ctree)
        ctree_z.addCurrent(channel_name)
        self.load(morph_n=morph_name, add_channel=True)
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

        # add sodium current
        for node in ctree: node.currents['L'][1] = -90.
        ctree.addCurrent(channel_name)
        ctree.computeGChan(v_end, i_in, channel_names=[channel_name], other_channel_names=['L'])


        print '\n>>> currents old method:'
        for node in ctree_z:
            print node.currents

        print '\n>>> currents new method:'
        for node in ctree:
            print node.currents



        self.load(morph_n=morph_name, add_channel=True)
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


        import matplotlib.pyplot as pl
        pl.plot(res['t'], res['v_m'][0], 'b')
        pl.plot(res['t'], res['v_m'][-1], 'r')
        pl.plot(res_['t'], res_['v_m'][0], 'b--', lw=1.6)
        pl.plot(res_['t'], res_['v_m'][-1], 'r--', lw=1.6)
        pl.plot(res__['t'], res__['v_m'][0], 'b-.', lw=1.6)
        pl.plot(res__['t'], res__['v_m'][-1], 'r-.', lw=1.6)
        pl.show()

        # print ctree    # define locations
        # xvals = np.linspace(0., 1., n_loc+1)[1:]
        # locs = [(1, 0.5)] + [(4, x) for x in xvals]
        # print locs
        # self.greens_tree.storeLocs(locs, name='locs')
        # # input current amplitudes
        # levels = [0.005, 0.01, 0.05]
        # loc_inds_list = [np.random.choice(range(n_loc), size=3, replace=False) for _ in range(3*3)]
        # from neatneuron import neuronmodel
        # i_in = np.zeros((len(locs),len(locs)*len(levels)))
        # v_end = np.zeros((len(locs),len(locs)*len(levels)))
        # for ii, loc in enumerate(locs):
        #     for jj, i_amp in enumerate(levels):
        #         i_in[ii, 3*ii+jj] = i_amp
        #         # temporary, should be replaced by native neat function
        #         sim_tree = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
        #         sim_tree.initModel(t_calibrate=500., factor_lambda=10.)
        #         sim_tree.addIClamp(loc, i_amp, 0., 500.)
        #         sim_tree.storeLocs(locs, name='rec locs')
        #         res = sim_tree.run(500.)
        #         print res['v_m'][:,-2:-1]
        #         v_end[:,ii] = res['v_m'][:,-2]
        # # do the fit
        # z_mat_pas = self.greens_tree.calcImpedanceMatrix(locs)
        # ctree = self.greens_tree.createCompartmentTree(locs)
        # ctree.computeGMC(z_mat_pas[0,:,:], channel_names=['L'])
        # # add sodium current
        # ctree.addCurrent(channel_name)
        # ctree.computeGChan(v_end, i_in, channel_names=[channel_name], other_channel_names=['L'])

        # print ctree





if __name__ == '__main__':
    tcomp = TestCompartmentTree()
    # tcomp.testTreeDerivation()
    # tcomp.testFitting()
    # tcomp.testReordering()
    # tcomp.testLocationMapping()
    tcomp.testGChanFit()

