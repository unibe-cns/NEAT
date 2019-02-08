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
        # set the impedances
        self.freqs = np.array([0.,1.,10.,100.,1000]) * 1j
        # self.freqs = np.array([0.]) * 1j
        self.greens_tree.setImpedance(self.freqs)

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
        ctree.computeC(self.freqs, z_mat_pas, channel_names=['L'])

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

        print '\n>>> currents old method:'
        for node in ctree_z:
            print node.currents

        print '\n>>> currents new method:'
        for node in ctree:
            print node.currents

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


        import matplotlib.pyplot as pl
        pl.figure('v')
        pl.plot(res['t'], res['v_m'][0], 'b')
        pl.plot(res['t'], res['v_m'][-1], 'r')
        pl.plot(res_['t'], res_['v_m'][0], 'b--', lw=1.6)
        pl.plot(res_['t'], res_['v_m'][-1], 'r--', lw=1.6)
        pl.plot(res__['t'], res__['v_m'][0], 'b-.', lw=1.6)
        pl.plot(res__['t'], res__['v_m'][-1], 'r-.', lw=1.6)

        pl.show()


    def simulateModel(self, v0, ca, gl, el, channel_name, gchan, echan, iin, dt=0.025):
        from neat.channels import channelcollection
        channel = eval('channelcollection.' + channel_name + '()')
        v_res, ic_res, il_res, ichan_res = np.zeros_like(iin), np.zeros_like(iin), np.zeros_like(iin), np.zeros_like(iin)
        sv_res = np.zeros((channel.statevars.shape[0], channel.statevars.shape[1], iin.shape[0]))
        t_res = dt * np.arange(len(iin))

        vv, vv_prev = v0, v0
        sv = channel.computeVarInf(v0)
        for kk, tt in enumerate(t_res):
            sv += dt * (channel.computeVarInf(vv) - sv) / channel.computeTauInf(vv)
            # compute currents
            ichan_res[kk] = gchan * channel.computePOpen(vv, statevars=sv) * (echan - vv)
            il_res[kk] = gl * (el - vv)
            # compute next voltage
            vv_aux = vv
            vv += dt / (ca*1e3) * (iin[kk] + ichan_res[kk] + il_res[kk])
            v_res[kk] = vv
            # store capacitive current
            ic_res[kk] = -ca*1e3 * (vv - vv_prev) / (2*dt)
            # ic_res[kk] = -ca*1e3 * (vv - vv_aux) / (dt)
            # update vv_prev
            vv_prev = vv_aux

        return t_res, v_res, ic_res, il_res, ichan_res


    def testGChanFitDynamic(self, n_loc=3, morph_name='ball_and_stick_long.swc'):
        t_max, dt = 200., 0.005
        channel_name = 'Na_Ta'
        self.load(morph_n=morph_name, channel=channel_name)
        # define locations
        # locs = [(1.,0.5)]
        # xvals = np.linspace(0., 1., n_loc+1)[1:]
        # locs = [(1, 0.5)] + [(4, x) for x in xvals]
        # locs = [(1, 0.5)] + [(4, .25), (4,.5), (4,.75)]
        locs = [(1, 0.5)] + [(4,.8)]
        self.greens_tree.storeLocs(locs, name='locs')
        # input current amplitudes
        # levels = [0.05, 0.1]
        levels = [0.1, 0.3]
        # levels = [0.05, 0.1, 0.3]
        # levels = [0.01, 0.05]
        t_start, t_dur = 10., 100.
        # levels = [0.05]
        from neatneuron import neuronmodel
        reslist = []
        for ii, loc in enumerate(locs):
            for jj, i_amp in enumerate(levels):
                # temporary, should be replaced by native neat function
                sim_tree = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
                sim_tree.initModel(dt=dt, t_calibrate=500., factor_lambda=10.)
                print sim_tree
                print 'capacitance:', sim_tree.sections[1](0.5).cm * sim_tree.sections[1](0.5).area()
                sim_tree.addIClamp(loc, i_amp, t_start, t_dur)
                sim_tree.storeLocs(locs, name='rec locs')
                # run the simulation
                res = sim_tree.run(t_max, record_from_iclamps=True, record_from_channels=True, record_v_deriv=True)
                sim_tree.deleteModel()
                # rearrange i_clamp data for fit
                i_mat = np.zeros_like(res['v_m'])
                i_mat[ii,:] = res['i_clamp'][0,:]
                res['i_in'] = i_mat
                # store data for fit
                reslist.append(res)


        # matrices for fit
        v_mat = np.concatenate([res['v_m'] for res in reslist], axis=1)
        dv_mat = np.concatenate([res['dv_dt'] for res in reslist], axis=1)
        i_mat = np.concatenate([-res['i_in'] for res in reslist], axis=1)
        p_o_mat = np.concatenate([res['chan'][channel_name]['p_open'] for res in reslist], axis=1)

        import matplotlib.pyplot as pl
        pl.figure('traces', figsize=(10,5))
        ax = pl.subplot(121)
        ax.set_title('p_open')
        ax.plot(res['t'], res['chan'][channel_name]['p_open'][0], 'b')
        ax.plot(res['t'], res['chan'][channel_name]['m'][0]**2 * res['chan'][channel_name]['h'][0], 'r--')

        ax = pl.subplot(122)
        ax.set_title('v_fit')
        for ii, v_m in enumerate(v_mat):
            ax.plot(np.arange(v_mat.shape[1]), v_m, label=str(locs[ii]))

        # do fit the passive model
        self.load(morph_n=morph_name, channel=None)
        z_mat_pas = self.greens_tree.calcImpedanceMatrix(locs)
        ctree = self.greens_tree.createCompartmentTree(locs)
        ctree.computeGMC(z_mat_pas[0,:,:], channel_names=['L'])
        ctree.computeC(self.freqs, z_mat_pas, channel_names=['L'])

        print '\n>>> passive model:'
        print ctree

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

        ctree_combined = copy.deepcopy(ctree)
        for node in ctree_combined: node.currents['L'][1] = -90.
        ctree_combined.addCurrent(channel_name)
        ctree_combined.computeGMCombined(dv_mat, v_mat, i_mat,
                                 z_mats,
                                 p_open_channels={channel_name: p_o_mat},
                                 e_eqs=es, freqs=self.freqs,
                                 weight_fit1=1., weight_fit2=10.)
        # fit the equilibirum potentials
        ctree_combined.setEEq(v_eq)
        ctree_combined.fitEL()

        # new form fit
        ctree_c2 = copy.deepcopy(ctree)
        for node in ctree_c2: node.currents['L'][1] = -90.
        ctree_c2.addCurrent(channel_name)
        # # add the z_mats
        # for z_mat, e_eq in zip(z_mats, es):
        #     ctree_c2.computeGChanFromImpedance(z_mat, e_eq, self.freqs,
        #                         channel_names=[channel_name], weight=1.)
        # ctree_c2.computeGChanFromTrace(dv_mat, v_mat, i_mat, p_open_channels={channel_name: p_o_mat}, weight=10.)
        ctree_c2.computeGChanFromTraceConv(dt, v_mat, i_mat, p_open_channels={channel_name: p_o_mat}, weight=10.)
        ctree_c2.runFit()

        # do the voltage fit for the channel
        for node in ctree: node.currents['L'][1] = -90.
        ctree.addCurrent(channel_name)
        ctree.computeGChanFromTrace(dv_mat, v_mat, i_mat, p_open_channels={channel_name: p_o_mat}, action='fit')

        print '\n>>> currents original:'
        for node in self.greens_tree:
            print node.currents

        print '\n>>> currents old method:'
        for node in ctree_z:
            print node.currents

        print '\n>>> currents new method:'
        for node in ctree:
            print node.currents

        print '\n>>> currents combined method:'
        for node in ctree_combined:
            print node.currents

        print '\n>>> currents combined method v2:'
        for node in ctree_c2:
            print node.currents

        self.load(morph_n=morph_name, channel=channel_name)
        sim_tree = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
        sim_tree.initModel(dt=dt, t_calibrate=500., factor_lambda=10.)
        sim_tree.addIClamp(locs[-1], levels[-1], t_start, t_dur)
        sim_tree.storeLocs(locs, name='rec locs')
        res = sim_tree.run(t_max)
        sim_tree.deleteModel()

        sim_tree_ = neuronmodel.createReducedModel(ctree)
        locs_ = ctree.getEquivalentLocs()
        sim_tree_.initModel(dt=dt, t_calibrate=500.)
        sim_tree_.addIClamp(locs_[-1], levels[-1], t_start, t_dur)
        sim_tree_.storeLocs(locs_, name='rec locs')
        res_ = sim_tree_.run(t_max)
        sim_tree_.deleteModel()

        sim_tree_ = neuronmodel.createReducedModel(ctree_z)
        locs_ = ctree.getEquivalentLocs()
        sim_tree_.initModel(dt=dt, t_calibrate=500.)
        sim_tree_.addIClamp(locs_[-1], levels[-1], t_start, t_dur)
        sim_tree_.storeLocs(locs_, name='rec locs')
        res_z = sim_tree_.run(t_max)
        sim_tree_.deleteModel()

        sim_tree_ = neuronmodel.createReducedModel(ctree_combined)
        locs_ = ctree.getEquivalentLocs()
        sim_tree_.initModel(dt=dt, t_calibrate=500.)
        sim_tree_.addIClamp(locs_[-1], levels[-1], t_start, t_dur)
        sim_tree_.storeLocs(locs_, name='rec locs')
        res_c = sim_tree_.run(t_max)
        sim_tree_.deleteModel()


        import matplotlib.pyplot as pl
        pl.figure('v')
        pl.plot(res['t'], res['v_m'][0], 'b')
        pl.plot(res['t'], res['v_m'][-1], 'r')
        pl.plot(res_['t'], res_['v_m'][0], 'b--', lw=2, label='trace fit')
        pl.plot(res_['t'], res_['v_m'][-1], 'r--', lw=2)
        pl.plot(res_z['t'], res_z['v_m'][0], 'b-.', lw=2, label='impedance fit')
        pl.plot(res_z['t'], res_z['v_m'][-1], 'r-.', lw=2)
        pl.plot(res_c['t'], res_c['v_m'][0], 'c:', lw=2, label='combined fit')
        pl.plot(res_c['t'], res_c['v_m'][-1], 'y:', lw=2)
        pl.legend(loc=0)

        pl.show()


    def testGChanFitDynamicComp(self, n_loc=1, morph_name='point_neuron.swc'):
        from neat.channels import channelcollection
        t_max, dt = 200., 0.001
        channel_name = 'Na_Ta'
        e_rev = channelcollection.E_REV_DICT[channel_name]
        self.load(morph_n=morph_name)
        # define locations
        # locs = [(1.,0.5)]
        # xvals = np.linspace(0., 1., n_loc+1)[1:]
        # locs = [(1, 0.5)] + [(4, x) for x in xvals]
        # locs = [(1, 0.5)] + [(4, .5)]
        locs = [(1, 0.5)]
        self.greens_tree.storeLocs(locs, name='locs')

        # do fit the passive model
        self.load(morph_n=morph_name, channel=None)
        z_mat_pas = self.greens_tree.calcImpedanceMatrix(locs)
        ctree = self.greens_tree.createCompartmentTree(locs)
        ctree.computeGMC(z_mat_pas[0,:,:], channel_names=['L'])
        ctree.computeC(self.freqs, z_mat_pas, channel_names=['L'])
        for node in ctree: node.currents['L'][1] = -90.

        # add the channel
        for node in ctree:
            g_l = node.currents['L'][0]
            node.addCurrent(channel_name, e_rev=e_rev, channel_storage=ctree.channel_storage)
            node.currents[channel_name][0] = 4.0*np.pi*(self.greens_tree[1].R*1e-4)**2 * 1.71*1e6 # sodium conducance
            # node.currents[channel_name][0] = 4.0*np.pi*(self.greens_tree[1].R*1e-4)**2 * 0.*1e6 # sodium conducance

        print '\n>>> model:'
        print ctree
        print '>>> currents:'
        for node in ctree:
            print node.currents

        # locs_comp = [(0,0.5), (1,0.5)]
        locs_comp = [(0,0.5)]

        v_test, ic_test, il_test, ichan_test, iin_test = [], [], [], [], []

        # input current amplitudes
        # levels = [0.05, 0.1]
        # levels = [0.05, 0.1, 0.3]
        levels = [0.01, 0.05]
        t_start, t_dur = 10., 100.
        # levels = [0.05]
        from neatneuron import neuronmodel
        reslist = []
        for ii, loc in enumerate(locs_comp ):
            for jj, i_amp in enumerate(levels):
                # temporary, should be replaced by native neat function
                sim_tree = neuronmodel.createReducedModel(ctree)
                sim_tree.initModel(dt=dt, t_calibrate=500.)
                print sim_tree
                print 'capacitance:', sim_tree.sections[0](0.5).cm * sim_tree.sections[0](0.5).area()
                sim_tree.addIClamp(loc, i_amp, t_start, t_dur)
                sim_tree.storeLocs(locs_comp, name='rec locs')
                # run the simulation
                res = sim_tree.run(t_max, record_from_iclamps=True, record_from_channels=True, record_v_deriv=True)
                sim_tree.deleteModel()
                # rearrange i_clamp data for fit
                i_mat = np.zeros_like(res['v_m'])
                i_mat[ii,:] = res['i_clamp'][0,:]
                res['i_in'] = i_mat
                # store data for fit
                reslist.append(res)

                # simulate the python model
                v0 = res['v_m'][0,0]
                ca = ctree[0].ca
                gl, el = ctree[0].currents['L']
                gchan, echan = ctree[0].currents[channel_name]
                iin = -i_mat[ii,:]
                t_res, v_res, ic_res, il_res, ichan_res = self.simulateModel(v0, ca, gl, el, channel_name, gchan, echan, iin, dt=dt)
                iin_test.append(iin)
                v_test.append(v_res)
                ic_test.append(ic_res)
                il_test.append(il_res)
                ichan_test.append(ichan_res)

        # as arrays
        iin_test = np.concatenate([i_r for i_r in iin_test])
        v_test = np.concatenate([v_r for v_r in v_test])
        ic_test = np.concatenate([ic_r for ic_r in ic_test])
        il_test = np.concatenate([il_r for il_r in il_test])
        ichan_test = np.concatenate([ichan_r for ichan_r in ichan_test])
        test_res = {'t': np.arange(len(v_test)), 'v': v_test, 'ic': ic_test, 'il': il_test, 'ichan': ichan_test, 'iin': iin_test}


        # matrices for fit
        v_mat = np.concatenate([res['v_m'] for res in reslist], axis=1)
        dv_mat = np.concatenate([res['dv_dt'] for res in reslist], axis=1)
        i_mat = np.concatenate([-res['i_in'] for res in reslist], axis=1)
        p_o_mat = np.concatenate([res['chan'][channel_name]['p_open'] for res in reslist], axis=1)


        import matplotlib.pyplot as pl
        pl.figure('traces', figsize=(10,5))
        ax = pl.subplot(121)
        ax.set_title('p_open')
        ax.plot(res['t'], res['chan'][channel_name]['p_open'][0], 'b')
        ax.plot(res['t'], res['chan'][channel_name]['m'][0]**3 * res['chan'][channel_name]['h'][0], 'r--')

        ax = pl.subplot(122)
        for ii, v_m in enumerate(v_mat):
            ax.plot(np.arange(v_mat.shape[1]), v_m, label=str(locs[ii]))
        ax.plot(np.arange(len(v_test)), v_test, 'r--')
        ax.set_ylim((-100., 50.))


        # # do the impedance matrix fit for the channel
        # ctree_z = copy.deepcopy(ctree)
        # ctree_z.addCurrent(channel_name)
        # self.load(morph_n=morph_name, channel=channel_name)
        # es = np.array([-75., -55., -35., -5.])
        # z_mats = []
        # for e in es:
        #     for node in self.greens_tree: node.setEEq(e)
        #     self.greens_tree.setImpedance(self.freqs)
        #     z_mats.append(self.greens_tree.calcImpedanceMatrix(locs))
        # ctree_z.computeGM(z_mats, e_eqs=es, freqs=self.freqs, channel_names=[channel_name], other_channel_names=['L'])
        # # create a biophysical simulation model
        # sim_tree_biophys = self.greens_tree.__copy__(new_tree=neuronmodel.NeuronSimTree())
        # # compute equilibrium potentials
        # sim_tree_biophys.initModel(t_calibrate=500., factor_lambda=10.)
        # sim_tree_biophys.storeLocs(locs, 'rec locs')
        # res_biophys = sim_tree_biophys.run(10.)
        # sim_tree_biophys.deleteModel()
        # v_eq = res_biophys['v_m'][:,-1]
        # # fit the equilibirum potentials
        # ctree_z.setEEq(v_eq)
        # ctree_z.fitEL()

        # do the voltage fit for the channel
        ctree_ = copy.deepcopy(ctree)
        ctree_.addCurrent(channel_name)
        ctree_.computeGChan(dv_mat, v_mat, i_mat, p_open_channels={channel_name: p_o_mat}, test=test_res)

        print '\n>>> model fit:'
        print ctree_

        # print '\n>>> currents original:'
        # for node in self.greens_tree:
        #     print node.currents

        # print '\n>>> currents old method:'
        # for node in ctree_z:
        #     print node.currents

        # print '\n>>> currents new method:'
        # for node in ctree:
        #     print node.currents

        sim_tree = neuronmodel.createReducedModel(ctree)
        sim_tree.initModel(dt=dt, t_calibrate=500., factor_lambda=10.)
        sim_tree.addIClamp(locs_comp[-1], 0.3, t_start, t_dur)
        sim_tree.storeLocs(locs_comp, name='rec locs')
        res = sim_tree.run(t_max)
        sim_tree.deleteModel()

        sim_tree_ = neuronmodel.createReducedModel(ctree_)
        locs_ = ctree.getEquivalentLocs()
        sim_tree_.initModel(dt=dt, t_calibrate=500.)
        sim_tree_.addIClamp(locs_comp[-1], 0.3, t_start, t_dur)
        sim_tree_.storeLocs(locs_comp, name='rec locs')
        res_ = sim_tree_.run(t_max)
        sim_tree_.deleteModel()



        import matplotlib.pyplot as pl
        pl.figure('v')
        pl.plot(res['t'], res['v_m'][0], 'b')
        pl.plot(res['t'], res['v_m'][-1], 'r')
        pl.plot(res_['t'], res_['v_m'][0], 'b--', lw=1.6)
        pl.plot(res_['t'], res_['v_m'][-1], 'r--', lw=1.6)

        pl.show()






if __name__ == '__main__':
    tcomp = TestCompartmentTree()
    # tcomp.testTreeDerivation()
    # tcomp.testFitting()
    # tcomp.testReordering()
    # tcomp.testLocationMapping()
    # tcomp.testGChanFitSteadyState()

    tcomp.testGChanFitDynamic()
    # tcomp.testGChanFitDynamicComp()

