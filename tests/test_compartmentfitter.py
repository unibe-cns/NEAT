import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
import os

import pytest
import random
import copy
import pickle

from neat import MorphLoc
from neat import PhysTree, GreensTree, GreensTreeTime, SOVTree
from neat import CompartmentFitter
import neat.modelreduction.compartmentfitter as compartmentfitter

import channelcollection_for_tests as channelcollection
import channel_installer
channel_installer.load_or_install_neuron_testchannels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    'test_morphologies'
))


class TestCompartmentFitter():
    def loadTTree(self):
        '''
        Load the T-tree model

          6--5--4--7--8
                |
                |
                1
        '''
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Tsovtree.swc')
        self.tree = PhysTree(fname, types=[1,3,4])
        self.tree.set_physiology(0.8, 100./1e6)
        self.tree.fit_leak_current(-75., 10.)
        self.tree.set_comp_tree()

    def loadBallAndStick(self):
        '''
        Load the ball and stick model

        1--4
        '''
        self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball_and_stick.swc'))
        self.tree.set_physiology(0.8, 100./1e6)
        self.tree.set_leak_current(100., -75.)
        self.tree.set_comp_tree()

    def loadBall(self):
        '''
        Load point neuron model
        '''
        self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball.swc'))
        # capacitance and axial resistance
        self.tree.set_physiology(0.8, 100./1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()
        self.tree.add_channel_current(k_chan, 0.766*1e6, -85.)
        na_chan = channelcollection.Na_Ta()
        self.tree.add_channel_current(na_chan, 1.71*1e6, 50.)
        # fit leak current
        self.tree.fit_leak_current(-75., 10.)
        # set equilibirum potententials
        self.tree.set_v_ep(-75.)
        # set computational tree
        self.tree.set_comp_tree()

    def loadTSegmentTree(self):
        '''
        Load T tree model
        '''
        self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Ttree_segments.swc'))
        # capacitance and axial resistance
        self.tree.set_physiology(0.8, 100./1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()

        g_k = {1: 0.766*1e6}
        g_k.update({n.index: 0.034*1e6 / self.tree.path_length((1,.5), (n.index,.5)) \
                    for n in self.tree if n.index != 1})

        self.tree.add_channel_current(k_chan, g_k, -85.)
        na_chan = channelcollection.Na_Ta()
        self.tree.add_channel_current(na_chan, 1.71*1e6, 50., node_arg=[self.tree[1]])
        # fit leak current
        self.tree.fit_leak_current(-75., 10.)
        # set equilibirum potententials
        self.tree.set_v_ep(-75.)
        # set computational tree
        self.tree.set_comp_tree()

    def testTreeStructure(self):
        self.loadTTree()
        cm = CompartmentFitter(self.tree)
        # set of locations
        fit_locs1 = [(1,.5), (4,.5), (5,.5)] # no bifurcations
        fit_locs2 = [(1,.5), (4,.5), (5,.5), (8,.5)] # w bifurcation, should be added
        fit_locs3 = [(1,.5), (4,1.), (5,.5), (8,.5)] # w bifurcation, already added

        # test fit_locs1, no bifurcation are added
        # input paradigm 1
        cm.setCTree(fit_locs1, extend_w_bifurc=True)

    def _checkChannels(self, tree, channel_names):
        assert isinstance(tree, compartmentfitter.FitTreeGF)
        assert set(tree.channel_storage.keys()) == set(channel_names)
        for node in tree:
            assert set(node.currents.keys()) == set(channel_names + ['L'])

    def testCreateTreeGF(self):
        self.loadBall()
        cm = CompartmentFitter(self.tree)

        # create tree with only 'L'
        tree_pas = cm.createTreeGF()
        self._checkChannels(tree_pas, [])
        # create tree with only 'Na_Ta'
        tree_na = cm.createTreeGF(['Na_Ta'])
        self._checkChannels(tree_na, ['Na_Ta'])
        # create tree with only 'Kv3_1'
        tree_k = cm.createTreeGF(['Kv3_1'])
        self._checkChannels(tree_k, ['Kv3_1'])
        # create tree with all channels
        tree_all = cm.createTreeGF(['Na_Ta', 'Kv3_1'])
        self._checkChannels(tree_all, ['Na_Ta', 'Kv3_1'])

    def reduceExplicit(self):
        self.loadBall()

        freqs = np.array([0.])
        locs = [(1, 0.5)]
        e_eqs = [-75., -55., -35., -15.]
        # create compartment tree
        ctree = self.tree.create_compartment_tree(locs)
        ctree.add_channel_current(channelcollection.Na_Ta(), 50.)
        ctree.add_channel_current(channelcollection.Kv3_1(), -85.)

        # create tree with only leak
        greens_tree_pas = GreensTree(self.tree)
        greens_tree_pas[1].currents = {'L': greens_tree_pas[1].currents['L']}
        greens_tree_pas.set_comp_tree()
        greens_tree_pas.set_impedance(freqs)
        # compute the passive impedance matrix
        z_mat_pas = greens_tree_pas.calc_impedance_matrix(locs)[0]

        # create tree with only potassium
        greens_tree_k = GreensTree(self.tree)
        greens_tree_k[1].currents = {key: val for key, val in greens_tree_k[1].currents.items() \
                                               if key != 'Na_Ta'}
        # compute potassium impedance matrices
        z_mats_k = []
        for e_eq in e_eqs:
            greens_tree_k.set_v_ep(e_eq)
            greens_tree_k.set_comp_tree()
            greens_tree_k.set_impedance(freqs)
            z_mats_k.append(greens_tree_k.calc_impedance_matrix(locs))

        # create tree with only sodium
        greens_tree_na = GreensTree(self.tree)
        greens_tree_na[1].currents = {key: val for key, val in greens_tree_na[1].currents.items() \
                                               if key != 'Kv3_1'}
        # create state variable expansion points
        svs = []; e_eqs_ = []
        na_chan = greens_tree_na.channel_storage['Na_Ta']
        for e_eq1 in e_eqs:
            sv1 = na_chan.computeVarinf(e_eq1)
            for e_eq2 in e_eqs:
                e_eqs_.append(e_eq1)
                sv2 = na_chan.computeVarinf(e_eq2)
                svs.append({'m': sv1['m'], 'h': sv2['h']})

        # compute sodium impedance matrices
        z_mats_na = []
        for sv, eh in zip(svs, e_eqs_):
            greens_tree_na.set_v_ep(eh)
            greens_tree_na[1].set_expansion_point('Na_Ta', sv)
            greens_tree_na.set_comp_tree()
            greens_tree_na.set_impedance(freqs)
            z_mats_na.append(greens_tree_na.calc_impedance_matrix(locs))

        # passive fit
        ctree.compute_gmc(z_mat_pas)

        # potassium channel fit matrices
        fit_mats_k = []
        print(z_mats_k, e_eqs)
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            mf, vt = ctree.compute_g_single_channel(
                'Kv3_1', z_mat_k, e_eq, freqs,
                all_channel_names=['Kv3_1'], other_channel_names=['L'],
                action='return'
            )
            fit_mats_k.append([mf, vt])

        # sodium channel fit matrices
        fit_mats_na = []
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            mf, vt = ctree.compute_g_single_channel(
                'Na_Ta', z_mat_na, e_eq, freqs,
                sv=sv,
                all_channel_names=['Na_Ta'], other_channel_names=['L'],
                action='return'
            )
            fit_mats_na.append([mf, vt])

        return fit_mats_na, fit_mats_k

    def testChannelFitMats(self):
        self.loadBall()
        cm = CompartmentFitter(self.tree, cache_name="channelfitmats", cache_path="neatcache/")
        cm.setCTree([(1,.5)])
        # check if reversals are correct
        for key in set(cm.ctree[0].currents) - {'L'}:
            assert np.abs(cm.ctree[0].currents[key][1] - \
                          self.tree[1].currents[key][1]) < 1e-10

        # fit the passive model
        cm.fitPassive(use_all_channels=False)

        fit_mats_cm_na = cm.evalChannel('Na_Ta')
        fit_mats_cm_k = cm.evalChannel('Kv3_1')
        fit_mats_control_na, fit_mats_control_k = self.reduceExplicit()

        # test whether potassium fit matrices agree
        for fm_cm, fm_control in zip(fit_mats_cm_k, fit_mats_control_k):
            assert np.allclose(fm_cm[1] / fm_control[1], fm_cm[0] / fm_control[0])
        # test whether sodium fit matrices agree
        for fm_cm, fm_control in zip(fit_mats_cm_na[4:], fit_mats_control_na):
            assert np.allclose(fm_cm[1] / fm_control[1], fm_cm[0] / fm_control[0])

    def _checkPasCondProps(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.currents['L'][0], n2.currents['L'][0])
            assert np.allclose(n1.g_c, n2.g_c)

    def _checkPasCaProps(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.ca, n2.ca)

    def _checkAllCurrProps(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        assert ctree1.channel_storage.keys() == ctree2.channel_storage.keys()
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.g_c, n2.g_c)
            for key in n1.currents:
                assert np.allclose(n1.currents[key][0], n2.currents[key][0])
                assert np.allclose(n1.currents[key][1], n2.currents[key][1])

    def _checkPhysTrees(self, tree1, tree2):
        assert len(tree1) == len(tree2)
        assert tree1.channel_storage.keys() == tree2.channel_storage.keys()
        for n1, n2 in zip(tree1, tree2):
            assert np.allclose(n1.r_a, n2.r_a)
            assert np.allclose(n1.c_m, n2.c_m)
            for key in n1.currents:
                assert np.allclose(n1.currents[key][0], n2.currents[key][0])
                assert np.allclose(n1.currents[key][1], n2.currents[key][1])

    def _checkEL(self, ctree, e_l):
        for n in ctree:
            assert np.allclose(n.currents['L'][1], e_l)

    def testPassiveFit(self):
        self.loadTTree()
        fit_locs = [(1,.5), (4,1.), (5,.5), (8,.5)]

        # fit a tree directly from CompartmentTree
        greens_tree = GreensTree(self.tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0].real
        ctree = greens_tree.create_compartment_tree(fit_locs)
        ctree.compute_gmc(z_mat)
        sov_tree = SOVTree(self.tree)
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree.compute_c(-alphas[0:1].real*1e3, phimat[0:1,:].real)

        # fit a tree with compartment fitter
        cm = CompartmentFitter(self.tree, cache_name="passivefit1", cache_path="neatcache/")
        cm.setCTree(fit_locs)
        cm.fitPassive()
        cm.fitCapacitance()
        cm.fitEEq()

        # check whether both trees are the same
        self._checkPasCondProps(ctree, cm.ctree)
        self._checkPasCaProps(ctree, cm.ctree)
        self._checkEL(cm.ctree, -75.)

        # test whether all channels are used correctly for passive fit
        self.loadBall()
        fit_locs = [(1,.5)]
        # fit ball model with only leak
        greens_tree = GreensTree(self.tree)
        greens_tree.channel_storage = {}
        for n in greens_tree:
            n.currents = {'L': n.currents['L']}
        greens_tree.set_comp_tree()
        freqs = np.array([0.])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0].real
        ctree_leak = greens_tree.create_compartment_tree(fit_locs)
        ctree_leak.compute_gmc(z_mat)
        sov_tree = SOVTree(greens_tree)
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree_leak.compute_c(-alphas[0:1].real*1e3, phimat[0:1,:].real)
        # make ball model with leak based on all channels
        tree = PhysTree(self.tree)
        tree.as_passive_membrane()
        tree.set_comp_tree()
        greens_tree = GreensTree(tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0].real
        ctree_all = greens_tree.create_compartment_tree(fit_locs)
        ctree_all.compute_gmc(z_mat)
        sov_tree = SOVTree(tree)
        sov_tree.set_comp_tree()
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree_all.compute_c(-alphas[0:1].real*1e3, phimat[0:1,:].real)

        # new compartment fitter
        cm = CompartmentFitter(self.tree, cache_name="passivefit2", cache_path="neatcache/")
        cm.setCTree(fit_locs)
        # test fitting
        cm.fitPassive(use_all_channels=False)
        cm.fitCapacitance()
        cm.fitEEq()
        self._checkPasCondProps(ctree_leak, cm.ctree)
        self._checkPasCaProps(ctree_leak, cm.ctree)
        with pytest.raises(AssertionError):
            self._checkEL(cm.ctree, self.tree[1].currents['L'][1])
        cm.fitPassive(use_all_channels=True)
        cm.fitCapacitance()
        cm.fitEEq()
        self._checkPasCondProps(ctree_all, cm.ctree)
        self._checkPasCaProps(ctree_all, cm.ctree)
        self._checkEL(cm.ctree, greens_tree[1].currents['L'][1])
        with pytest.raises(AssertionError):
            self._checkEL(cm.ctree, self.tree[1].currents['L'][1])
        with pytest.raises(AssertionError):
            self._checkPasCondProps(ctree_leak, ctree_all)

    def testRecalc_impedance_matrix(self, g_inp=np.linspace(0.,0.01, 20)):
        self.loadBall()
        fit_locs = [(1,.5)]
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        cm.setCTree(fit_locs)

        # test only leak
        # compute impedances explicitly
        greens_tree = cm.createTreeGF(
            channel_names=[],
        )
        greens_tree.set_v_ep(-75.)
        greens_tree.set_impedancesInTree(freqs=0., pprint=True)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs, explicit_method=False)
        z_test = z_mat[:,:,None] / (1. + z_mat[:,:,None] * g_inp[None,None,:])
        # compute impedances with compartmentfitter function
        z_calc = np.array([
            cm.recalc_impedance_matrix('fit locs', [g_i], channel_names=[]) for g_i in g_inp
        ])
        z_calc = np.swapaxes(z_calc, 0, 2)
        assert np.allclose(z_calc, z_test)

        # test with z based on all channels (passive)
        # compute impedances explicitly
        greens_tree = cm.createTreeGF(
            channel_names=list(cm.tree.channel_storage.keys()),
        )
        greens_tree.set_v_ep(-75.)
        greens_tree.set_impedancesInTree(freqs=0., pprint=True)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs, explicit_method=False)
        z_test = z_mat[:,:,None] / (1. + z_mat[:,:,None] * g_inp[None,None,:])
        # compute impedances with compartmentfitter function
        z_calc = np.array([
            cm.recalc_impedance_matrix(
                'fit locs', [g_i], 
                channel_names=list(cm.tree.channel_storage.keys())
            ) for g_i in g_inp
        ])
        z_calc = np.swapaxes(z_calc, 0, 2)
        assert np.allclose(z_calc, z_test)

    def testSynRescale(self, g_inp=np.linspace(0.,0.01, 20)):
        e_rev, v_eq = 0., -75.
        self.loadBallAndStick()
        fit_locs = [(4,.7)]
        syn_locs = [(4,1.)]
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        cm.setCTree(fit_locs)
        # compute impedance matrix
        greens_tree = cm.createTreeGF(
            channel_names=[],
        )
        greens_tree.set_v_ep(-75.)
        greens_tree.set_impedancesInTree(freqs=0.)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs+syn_locs)
        # analytical synapse scale factors
        beta_calc = 1. / (1. + (z_mat[1,1] - z_mat[0,0]) * g_inp)
        beta_full = z_mat[0,1] / z_mat[0,0] * (e_rev - v_eq) / \
                    ((1. + (z_mat[1,1] - z_mat[0,0]) * g_inp ) * (e_rev - v_eq))
        # synapse scale factors from compartment fitter
        beta_cm = np.array([cm.fitSynRescale(fit_locs, syn_locs, [0], [g_i], e_revs=[0.])[0] \
                            for g_i in g_inp])
        assert np.allclose(beta_calc, beta_cm, atol=.020)
        assert np.allclose(beta_full, beta_cm, atol=.015)

    def fitBall(self):
        self.loadBall()
        freqs = np.array([0.])
        locs = [(1, 0.5)]
        e_eqs = [-75., -55., -35., -15.]
        # create compartment tree
        ctree = self.tree.create_compartment_tree(locs)
        ctree.add_channel_current(channelcollection.Na_Ta(), 50.)
        ctree.add_channel_current(channelcollection.Kv3_1(), -85.)

        # create tree with only leak
        greens_tree_pas = GreensTree(self.tree)
        greens_tree_pas[1].currents = {'L': greens_tree_pas[1].currents['L']}
        greens_tree_pas.set_comp_tree()
        greens_tree_pas.set_impedance(freqs)
        # compute the passive impedance matrix
        z_mat_pas = greens_tree_pas.calc_impedance_matrix(locs)[0]

        # create tree with only potassium
        greens_tree_k = GreensTree(self.tree)
        greens_tree_k[1].currents = {key: val for key, val in greens_tree_k[1].currents.items() \
                                               if key != 'Na_Ta'}
        # compute potassium impedance matrices
        z_mats_k = []
        for e_eq in e_eqs:
            greens_tree_k.set_v_ep(e_eq)
            greens_tree_k.set_comp_tree()
            greens_tree_k.set_impedance(freqs)
            z_mats_k.append(greens_tree_k.calc_impedance_matrix(locs))

        # create tree with only sodium
        greens_tree_na = GreensTree(self.tree)
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
            greens_tree_na.set_v_ep(e_eqs[ii%len(e_eqs)])
            greens_tree_na[1].set_expansion_point('Na_Ta', sv)
            greens_tree_na.set_comp_tree()
            greens_tree_na.set_impedance(freqs)
            z_mats_na.append(greens_tree_na.calc_impedance_matrix(locs))

        # passive fit
        ctree.compute_gmc(z_mat_pas)
        # get SOV constants for capacitance fit
        sov_tree = SOVTree(greens_tree_pas)
        sov_tree.set_comp_tree()
        sov_tree.calc_sov_equations()
        alphas, phimat, importance = sov_tree.get_important_modes(loc_arg=locs,
                                            sort_type='importance', eps=1e-12,
                                            return_importance=True)
        # fit the capacitances from SOV time-scales
        ctree.compute_c(-alphas[0:1].real*1e3, phimat[0:1,:].real, weights=importance[0:1])

        # potassium channel fit
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            ctree.compute_g_single_channel('Kv3_1', z_mat_k, e_eq, freqs,
                                                    other_channel_names=['L'])
        ctree.run_fit()
        # sodium channel fit
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree.compute_g_single_channel('Na_Ta', z_mat_na, e_eq, freqs,
                                                    sv=sv, other_channel_names=['L'])
        ctree.run_fit()

        ctree.set_e_eq(-75.)
        ctree.remove_expansion_points()
        ctree.fit_e_leak()

        self.ctree = ctree

    def testFitModel(self):
        self.loadTTree()
        fit_locs = [(1,.5), (4,1.), (5,.5), (8,.5)]

        # fit a tree directly from CompartmentTree
        greens_tree = GreensTree(self.tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0]
        ctree = greens_tree.create_compartment_tree(fit_locs)
        ctree.compute_gmc(z_mat)
        sov_tree = SOVTree(self.tree)
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree.compute_c(-alphas[0:1].real*1e3, phimat[0:1,:].real)

        # fit a tree with compartmentfitter
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        ctree_cm = cm.fitModel(fit_locs)

        # compare the two trees
        self._checkPasCondProps(ctree_cm, ctree)
        self._checkPasCaProps(ctree_cm, ctree)
        self._checkEL(ctree_cm, -75.)

        # check active channel
        self.fitBall()
        locs = [(1, 0.5)]
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        print("(i)")
        ctree_cm_1 = cm.fitModel(locs, use_all_channels_for_passive=False)
        print("(ii)")
        ctree_cm_2 = cm.fitModel(locs, use_all_channels_for_passive=True)
        print("(iii)")

        self._checkAllCurrProps(self.ctree, ctree_cm_1)
        self._checkAllCurrProps(self.ctree, ctree_cm_2)

    def testPickling(self):
        self.loadBall()

        # of PhysTree
        ss = pickle.dumps(self.tree)
        pt_ = pickle.loads(ss)
        self._checkPhysTrees(self.tree, pt_)

        # of GreensTree
        greens_tree = GreensTree(self.tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.])
        greens_tree.set_impedance(freqs)

        ss = pickle.dumps(greens_tree)
        gt_ = pickle.loads(ss)
        self._checkPhysTrees(greens_tree, gt_)

        # of SOVTree
        sov_tree = SOVTree(self.tree)
        sov_tree.calc_sov_equations()

        # works with pickle
        ss = pickle.dumps(sov_tree)
        st_ = pickle.loads(ss)
        self._checkPhysTrees(sov_tree, st_)


    def testParallel(self, w_benchmark=False):
        self.loadTSegmentTree()
        locs = [(nn.index,0.5) for nn in self.tree.nodes[:30]]
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)

        ctree_cm = cm.fitModel(locs, use_all_channels_for_passive=True)

        if w_benchmark:
            from timeit import default_timer as timer
            t0 = timer()
            cm.fitChannels(pprint=False)
            t1 = timer()
            print('Not parallel: %.8f s'%(t1-t0))
            t0 = timer()
            cm.fitChannels(pprint=False)
            t1 = timer()
            print('Parallel: %.8f s'%(t1-t0))

    def testCacheing(self):
        self.loadTSegmentTree()
        locs = [(1, 0.5), (4,.9)]

        cm1 = CompartmentFitter(self.tree,
            cache_name='cacheingtest', cache_path='neatcache/', recompute_cache=True,
        )
        ctree_cm_1a = cm1.fitModel(locs, use_all_channels_for_passive=False)
        ctree_cm_1b = cm1.fitModel(locs, use_all_channels_for_passive=True)

        sv_h = compartmentfitter.get_expansion_points(cm1.cfg.e_hs, channelcollection.Na_Ta())
        tree_na_1 = cm1.createTreeGF(['Na_Ta'])
        tree_na_1.set_impedancesInTree(
            freqs=cm1.cfg.freqs,
            sv_h={'Na_Ta': sv_h}
        )
        del cm1

        cm2 = CompartmentFitter(self.tree,
            cache_name='cacheingtest', cache_path='neatcache/', recompute_cache=False,
        )
        ctree_cm_2a = cm2.fitModel(locs, use_all_channels_for_passive=False)
        ctree_cm_2b = cm2.fitModel(locs, use_all_channels_for_passive=True)

        sv_h = compartmentfitter.get_expansion_points(cm2.cfg.e_hs, channelcollection.Na_Ta())
        tree_na_2 = cm2.createTreeGF(['Na_Ta'])
        tree_na_2.set_impedancesInTree(
            freqs=cm2.cfg.freqs,
            sv_h={'Na_Ta': sv_h}
        )
        del cm2

        self._checkAllCurrProps(ctree_cm_1a, ctree_cm_2a)
        self._checkAllCurrProps(ctree_cm_1b, ctree_cm_2b)

        with pytest.raises(AssertionError):
            self._checkAllCurrProps(ctree_cm_1a, ctree_cm_1b)

        print(tree_na_1.unique_hash())

        assert tree_na_1.unique_hash() == "eca2103ed2fccfada7706b1f0675df2e0d5f926c54c26527f087d419d2088f16"
        assert repr(tree_na_1) == repr(tree_na_2)


def test_expansionpoints():
    kv3_1 = channelcollection.Kv3_1()
    na_ta = channelcollection.Na_Ta()

    e_hs = np.array([-75., -15.])

    # test expansion point for channel with 1 state variable
    sv_hs = compartmentfitter.get_expansion_points(e_hs, kv3_1)
    for svar, f_inf in kv3_1.f_varinf.items():
        assert np.allclose(f_inf(e_hs), sv_hs[svar])
    assert np.allclose(sv_hs['v'], e_hs)

    # test expansion point for channel with 2 state variables
    sv_hs = compartmentfitter.get_expansion_points(e_hs, na_ta)
    v_act = np.array([-75.,-15.,-75.,-75.,-15.,-15.])
    v_inact = np.array([-75.,-15.,-75.,-15.,-75.,-15.])

    assert np.allclose(na_ta.f_varinf['m'](v_act), sv_hs['m'])
    assert np.allclose(na_ta.f_varinf['h'](v_inact), sv_hs['h'])
    assert not np.allclose(na_ta.f_varinf['h'](v_act), sv_hs['h'])
    assert np.allclose(v_act, sv_hs['v'])


if __name__ == '__main__':
    tcf = TestCompartmentFitter()
    # tcf.testTreeStructure()
    # tcf.testCreateTreeGF()
    # tcf.testChannelFitMats()
    tcf.testPassiveFit()
    # tcf.testRecalc_impedance_matrix()
    # tcf.testSynRescale()
    # tcf.testFitModel()
    # tcf.testPickling()
    # tcf.testParallel(w_benchmark=True)
    # tcf.testCacheing()
    # test_expansionpoints()
