import numpy as np
import matplotlib.pyplot as pl

import pytest
import random
import copy

from neat import MorphLoc
from neat import PhysTree, GreensTree, SOVTree
from neat import CompartmentFitter
from neat.channels import channelcollection
import neat.tools.fittools.compartmentfitter as compartmentfitter


class TestCompartmentFitter():
    def loadTTree(self):
        '''
        Load the T-tree model

          6--5--4--7--8
                |
                |
                1
        '''
        print('>>> loading T-tree <<<')
        fname = 'test_morphologies/Tsovtree.swc'
        self.tree = PhysTree(fname, types=[1,3,4])
        self.tree.setPhysiology(0.8, 100./1e6)
        self.tree.fitLeakCurrent(-75., 10.)
        self.tree.setCompTree()

    def loadBallAndStick(self):
        '''
        Load the ball and stick model

        1--4
        '''
        self.tree = PhysTree(file_n='test_morphologies/ball_and_stick.swc')
        self.tree.setPhysiology(0.8, 100./1e6)
        self.tree.setLeakCurrent(100., -75.)
        self.tree.setCompTree()

    def loadBall(self):
        '''
        Load point neuron model
        '''
        self.tree = PhysTree(file_n='test_morphologies/ball.swc')
        # capacitance and axial resistance
        self.tree.setPhysiology(0.8, 100./1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()
        self.tree.addCurrent(k_chan, 0.766*1e6, -85.)
        na_chan = channelcollection.Na_Ta()
        self.tree.addCurrent(na_chan, 1.71*1e6, 50.)
        # fit leak current
        self.tree.fitLeakCurrent(-75., 10.)
        # set equilibirum potententials
        self.tree.setEEq(-75.)
        # set computational tree
        self.tree.setCompTree()

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
        fl1_a = cm.tree.getLocs('fit locs')
        cm.setCTree(fit_locs1, extend_w_bifurc=False)
        fl1_b = cm.tree.getLocs('fit locs')
        assert len(fl1_a) == len(fl1_b)
        for fla, flb in zip(fl1_a, fl1_b): assert fla == flb
        # input paradigm 2
        cm.tree.storeLocs(fit_locs1, 'fl1')
        cm.setCTree('fl1', extend_w_bifurc=True)
        fl1_a = cm.tree.getLocs('fit locs')
        assert len(fl1_a) == len(fl1_b)
        for fla, flb in zip(fl1_a, fl1_b): assert fla == flb
        # test tree structure
        assert len(cm.ctree) == 3
        for cn in cm.ctree: assert len(cn.child_nodes) <= 1

        # test fit_locs2, a bifurcation should be added
        with pytest.warns(UserWarning):
            cm.setCTree(fit_locs2, extend_w_bifurc=False)
        fl2_b = cm.tree.getLocs('fit locs')
        cm.setCTree(fit_locs2, extend_w_bifurc=True)
        fl2_a = cm.tree.getLocs('fit locs')
        assert len(fl2_a) == len(fl2_b) + 1
        for fla, flb in zip(fl2_a, fl2_b): assert fla == flb
        assert fl2_a[-1] == (4,1.)
        # test tree structure
        assert len(cm.ctree) == 5
        for cn in cm.ctree:
            assert len(cn.child_nodes) <= 1 if cn.loc_ind != 4 else \
                   len(cn.child_nodes) == 2

        # test fit_locs2, no bifurcation should be added as it is already present
        cm.setCTree(fit_locs3, extend_w_bifurc=True)
        fl3 = cm.tree.getLocs('fit locs')
        for fl_, fl3 in zip(fit_locs3, fl3): assert fl_ == fl3
        # test tree structure
        assert len(cm.ctree) == 4
        for cn in cm.ctree:
            assert len(cn.child_nodes) <= 1 if cn.loc_ind != 1 else \
                   len(cn.child_nodes) == 2

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
        ctree = self.tree.createCompartmentTree(locs)
        ctree.addCurrent(channelcollection.Na_Ta(), 50.)
        ctree.addCurrent(channelcollection.Kv3_1(), -85.)

        # create tree with only leak
        greens_tree_pas = self.tree.__copy__(new_tree=GreensTree())
        greens_tree_pas[1].currents = {'L': greens_tree_pas[1].currents['L']}
        greens_tree_pas.setCompTree()
        greens_tree_pas.setImpedance(freqs)
        # compute the passive impedance matrix
        z_mat_pas = greens_tree_pas.calcImpedanceMatrix(locs)[0]

        # create tree with only potassium
        greens_tree_k = self.tree.__copy__(new_tree=GreensTree())
        greens_tree_k[1].currents = {key: val for key, val in greens_tree_k[1].currents.items() \
                                               if key != 'Na_Ta'}
        # compute potassium impedance matrices
        z_mats_k = []
        for e_eq in e_eqs:
            greens_tree_k.setEEq(e_eq)
            greens_tree_k.setCompTree()
            greens_tree_k.setImpedance(freqs)
            z_mats_k.append(greens_tree_k.calcImpedanceMatrix(locs))

        # create tree with only sodium
        greens_tree_na = self.tree.__copy__(new_tree=GreensTree())
        greens_tree_na[1].currents = {key: val for key, val in greens_tree_na[1].currents.items() \
                                               if key != 'Kv3_1'}
        # create state variable expansion points
        svs = []; e_eqs_ = []
        na_chan = greens_tree_na.channel_storage['Na_Ta']
        for e_eq1 in e_eqs:
            sv1 = na_chan.computeVarInf(e_eq1)
            for e_eq2 in e_eqs:
                e_eqs_.append(e_eq1)
                sv2 = na_chan.computeVarInf(e_eq2)
                svs.append(np.array([[sv1[0,0], sv2[0,1]]]))

        # compute sodium impedance matrices
        z_mats_na = []
        for sv, eh in zip(svs, e_eqs_):
            greens_tree_na.setEEq(eh)
            greens_tree_na[1].setExpansionPoint('Na_Ta', sv)
            greens_tree_na.setCompTree()
            greens_tree_na.setImpedance(freqs)
            z_mats_na.append(greens_tree_na.calcImpedanceMatrix(locs))

        # passive fit
        ctree.computeGMC(z_mat_pas)

        # potassium channel fit matrices
        fit_mats_k = []
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            mf, vt = ctree.computeGSingleChanFromImpedance(
                            'Kv3_1', z_mat_k, e_eq, freqs,
                            other_channel_names=['L'], action='return'
                            )
            fit_mats_k.append([mf, vt])

        # sodium channel fit matrices
        fit_mats_na = []
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            mf, vt = ctree.computeGSingleChanFromImpedance(
                            'Na_Ta', z_mat_na, e_eq, freqs,
                            sv=sv, other_channel_names=['L'], action='return'
                            )
            fit_mats_na.append([mf, vt])

        return fit_mats_na, fit_mats_k

    def testChannelFitMats(self):
        self.loadBall()
        cm = CompartmentFitter(self.tree)
        cm.setCTree([(1,.5)])
        # check if reversals are correct
        for key in set(cm.ctree[0].currents) - {'L'}:
            assert np.abs(cm.ctree[0].currents[key][1] - \
                          self.tree[1].currents[key][1]) < 1e-10

        # fit the passive model
        cm.fitPassive(use_all_channels=False)

        fit_mats_cm_na = cm.evalChannel('Na_Ta', parallel=False)
        fit_mats_cm_k = cm.evalChannel('Kv3_1', parallel=False)
        fit_mats_control_na, fit_mats_control_k = self.reduceExplicit()
        # test whether potassium fit matrices agree
        for fm_cm, fm_control in zip(fit_mats_cm_k, fit_mats_control_k):
            assert np.allclose(np.sum(fm_cm[0]), fm_control[0][0,0]) # feature matrices
            assert np.allclose(fm_cm[1], fm_control[1]) # target vectors
        # test whether sodium fit matrices agree
        for fm_cm, fm_control in zip(fit_mats_cm_na[4:], fit_mats_control_na):
            assert np.allclose(np.sum(fm_cm[0]), fm_control[0][0,0]) # feature matrices
            assert np.allclose(fm_cm[1], fm_control[1]) # target vectors

    def _checkPasCondProps(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.currents['L'][0], n2.currents['L'][0])
            assert np.allclose(n1.g_c, n2.g_c)

    def _checkPasCaProps(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.ca, n2.ca)

    def _checkEL(self, ctree, e_l):
        for n in ctree:
            assert np.allclose(n.currents['L'][1], e_l)

    def testPassiveFit(self):
        self.loadTTree()
        fit_locs = [(1,.5), (4,1.), (5,.5), (8,.5)]

        # fit a tree directly from CompartmentTree
        greens_tree = self.tree.__copy__(new_tree=GreensTree())
        greens_tree.setCompTree()
        freqs = np.array([0.])
        greens_tree.setImpedance(freqs)
        z_mat = greens_tree.calcImpedanceMatrix(fit_locs)[0].real
        ctree = greens_tree.createCompartmentTree(fit_locs)
        ctree.computeGMC(z_mat)
        sov_tree = self.tree.__copy__(new_tree=SOVTree())
        sov_tree.calcSOVEquations()
        alphas, phimat = sov_tree.getImportantModes(locarg=fit_locs)
        ctree.computeC(-alphas[0:1].real*1e3, phimat[0:1,:].real)

        # fit a tree with compartment fitter
        cm = CompartmentFitter(self.tree)
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
        greens_tree = self.tree.__copy__(new_tree=GreensTree())
        greens_tree.channel_storage = {}
        for n in greens_tree:
            n.currents = {'L': n.currents['L']}
        greens_tree.setCompTree()
        freqs = np.array([0.])
        greens_tree.setImpedance(freqs)
        z_mat = greens_tree.calcImpedanceMatrix(fit_locs)[0].real
        ctree_leak = greens_tree.createCompartmentTree(fit_locs)
        ctree_leak.computeGMC(z_mat)
        sov_tree = greens_tree.__copy__(new_tree=SOVTree())
        sov_tree.calcSOVEquations()
        alphas, phimat = sov_tree.getImportantModes(locarg=fit_locs)
        ctree_leak.computeC(-alphas[0:1].real*1e3, phimat[0:1,:].real)
        # make ball model with leak based on all channels
        tree = self.tree.__copy__()
        tree.asPassiveMembrane()
        greens_tree = tree.__copy__(new_tree=GreensTree())
        greens_tree.setCompTree()
        freqs = np.array([0.])
        greens_tree.setImpedance(freqs)
        z_mat = greens_tree.calcImpedanceMatrix(fit_locs)[0].real
        ctree_all = greens_tree.createCompartmentTree(fit_locs)
        ctree_all.computeGMC(z_mat)
        sov_tree = tree.__copy__(new_tree=SOVTree())
        sov_tree.calcSOVEquations()
        alphas, phimat = sov_tree.getImportantModes(locarg=fit_locs)
        ctree_all.computeC(-alphas[0:1].real*1e3, phimat[0:1,:].real)

        # new compartment fitter
        cm = CompartmentFitter(self.tree)
        cm.setCTree(fit_locs)
        # test fitting
        cm.fitPassive(use_all_channels=False)
        cm.fitCapacitance(use_all_channels=False)
        cm.fitEEq()
        self._checkPasCondProps(ctree_leak, cm.ctree)
        self._checkPasCaProps(ctree_leak, cm.ctree)
        with pytest.raises(AssertionError):
            self._checkEL(cm.ctree, self.tree[1].currents['L'][1])
        cm.fitPassive(use_all_channels=True)
        cm.fitCapacitance(use_all_channels=True)
        cm.fitEEq()
        self._checkPasCondProps(ctree_all, cm.ctree)
        self._checkPasCaProps(ctree_all, cm.ctree)
        self._checkEL(cm.ctree, greens_tree[1].currents['L'][1])
        with pytest.raises(AssertionError):
            self._checkEL(cm.ctree, self.tree[1].currents['L'][1])
        with pytest.raises(AssertionError):
            self._checkPasCondProps(ctree_leak, ctree_all)

    def testRecalcImpedanceMatrix(self, g_inp=np.linspace(0.,0.01, 20)):
        self.loadBall()
        fit_locs = [(1,.5)]
        cm = CompartmentFitter(self.tree)
        cm.setCTree(fit_locs)

        # test only leak
        # compute impedances explicitly
        greens_tree = cm.createTreeGF(channel_names=[])
        greens_tree.setEEq(-75.)
        greens_tree.setImpedancesInTree()
        z_mat = greens_tree.calcImpedanceMatrix(fit_locs, explicit_method=False)[0]
        z_test = z_mat[:,:,None] / (1. + z_mat[:,:,None] * g_inp[None,None,:])
        # compute impedances with compartmentfitter function
        z_calc = np.array([ \
                           cm.recalcImpedanceMatrix('fit locs', [g_i], \
                               channel_names=[]
                           ) \
                           for g_i in g_inp \
                          ])
        z_calc = np.swapaxes(z_calc, 0, 2)
        assert np.allclose(z_calc, z_test)

        # test with z based on all channels (passive)
        # compute impedances explicitly
        greens_tree = cm.createTreeGF(channel_names=list(cm.tree.channel_storage.keys()))
        greens_tree.setEEq(-75.)
        greens_tree.setImpedancesInTree()
        z_mat = greens_tree.calcImpedanceMatrix(fit_locs, explicit_method=False)[0]
        z_test = z_mat[:,:,None] / (1. + z_mat[:,:,None] * g_inp[None,None,:])
        # compute impedances with compartmentfitter function
        z_calc = np.array([ \
                           cm.recalcImpedanceMatrix('fit locs', [g_i], \
                               channel_names=list(cm.tree.channel_storage.keys())) \
                           for g_i in g_inp \
                          ])
        z_calc = np.swapaxes(z_calc, 0, 2)
        assert np.allclose(z_calc, z_test)

    def testSynRescale(self, g_inp=np.linspace(0.,0.01, 20)):
        e_rev, v_eq = 0., -75.
        self.loadBallAndStick()
        fit_locs = [(4,.7)]
        syn_locs = [(4,1.)]
        cm = CompartmentFitter(self.tree)
        cm.setCTree(fit_locs)
        # compute impedance matrix
        greens_tree = cm.createTreeGF(channel_names=[])
        greens_tree.setEEq(-75.)
        greens_tree.setImpedancesInTree()
        z_mat = greens_tree.calcImpedanceMatrix(fit_locs+syn_locs)[0]
        # analytical synapse scale factors
        beta_calc = 1. / (1. + (z_mat[1,1] - z_mat[0,0]) * g_inp)
        beta_full = z_mat[0,1] / z_mat[0,0] * (e_rev - v_eq) / \
                    ((1. + (z_mat[1,1] - z_mat[0,0]) * g_inp ) * (e_rev - v_eq))
        # synapse scale factors from compartment fitter
        beta_cm = np.array([cm.fitSynRescale(fit_locs, syn_locs, [0], [g_i], e_revs=[0.])[0] \
                            for g_i in g_inp])
        assert np.allclose(beta_calc, beta_cm, atol=.020)
        assert np.allclose(beta_full, beta_cm, atol=.015)


if __name__ == '__main__':
    tcf = TestCompartmentFitter()
    # tcf.testTreeStructure()
    # tcf.testCreateTreeGF()
    # tcf.testChannelFitMats()
    # tcf.testPassiveFit()
    # tcf.testRecalcImpedanceMatrix()
    tcf.testSynRescale()
