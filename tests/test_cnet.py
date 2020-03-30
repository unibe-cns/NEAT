import numpy as np
import os

import pytest

from neat import netsim
from neat import NETNode, NET, Kernel
from neat import GreensTree, NeuronSimTree, SOVTree

from neat.channels.channelcollection import channelcollection


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_morphologies'))


class TestCNET():
    def createPointNeurons(self, v_eq=-75.):
        self.v_eq = v_eq
        self.dt = .025
        gh, eh = 50., -43.
        h_chan = channelcollection.h()

        self.greens_tree = GreensTree(file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball.swc'))
        self.greens_tree.setPhysiology(1., 100./1e6)
        self.greens_tree.addCurrent(h_chan, gh, eh)
        self.greens_tree.fitLeakCurrent(v_eq, 10.)
        self.greens_tree.setEEq(v_eq)
        self.greens_tree_pas = self.greens_tree.__copy__(new_tree=GreensTree())
        self.greens_tree_pas.asPassiveMembrane()
        self.sim_tree = self.greens_tree.__copy__(new_tree=NeuronSimTree())
        # set the impedances
        self.greens_tree_pas.setCompTree()
        self.freqs = np.array([0.])
        self.greens_tree_pas.setImpedance(self.freqs)
        # create sov tree
        self.sov_tree = self.greens_tree_pas.__copy__(new_tree=SOVTree())
        self.sov_tree.calcSOVEquations(maxspace_freq=50.)

        z_inp = self.greens_tree_pas.calcZF((1,.5), (1,.5))[0]
        alphas, gammas = self.sov_tree.getSOVMatrices(locarg=[(1.,.5)])
        # create NET
        node_0 = NETNode(0, [0], [0], z_kernel=(alphas, gammas[:,0]**2))
        net_py = NET()
        net_py.setRoot(node_0)
        # check if correct
        assert np.abs(gammas[0,0]**2/np.abs(alphas[0]) - z_inp) < 1e-10
        assert np.abs(node_0.z_bar - z_inp) < 1e-10

        # to initialize neuron tree
        self.sim_tree.initModel(dt=self.dt)
        # add ion channel to NET simulator
        a_soma = 4. *  np.pi * (self.sim_tree[1].R*1e-4)**2
        self.cnet = netsim.NETSim(net_py, v_eq=self.v_eq)

        hchan = channelcollection.h()
        self.cnet.addChannel(hchan, 0, gh*a_soma, eh)

        # add the synapse
        # to neuron tree
        self.sim_tree.addDoubleExpSynapse((1,.5), .2, 3., 0.)
        self.sim_tree.setSpikeTrain(0, 0.001, [5.])
        # to net sim
        self.cnet.addSynapse(0, {'tau_r': .2, 'tau_d': 3., 'e_r': 0.}, g_max=0.001)
        self.cnet.setSpikeTimes(0, [5.+self.dt])

    def createTree(self, reinitialize=1, v_eq=-75.):
        """
        Create simple NET structure

        2     3
        |     |
        |     |
        ---1---
           |
           |
           0
           |
        """
        self.v_eq = v_eq
        loc_ind = np.array([0,1,2])

        # kernel constants
        alphas = 1. / np.array([.5, 8.]); gammas = np.array([-1.,1.])
        alphas_ = 1. / np.array([1.]); gammas_ = np.array([1.])
        # nodes
        node_0 = NETNode(0, [0,1,2], [],  z_kernel=(alphas, gammas))
        node_1 = NETNode(1, [0,1,2], [0],  z_kernel=(alphas_, gammas_))
        node_2 = NETNode(2, [1], [1],  z_kernel=(alphas_, gammas_))
        node_3 = NETNode(3, [2], [2],  z_kernel=(alphas_, gammas_))
        # add nodes to tree
        net_py = NET()
        net_py.setRoot(node_0)
        net_py.addNodeWithParent(node_1, node_0)
        net_py.addNodeWithParent(node_2, node_1)
        net_py.addNodeWithParent(node_3, node_1)
        # store
        self.net_py = net_py
        self.cnet = netsim.NETSim(net_py, v_eq=self.v_eq)

    def createTree2(self, reinitialize=1, add_lin=True, v_eq=-75.):
        """
        Create simple NET structure

                3     4
                |     |
                |     |
                ---2---
             1     |
             |     |
             ---0---
                |
        """
        self.v_eq = v_eq
        loc_ind = np.array([0,1,2])

        # kernel constants
        alphas = 1. / np.array([1.]); gammas = np.array([1.])
        # nodes
        node_0 = NETNode(0, [0,1,2], [],  z_kernel=(alphas, gammas))
        node_1 = NETNode(1, [0], [0],  z_kernel=(alphas, gammas))
        node_2 = NETNode(2, [1,2], [],  z_kernel=(alphas, gammas))
        node_3 = NETNode(3, [1], [1],  z_kernel=(alphas, gammas))
        node_4 = NETNode(4, [2], [2],  z_kernel=(alphas, gammas))
        # add nodes to tree
        net_py = NET()
        net_py.setRoot(node_0)
        net_py.addNodeWithParent(node_1, node_0)
        net_py.addNodeWithParent(node_2, node_0)
        net_py.addNodeWithParent(node_3, node_2)
        net_py.addNodeWithParent(node_4, node_2)
        # linear terms
        alphas = 1. / np.array([1.])
        gammas = np.array([1.])
        self.lin_terms = {1: Kernel((alphas, gammas)),
                          2: Kernel((alphas, gammas))} if add_lin else {}
        # store
        self.net_py = net_py
        self.cnet = netsim.NETSim(net_py, lin_terms=self.lin_terms,
                                  v_eq=self.v_eq)

    def createTree3(self, reinitialize=1, add_lin=True, v_eq=-75.):
        """
        Create simple NET structure

                         6
                4     5  |
                |     |  |
                |     |  |
             2  ---3---  |
             |     |     |
             ---1---     |
                   |     |
                   0------
                   |
        """
        self.v_eq = v_eq

        # kernel constants
        alphas = 1. / np.array([1.]); gammas = np.array([1.])
        # nodes
        node_0 = NETNode(0, [0,1,2,3], [],  z_kernel=(alphas, gammas))
        node_1 = NETNode(1, [0,1,2], [],  z_kernel=(alphas, gammas))
        node_2 = NETNode(2, [0], [0],  z_kernel=(alphas, gammas))
        node_3 = NETNode(3, [1,2], [],  z_kernel=(alphas, gammas))
        node_4 = NETNode(4, [1], [1],  z_kernel=(alphas, gammas))
        node_5 = NETNode(5, [2], [2],  z_kernel=(alphas, gammas))
        node_6 = NETNode(6, [3], [3],  z_kernel=(alphas, gammas))
        # add nodes to tree
        net_py = NET()
        net_py.setRoot(node_0)
        net_py.addNodeWithParent(node_1, node_0)
        net_py.addNodeWithParent(node_2, node_1)
        net_py.addNodeWithParent(node_3, node_1)
        net_py.addNodeWithParent(node_4, node_3)
        net_py.addNodeWithParent(node_5, node_3)
        net_py.addNodeWithParent(node_6, node_0)
        # linear terms
        alphas = 1. / np.array([1.])
        gammas = np.array([1.])
        self.lin_terms = {1: Kernel((alphas, gammas)),
                          2: Kernel((alphas, gammas)),
                          3: Kernel((alphas, gammas))} if add_lin else {}
        # store
        self.net_py = net_py
        self.cnet = netsim.NETSim(net_py, lin_terms=self.lin_terms)

    def testIOFunctions(self):
        self.createTree()
        # storing and reading voltages from node voltage
        vnode = np.array([8.,10.,12.,14.])
        self.cnet.setVNodeFromVNode(vnode)
        vnode_back1 = self.cnet.getVNode()
        vnode_back2 = np.zeros(4)
        self.cnet.addVNodeToArr(vnode_back2)
        assert np.allclose(vnode_back1, vnode)
        assert np.allclose(vnode_back2, vnode)
        vloc_back1 = self.cnet.getVLoc()
        vloc_back2 = np.zeros(3)
        self.cnet.addVLocToArr(vloc_back2)
        assert np.allclose(vloc_back1, np.array([18.,30.,32.])+self.v_eq)
        assert np.allclose(vloc_back2, np.array([18.,30.,32.])+self.v_eq)
        with pytest.raises(ValueError):
            self.cnet.setVNodeFromVNode(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.addVNodeToArr(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.setVNodeFromVLoc(np.zeros(4))
        with pytest.raises(ValueError):
            self.cnet.addVLocToArr(np.zeros(4))
        # storing and reading voltages from location voltage
        vloc = np.array([12.,14.,16.])+self.v_eq
        self.cnet.setVNodeFromVLoc(vloc)
        vnode_back1 = self.cnet.getVNode()
        vnode_back2 = np.zeros(4)
        self.cnet.addVNodeToArr(vnode_back2)
        assert np.allclose(vnode_back1, np.array([0.,12.,2.,4.]))
        assert np.allclose(vnode_back2, np.array([0.,12.,2.,4.]))
        vloc_back1 = self.cnet.getVLoc()
        vloc_back2 = np.zeros(3)
        self.cnet.addVLocToArr(vloc_back2)
        assert np.allclose(vloc_back1, vloc)
        assert np.allclose(vloc_back2, vloc)
        with pytest.raises(ValueError):
            self.cnet.setVNodeFromVNode(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.addVNodeToArr(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.setVNodeFromVLoc(np.zeros(4))
        with pytest.raises(ValueError):
            self.cnet.addVLocToArr(np.zeros(4))

    def testSolver(self):
        self.createTree()
        netp = self.net_py
        # test if single AMPA synapse agrees with analytical solution
        # add synapse
        self.cnet.addSynapse(1, "AMPA")
        g_syn = 1.
        g_list = [np.array([]), np.array([g_syn]), np.array([])]
        # solve numerically
        v_loc = self.cnet.solveNewton(g_list)
        v_node = self.cnet.getVNode()
        # solve analytically
        g_rescale = g_syn / (1. + netp[2].z_bar * g_syn)
        z_0plus1 = netp[0].z_bar + netp[1].z_bar
        v_0plus1 = z_0plus1 * g_rescale / (1. + z_0plus1 * g_rescale) * \
                   (0. - self.v_eq)
        v_2 = netp[2].z_bar * g_rescale * (0. - self.v_eq - v_0plus1)
        # test if both solutions agree
        assert np.abs(v_node[0] + v_node[1] - v_0plus1) < 1e-9
        assert np.abs(v_node[2] - v_2) < 1e-9
        assert np.abs(v_node[3] - 0.0) < 1e-9
        assert np.abs(v_loc[0] - self.v_eq - v_0plus1) < 1e-9
        assert np.abs(v_loc[1] - self.v_eq - v_0plus1 - v_2) < 1e-9
        assert np.abs(v_loc[2] - self.v_eq - v_0plus1) < 1e-9
        # test if AMPA and GABA synapses agree with analytical solution
        # add synapse
        self.cnet.addSynapse(2, "GABA")
        g_exc = 1.
        g_inh = 1.
        g_list = [np.array([]), np.array([g_exc]), np.array([g_inh])]
        # solve numerically
        v_loc = self.cnet.solveNewton(g_list)
        v_node = self.cnet.getVNode()
        # solve analytically
        g_exc_ = g_exc / (1. + netp[2].z_bar * g_exc)
        g_inh_ = g_inh / (1. + netp[3].z_bar * g_inh)
        z_0plus1 = netp[0].z_bar + netp[1].z_bar
        v_0plus1 = z_0plus1 * g_exc_ / (1. + z_0plus1 * (g_exc_ + g_inh_)) * \
                   (0. - self.v_eq) + \
                   z_0plus1 * g_inh_ / (1. + z_0plus1 * (g_exc_ + g_inh_)) * \
                   (-80. - self.v_eq)
        v_2 = netp[2].z_bar * g_exc_ * (0. - self.v_eq - v_0plus1)
        v_3 = netp[3].z_bar * g_inh_ * (-80. - self.v_eq - v_0plus1)
        # test if both solutions agree
        assert np.abs(v_node[0] + v_node[1] - v_0plus1) < 1e-9
        assert np.abs(v_node[2] - v_2) < 1e-9
        assert np.abs(v_node[3] - v_3) < 1e-9
        assert np.abs(v_loc[0] - self.v_eq - v_0plus1) < 1e-9
        assert np.abs(v_loc[1] - self.v_eq - v_0plus1 - v_2) < 1e-9
        assert np.abs(v_loc[2] - self.v_eq - v_0plus1 - v_3) < 1e-9
        # test if NMDA synapse is solved correctly
        # check if removing synapse works correctly
        self.cnet.removeSynapseFromLoc(1, 0)
        self.cnet.removeSynapseFromLoc(2, 0)
        with pytest.raises(IndexError):
            self.cnet.removeSynapseFromLoc(3, 0)
        with pytest.raises(IndexError):
            self.cnet.removeSynapseFromLoc(1, 2)
        # create NMDA synapse
        self.cnet.addSynapse(1, "NMDA")
        # solve for low conductance
        g_syn_low = 1.
        g_list = [np.array([]), np.array([g_syn_low]), np.array([])]
        # solve numerically
        v_loc_low = self.cnet.solveNewton(g_list)
        v_node_low = self.cnet.getVNode()
        # solve for high conductance
        g_syn_high = 4.
        g_list = [np.array([]), np.array([g_syn_high]), np.array([])]
        # solve numerically
        v_loc_high = self.cnet.solveNewton(g_list)
        v_node_high = self.cnet.getVNode()
        # solve for moderate conductance
        g_syn_middle = 2.
        g_list = [np.array([]), np.array([g_syn_middle]), np.array([])]
        # solve numerically
        v_loc_middle_0 = self.cnet.solveNewton(g_list)
        v_node_middle_0 = self.cnet.getVNode()
        v_loc_middle_1 = self.cnet.solveNewton(g_list, v_0=np.array([0.,0.,0.]),
                                                    v_alt=self.v_eq*np.ones(3))
        v_node_middle_1 = self.cnet.getVNode()
        # check if correct
        z_sum = netp[0].z_bar + netp[1].z_bar + netp[2].z_bar
        checkfun = lambda vv: (0. - vv) / (1. + 0.3 * np.exp(-.1 * vv))
        vv = v_loc_low[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_low * checkfun(vv)) < 1e-3
        vv = v_loc_high[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_high * checkfun(vv)) < 1e-3
        vv = v_loc_middle_0[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_middle * checkfun(vv)) < 1e-3
        vv = v_loc_middle_1[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_middle * checkfun(vv)) < 1e-3
        assert np.abs(v_loc_middle_0[1] - v_loc_middle_1[1]) > 10.

    def testIntegration(self):
        tmax = 1000.; dt = 0.025
        self.createTree()
        # add synapse and check additional synapse functions
        self.cnet.addSynapse(1, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(1, "AMPA+NMDA", g_max=1., nmda_ratio=5.)
        assert self.cnet.syn_map_py[0] == {'loc_index': 1, 'syn_index_at_loc': 0,
                                            'n_syn_at_loc': 1, 'g_max': [dt*0.1]}
        assert self.cnet.syn_map_py[1] == {'loc_index': 1, 'syn_index_at_loc': 1,
                                            'n_syn_at_loc': 2, 'g_max': [1., 5.]}
        assert self.cnet.n_syn[1] == 3
        with pytest.raises(ValueError):
            self.cnet.addSynapse(1, "NONSENSE")
        with pytest.raises(IndexError):
            self.cnet.addSynapse(8, "AMPA")
        with pytest.raises(TypeError):
            self.cnet.addSynapse(1, ["NONSENSE LIST"])
        # check if synapse is correctly removed
        self.cnet.removeSynapse(1)
        assert len(self.cnet.syn_map_py) == 1
        # add spike times
        self.cnet.setSpikeTimes(0, np.arange(dt/2., tmax, dt))
        # run sim
        res = self.cnet.runSim(tmax, dt, step_skip=1, rec_v_node=True,
                                                      rec_g_syn_inds=[0])
        v_loc_sim = res['v_loc'][:,-1]
        # solve newton
        v_loc_newton = self.cnet.solveNewton([res['g_syn'][0][0][-1]])
        # compare
        assert np.allclose(v_loc_sim, v_loc_newton, atol=0.5)
        # do again with other synapses
        self.cnet.removeSynapse(0)
        self.cnet.addSynapse(2, "GABA", g_max=dt*0.1)
        self.cnet.addSynapse(1, "AMPA", g_max=dt*0.1)
        # add spike times
        self.cnet.setSpikeTimes(0, np.arange(dt/2., tmax, dt)) # GABA synapse
        self.cnet.setSpikeTimes(1, np.arange(dt/2., tmax, dt)) # AMPA synapse
        # run sim
        res = self.cnet.runSim(tmax, dt, step_skip=1, rec_v_node=True,
                                                      rec_g_syn_inds=[0,1])
        v_loc_sim = res['v_loc'][:,-1]
        g_newton = [res['g_syn'][ii][0][-1] for ii in [0,1]]
        # solve newton
        v_loc_newton = self.cnet.solveNewton(g_newton)
        # compare
        assert np.allclose(v_loc_sim, v_loc_newton, atol=0.5)

        # add NMDA synapse
        self.cnet.addSynapse(0, "AMPA+NMDA", g_max=dt*0.1, nmda_ratio=5.)
        self.cnet.addSynapse(1, "AMPA+NMDA", g_max=dt*0.1, nmda_ratio=5.)
        # set spiketimes for second synapse
        self.cnet.setSpikeTimes(3, np.arange(dt/2., tmax, dt)) # AMPA+NMDA synapse
        # remove first AMPA+NMDA synapse to see if spike times are correctly re-allocated
        assert len(self.cnet.spike_times_py[2]) == 0
        self.cnet.removeSynapse(2)
        for spk_tm in self.cnet.spike_times_py:
            assert len(spk_tm) > 0
        # run sim
        res = self.cnet.runSim(tmax, dt, step_skip=1, rec_v_node=True,
                                                      rec_g_syn_inds=[0,1,2])
        v_loc_sim = res['v_loc'][:,-1]
        g_newton = [res['g_syn'][ii][0][-1] for ii in [0,1,2]]
        # solve newton
        v_loc_newton = self.cnet.solveNewton(g_newton)
        # compare
        assert np.allclose(v_loc_sim, v_loc_newton, atol=.5)

        # test whether sparse storage works
        ss = 33 # number of timesteps not a multiple of storage step
        # set spiketimes
        self.cnet.setSpikeTimes(0, np.array([5.]))
        self.cnet.setSpikeTimes(1, np.array([10.]))
        # run sim
        res1 = self.cnet.runSim(tmax, dt, step_skip=1, rec_v_node=True,
                                                      rec_g_syn_inds=[0,1,2])
        # set spiketimes
        self.cnet.setSpikeTimes(0, np.array([5.]))
        self.cnet.setSpikeTimes(1, np.array([10.]))
        # run sim
        res2 = self.cnet.runSim(tmax, dt, step_skip=ss, rec_v_node=True,
                                                      rec_g_syn_inds=[0,1,2])
        # check if results are idendtical
        assert len(res2['t']) == len(res2['v_loc'][0])
        np.allclose(res1['v_loc'][0][ss-1:][::ss], res2['v_loc'][0])
        np.allclose(res1['v_node'][0][ss-1:][::ss], res2['v_node'][0])
        np.allclose(res1['g_syn'][0][0][ss-1:][::ss], res2['g_syn'][0])
        # test whether sparse storage works
        ss = 20 # number of timesteps a multiple of storage step
        # set spiketimes
        self.cnet.setSpikeTimes(0, np.array([5.]))
        self.cnet.setSpikeTimes(1, np.array([10.]))
        # run sim
        res1 = self.cnet.runSim(tmax, dt, step_skip=1, rec_v_node=True,
                                                      rec_g_syn_inds=[0,1,2])
        # set spiketimes
        self.cnet.setSpikeTimes(0, np.array([5.]))
        self.cnet.setSpikeTimes(1, np.array([10.]))
        # run sim
        res2 = self.cnet.runSim(tmax, dt, step_skip=ss, rec_v_node=True,
                                                      rec_g_syn_inds=[0,1,2])
        # check if results are idendtical
        assert len(res2['t']) == len(res2['v_loc'][0])
        np.allclose(res1['v_loc'][0][ss-1:][::ss], res2['v_loc'][0])
        np.allclose(res1['v_node'][0][ss-1:][::ss], res2['v_node'][0])
        np.allclose(res1['g_syn'][0][0][ss-1:][::ss], res2['g_syn'][0])

    def testInversion(self):
        dt = 0.1

        # tests without linear terms
        # test with two non-leafs that integrate soma
        self.createTree2(add_lin=False)
        # add synapses
        self.cnet.addSynapse(0, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(1, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(2, "AMPA", g_max=dt*0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recastInput([1.,1.,1.])
        self.cnet._constructInput(np.array([self.v_eq, self.v_eq, self.v_eq]), g_in)
        # recursive matrix inversion
        self.cnet.invertMatrix()
        v_node = self.cnet.getVNode()
        # construct full matrix
        mat, vec = self.cnet.getMatAndVec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)
        # test with two non-leafs that integrate soma
        self.createTree3(add_lin=False)
        # add synapses
        self.cnet.addSynapse(0, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(1, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(2, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(3, "AMPA", g_max=dt*0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recastInput(np.ones(4))
        self.cnet._constructInput(self.v_eq*np.ones(4), g_in)
        # recursive matrix inversion
        self.cnet.invertMatrix()
        v_node = self.cnet.getVNode()
        # construct full matrix
        mat, vec = self.cnet.getMatAndVec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)

        # tests with linear terms
        # test with one non-leafs that integrate soma
        self.createTree2(add_lin=True)
        # add synapses
        self.cnet.addSynapse(0, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(1, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(2, "AMPA", g_max=dt*0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recastInput([1.,1.,1.])
        self.cnet._constructInput(np.array([self.v_eq, self.v_eq, self.v_eq]), g_in)
        # recursive matrix inversion
        self.cnet.invertMatrix()
        v_node = self.cnet.getVNode()
        # construct full matrix
        mat, vec = self.cnet.getMatAndVec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)
        # test with two non-leafs that integrate soma
        self.createTree3(add_lin=True)
        # add synapses
        self.cnet.addSynapse(0, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(1, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(2, "AMPA", g_max=dt*0.1)
        self.cnet.addSynapse(3, "AMPA", g_max=dt*0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recastInput(np.ones(4))
        self.cnet._constructInput(self.v_eq*np.ones(4), g_in)
        # recursive matrix inversion
        self.cnet.invertMatrix()
        v_node = self.cnet.getVNode()
        # construct full matrix
        mat, vec = self.cnet.getMatAndVec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)

    def testChannel(self):
        self.createPointNeurons()
        # simulate neuron and NET model
        res_neuron = self.sim_tree.run(100)
        res_net = self.cnet.runSim(100., self.dt)
        # test if traces equal
        assert np.allclose(res_neuron['v_m'][0,:-1], res_net['v_loc'][0,:], atol=.1)


if __name__ == '__main__':
    tst = TestCNET()
    # tst.testIOFunctions()
    # tst.testSolver()
    # tst.testIntegration()
    # tst.testInversion()
    tst.testChannel()

