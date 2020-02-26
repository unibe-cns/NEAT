import numpy as np
import matplotlib.pyplot as pl

import neuron
from neuron import h

import pytest

import itertools
from neat import GreensTree
from neat import CompartmentNode, CompartmentTree
import neat.tools.kernelextraction as ke
from neat import NeuronSimTree, createReducedModel


colours = ['DeepPink', 'Purple', 'MediumSlateBlue', 'Blue', 'Teal',
                'ForestGreen',  'DarkOliveGreen', 'DarkGoldenRod',
                'DarkOrange', 'Coral', 'Red', 'Sienna', 'Black', 'DarkGrey']


class TestNeuron():
    def loadTTreePassive(self):
        """
        Load the T-tree morphology in memory with passive conductance

          6--5--4--7--8
                |
                |
                1
        """
        v_eq = -75.
        self.dt = 0.025
        self.tmax = 100.
        # for frequency derivation
        self.ft = ke.FourrierTools(np.arange(0., self.tmax, self.dt))
        # load the morphology
        print('>>> loading T-tree <<<')
        fname = 'test_morphologies/Tsovtree.swc'
        self.greenstree = GreensTree(fname, types=[1,3,4])
        self.greenstree.fitLeakCurrent(e_eq_target=v_eq, tau_m_target=10.)
        self.greenstree.setCompTree()
        self.greenstree.setImpedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(dt=self.dt, t_calibrate=10., v_eq=v_eq,
                                              factor_lambda=25.)
        self.greenstree.__copy__(self.neurontree)
        self.neurontree.treetype = 'computational'

    def loadTTreeActive(self):
        """
        Load the T-tree morphology in memory with h-current

          6--5--4--7--8
                |
                |
                1
        """
        v_eq = -75.
        self.dt = 0.1
        self.tmax = 100.
        # for frequency derivation
        self.ft = ke.FourrierTools(np.arange(0., self.tmax, self.dt))
        # load the morphology
        print('>>> loading T-tree <<<')
        fname = 'test_morphologies/Tsovtree.swc'
        self.greenstree = GreensTree(fname, types=[1,3,4])
        self.greenstree.addCurrent('h', 50., -43.)
        self.greenstree.fitLeakCurrent(e_eq_target=v_eq, tau_m_target=10.)
        self.greenstree.setCompTree()
        self.greenstree.setImpedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(dt=self.dt, t_calibrate=10., v_eq=v_eq,
                                              factor_lambda=25.)
        self.greenstree.__copy__(self.neurontree)
        self.neurontree.treetype = 'computational'

    def loadTTreeTestChannel(self):
        """
        Load the T-tree morphology in memory with h-current

          6--5--4--7--8
                |
                |
                1
        """
        v_eq = -75.
        self.dt = 0.025
        self.tmax = 100.
        # for frequency derivation
        self.ft = ke.FourrierTools(np.arange(0., self.tmax, self.dt))
        # load the morphology
        print('>>> loading T-tree <<<')
        fname = 'test_morphologies/Tsovtree.swc'
        self.greenstree = GreensTree(fname, types=[1,3,4])
        self.greenstree.addCurrent('TestChannel2', 50., -23.)
        self.greenstree.fitLeakCurrent(e_eq_target=v_eq, tau_m_target=10.)
        for node in self.greenstree:
            print(node)
        self.greenstree.setCompTree()
        self.greenstree.setImpedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(dt=self.dt, t_calibrate=100., v_eq=v_eq,
                                              factor_lambda=25.)
        self.greenstree.__copy__(self.neurontree)
        self.neurontree.treetype = 'computational'

    def loadTTreeTestChannelSoma(self):
        """
        Load the T-tree morphology in memory with h-current

          6--5--4--7--8
                |
                |
                1
        """
        v_eq = -75.
        self.dt = 0.025
        self.tmax = 100.
        # for frequency derivation
        self.ft = ke.FourrierTools(np.arange(0., self.tmax, self.dt))
        # load the morphology
        print('>>> loading T-tree <<<')
        fname = 'test_morphologies/Tsovtree.swc'
        self.greenstree = GreensTree(fname, types=[1,3,4])
        self.greenstree[1].addCurrent('TestChannel2', 50., e_rev=23.)
        self.greenstree.fitLeakCurrent(e_eq_target=v_eq, tau_m_target=10.)
        # for node in self.greenstree:
        #     print node.getGTot(channel_storage=self.greenstree.channel_storage)
        #     print node.currents
        self.greenstree.setCompTree()
        self.greenstree.setImpedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(dt=self.dt, t_calibrate=100., v_eq=v_eq,
                                              factor_lambda=25.)
        self.greenstree.__copy__(self.neurontree)
        self.neurontree.treetype = 'computational'

    def testPassive(self, pplot=False):
        self.loadTTreePassive()
        # set of locations
        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (6, .5), (7, .5), (8, .5)]
        # compute impedance matrix with Green's function
        zf_mat_gf = self.greenstree.calcImpedanceMatrix(locs)
        z_mat_gf = zf_mat_gf[self.ft.ind_0s].real
        # convert impedance matrix to time domain
        zk_mat_gf = np.zeros((len(self.ft.t), len(locs), len(locs)))
        for (ii, jj) in itertools.product(list(range(len(locs))), list(range(len(locs)))):
            zk_mat_gf[:,ii,jj] = self.ft.FT_inv(zf_mat_gf[:,ii,jj])[1].real * 1e-3
        # test the steady state impedance matrix
        z_mat_neuron = self.neurontree.calcImpedanceMatrix(locs)
        assert np.allclose(z_mat_gf, z_mat_neuron, atol=1.)
        # test the temporal matrix
        tk, zk_mat_neuron = self.neurontree.calcImpedanceKernelMatrix(locs)
        assert np.allclose(zk_mat_gf[int(2./self.dt):,:,:],
                           zk_mat_neuron[int(2./self.dt):,:,:], atol=.2)
        if pplot:
            # plot kernels
            pl.figure()
            cc = 0
            for ii in range(len(locs)):
                jj = 0
                while jj <= ii:
                    pl.plot(tk, zk_mat_neuron[:,ii,jj], c=colours[cc%len(colours)])
                    pl.plot(tk, zk_mat_gf[:,ii,jj], ls='--', lw=2, c=colours[cc%len(colours)])
                    cc += 1
                    jj += 1
            pl.show()

    def testActive(self, pplot=False):
        self.loadTTreeActive()
        # set of locations
        locs = [(1, .5), (4, .5), (6, .5), (7, .5), (8, .5)]
        # compute impedance matrix with Green's function
        zf_mat_gf = self.greenstree.calcImpedanceMatrix(locs)
        z_mat_gf = zf_mat_gf[self.ft.ind_0s].real
        # convert impedance matrix to time domain
        zk_mat_gf = np.zeros((len(self.ft.t), len(locs), len(locs)))
        for (ii, jj) in itertools.product(list(range(len(locs))), list(range(len(locs)))):
            zk_mat_gf[:,ii,jj] = self.ft.FT_inv(zf_mat_gf[:,ii,jj])[1].real * 1e-3
        # test the steady state impedance matrix
        z_mat_neuron = self.neurontree.calcImpedanceMatrix(locs, t_dur=500.)
        assert np.allclose(z_mat_gf, z_mat_neuron, atol=5.)
        # test the temporal matrix
        tk, zk_mat_neuron = self.neurontree.calcImpedanceKernelMatrix(locs)
        assert np.allclose(zk_mat_gf[int(2./self.dt):,:,:],
                           zk_mat_neuron[int(2./self.dt):,:,:], atol=.5)
        if pplot:
            # plot kernels
            pl.figure()
            cc = 0
            for ii in range(len(locs)):
                jj = 0
                while jj <= ii:
                    pl.plot(tk, zk_mat_neuron[:,ii,jj], c=colours[cc%len(colours)])
                    pl.plot(tk, zk_mat_gf[:,ii,jj], ls='--', lw=2, c=colours[cc%len(colours)])
                    cc += 1
                    jj += 1
            pl.show()

    def testChannelRecording(self):
        self.loadTTreeTestChannel()
        # set of locations
        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (6, .5), (7, .5), (8, .5)]
        # create simulation tree
        self.neurontree.initModel(t_calibrate=100., factor_lambda=10.)
        self.neurontree.storeLocs(locs, name='rec locs')
        # run test simulation
        res = self.neurontree.run(10., record_from_channels=True)
        # check if results are stored correctly
        assert set(res['chan']['TestChannel2'].keys()) == {'a00', 'a01', 'a10', 'a11', 'p_open'}
        # check if values are correct
        assert np.allclose(res['chan']['TestChannel2']['a00'], .3)
        assert np.allclose(res['chan']['TestChannel2']['a01'], .5)
        assert np.allclose(res['chan']['TestChannel2']['a10'], .4)
        assert np.allclose(res['chan']['TestChannel2']['a11'], .6)
        assert np.allclose(res['chan']['TestChannel2']['p_open'], .9 * .3**3 * .5**2 + .1 * .4**2 * .6**1)
        # check if shape is correct
        n_loc, n_step = len(locs), len(res['t'])
        assert res['chan']['TestChannel2']['a00'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['a01'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['a10'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['a11'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['p_open'].shape == (n_loc, n_step)
        # channel only at soma
        self.loadTTreeTestChannelSoma()
        # create simulation tree
        self.neurontree.initModel(t_calibrate=100., factor_lambda=10.)
        self.neurontree.storeLocs(locs, name='rec locs')
        # run test simulation
        res = self.neurontree.run(10., record_from_channels=True)
        # check if results are stored correctly
        assert set(res['chan']['TestChannel2'].keys()) == {'a00', 'a01', 'a10', 'a11', 'p_open'}
        # check if values are correct
        assert np.allclose(res['chan']['TestChannel2']['a00'][0,:], .3)
        assert np.allclose(res['chan']['TestChannel2']['a01'][0,:], .5)
        assert np.allclose(res['chan']['TestChannel2']['a10'][0,:], .4)
        assert np.allclose(res['chan']['TestChannel2']['a11'][0,:], .6)
        assert np.allclose(res['chan']['TestChannel2']['p_open'][0,:], .9 * .3**3 * .5**2 + .1 * .4**2 * .6**1)
        assert np.allclose(res['chan']['TestChannel2']['a00'][1:,:], 0.)
        assert np.allclose(res['chan']['TestChannel2']['a01'][1:,:], 0.)
        assert np.allclose(res['chan']['TestChannel2']['a10'][1:,:], 0.)
        assert np.allclose(res['chan']['TestChannel2']['a11'][1:,:], 0.)
        assert np.allclose(res['chan']['TestChannel2']['p_open'][1:,:], 0.)
        # check if shape is correct
        n_loc, n_step = len(locs), len(res['t'])
        assert res['chan']['TestChannel2']['a00'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['a01'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['a10'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['a11'].shape == (n_loc, n_step)
        assert res['chan']['TestChannel2']['p_open'].shape == (n_loc, n_step)


class TestReducedNeuron():
    def addLocinds(self):
        for ii, cn in enumerate(self.ctree):
            cn.loc_ind = ii

    def loadTwoCompartmentModel(self, w_locinds=True):
        # simple two compartment model
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(root=pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.addNodeWithParent(cnode, pnode)

        if w_locinds:
            self.addLocinds()

    def loadTModel(self, w_locinds=True):
        # simple T compartment model
        # pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        # self.ctree = CompartmentTree(root=pnode)
        # cnode = CompartmentNode(1, ca=1.5e-6, g_l=2.5e-4, g_c=2e-3)
        # self.ctree.addNodeWithParent(cnode, pnode)
        # lnode0 = CompartmentNode(2, ca=1.5e-6, g_l=2.5e-4, g_c=2e-3)
        # self.ctree.addNodeWithParent(lnode0, cnode)
        # lnode1 = CompartmentNode(3, ca=1.5e-6, g_l=2.5e-4, g_c=2e-3)
        # self.ctree.addNodeWithParent(lnode1, cnode)
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(root=pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.addNodeWithParent(cnode, pnode)
        lnode0 = CompartmentNode(2, ca=1.5e-6, g_l=2.5e-4, g_c=3e-3)
        self.ctree.addNodeWithParent(lnode0, cnode)
        lnode1 = CompartmentNode(3, ca=1.5e-6, g_l=2.5e-4, g_c=5e-3)
        self.ctree.addNodeWithParent(lnode1, cnode)

        if w_locinds:
            self.addLocinds()

    def loadThreeCompartmentModel(self, w_locinds=True):
        # simple 3 compartment model
        pnode = CompartmentNode(0, ca=1.9e-6, g_l=1.8e-3)
        self.ctree = CompartmentTree(root=pnode)
        cnode = CompartmentNode(1, ca=2.4e-6, g_l=0.3e-4, g_c=3.9)
        self.ctree.addNodeWithParent(cnode, pnode)
        lnode0 = CompartmentNode(2, ca=1.9e-6, g_l=0.3e-4, g_c=3.8e-3)
        self.ctree.addNodeWithParent(lnode0, cnode)

        if w_locinds:
            self.addLocinds()

    def loadMultiDendModel(self, w_locinds=True):
        # simple 3 compartment model
        pnode = CompartmentNode(0, ca=1.9e-6, g_l=1.8e-3)
        self.ctree = CompartmentTree(root=pnode)
        cnode0 = CompartmentNode(1, ca=2.4e-6, g_l=0.3e-4, g_c=3.9)
        self.ctree.addNodeWithParent(cnode0, pnode)
        cnode1 = CompartmentNode(2, ca=1.9e-6, g_l=0.4e-4, g_c=3.8e-3)
        self.ctree.addNodeWithParent(cnode1, pnode)
        cnode2 = CompartmentNode(3, ca=1.3e-6, g_l=0.5e-4, g_c=2.7e-2)
        self.ctree.addNodeWithParent(cnode2, pnode)

        if w_locinds:
            self.addLocinds()

    def testGeometry1(self):
        fake_c_m = 1.
        fake_r_a = 100.*1e-6
        factor_r_a = 1e-6

        ## test method 1
        # test with two compartments
        self.loadTwoCompartmentModel()
        ctree = self.ctree
        # check if fake geometry is correct
        points, _ = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                              factor_r_a=1e-6, delta=1e-14,
                                              method=1)
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            h.pt3dadd(*points[ii][0], sec=comps[-1])
            h.pt3dadd(*points[ii][1], sec=comps[-1])
            h.pt3dadd(*points[ii][2], sec=comps[-1])
            h.pt3dadd(*points[ii][3], sec=comps[-1])
            comps[-1].Ra = fake_r_a*1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1
        # check areas
        assert np.abs(comps[0](0.5).area()*1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area()*1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs(comps[0](0.5).ri() - 1.) < 1e-6
        assert np.abs((comps[1](0.5).ri() - 1./ctree[1].g_c) / comps[1](0.5).ri()) < 1e-6
        assert np.abs((comps[1](0.5).ri() * factor_r_a - comps[1](1.).ri()) / comps[1](1.).ri()) < 1e-6

        # test with three compartments
        self.loadThreeCompartmentModel()
        ctree = self.ctree
        # check if fake geometry is correct
        points, _ = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                              factor_r_a=1e-6, delta=1e-14,
                                              method=1)
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            h.pt3dadd(*points[ii][0], sec=comps[-1])
            h.pt3dadd(*points[ii][1], sec=comps[-1])
            h.pt3dadd(*points[ii][2], sec=comps[-1])
            h.pt3dadd(*points[ii][3], sec=comps[-1])
            comps[-1].Ra = fake_r_a*1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1
        # check areas
        assert np.abs(comps[0](0.5).area()*1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area()*1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area()*1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs(comps[0](0.5).ri() - 1.) < 1e-6
        assert np.abs((comps[1](0.5).ri() - 1./ctree[1].g_c) / comps[1](0.5).ri()) < 1e-6
        assert np.abs((comps[2](0.5).ri() - 1./ctree[2].g_c) / comps[2](0.5).ri()) < 1e-6
        assert np.abs((comps[1](0.5).ri() * factor_r_a - comps[1](1.).ri()) / comps[1](1.).ri()) < 1e-6
        assert np.abs((comps[2](0.5).ri() * factor_r_a - comps[2](1.).ri()) / comps[2](1.).ri()) < 1e-6

        # test the T model
        self.loadTModel()
        ctree = self.ctree
        # check if fake geometry is correct
        points, _ = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                              factor_r_a=1e-6, delta=1e-14,
                                              method=1)
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            h.pt3dadd(*points[ii][0], sec=comps[-1])
            h.pt3dadd(*points[ii][1], sec=comps[-1])
            h.pt3dadd(*points[ii][2], sec=comps[-1])
            h.pt3dadd(*points[ii][3], sec=comps[-1])
            comps[-1].Ra = fake_r_a*1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1
        # check areas
        assert np.abs(comps[0](0.5).area()*1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area()*1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area()*1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        assert np.abs(comps[3](0.5).area()*1e-8 * fake_c_m - ctree[3].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs(comps[0](0.5).ri() - 1.) < 1e-6
        assert np.abs((comps[1](0.5).ri() - 1./ctree[1].g_c) / comps[1](0.5).ri()) < 1e-6
        assert np.abs((comps[2](0.5).ri() - 1./ctree[2].g_c) / comps[2](0.5).ri()) < 1e-6
        assert np.abs((comps[3](0.5).ri() - 1./ctree[3].g_c) / comps[3](0.5).ri()) < 1e-6
        assert np.abs((comps[1](0.5).ri() * factor_r_a - comps[1](1.).ri()) / comps[1](1.).ri()) < 1e-6
        assert np.abs((comps[2](0.5).ri() * factor_r_a - comps[2](1.).ri()) / comps[2](1.).ri()) < 1e-6
        assert np.abs((comps[3](0.5).ri() * factor_r_a - comps[3](1.).ri()) / comps[3](1.).ri()) < 1e-6


    def testImpedanceProperties1(self):
        fake_c_m = 1.
        fake_r_a = 100.*1e-6
        # create the two compartment model without locinds
        self.loadTwoCompartmentModel(w_locinds=False)
        ctree = self.ctree
        # check if error is raised if loc_inds have not been set
        with pytest.raises(AttributeError):
            ctree.calcImpedanceMatrix()

        # create the two compartment model with locinds
        self.loadTwoCompartmentModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=1)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5)])
        # assert np.allclose(z_mat_sim, z_mat_comp, atol=1e-2)

        # create the three compartmental model
        self.loadThreeCompartmentModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=1)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5), (2,0.5)])
        # assert np.allclose(z_mat_sim, z_mat_comp)

        # create the T compartmental model
        self.loadTModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=1)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5), (2,0.5), (3,0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp)

        # create the multidend model
        self.loadMultiDendModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=1)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5), (2,0.5), (3,0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp)

    def testGeometry2(self):
        fake_c_m = 1.
        fake_r_a = 100.*1e-6
        factor_r_a = 1e-6

        ## test method 2
        # test with two compartments
        self.loadTwoCompartmentModel()
        ctree = self.ctree
        # check if fake geometry is correct
        lengths, radii = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                                   factor_r_a=1e-6, delta=1e-14,
                                                   method=2)
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            comps[-1].diam = 2.*radii[ii] * 1e4
            comps[-1].L = lengths[ii] * 1e4
            comps[-1].Ra = fake_r_a * 1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1

        # check areas
        assert np.abs(comps[0](0.5).area()*1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area()*1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs((comps[1](0.5).ri() - 1./ctree[1].g_c) / comps[1](0.5).ri()) < 1e-12
        assert np.abs((comps[1](0.5).ri()  - comps[1](1.).ri()) / comps[1](1.).ri()) < 1e-12

        # test with three compartments
        self.loadThreeCompartmentModel()
        ctree = self.ctree
        # check if fake geometry is correct
        lengths, radii = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                                   factor_r_a=1e-6, delta=1e-14,
                                                   method=2)
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            comps[-1].diam = 2.*radii[ii] * 1e4
            comps[-1].L = lengths[ii] * 1e4
            comps[-1].Ra = fake_r_a*1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1

        # check areas
        assert np.abs(comps[0](0.5).area()*1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area()*1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area()*1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs((comps[1](0.5).ri() - 1./ctree[1].g_c) / comps[1](0.5).ri()) < 1e-12
        assert np.abs((comps[1](0.5).ri()  - comps[1](1.).ri()) / comps[1](1.).ri()) < 1e-12
        assert np.abs((comps[2](0.5).ri() - 1./ctree[2].g_c) / comps[2](0.5).ri()) < 1e-12
        assert np.abs((comps[2](0.5).ri()  - comps[2](1.).ri()) / comps[2](1.).ri()) < 1e-12

        # test the T model
        self.loadTModel()
        ctree = self.ctree
        # check if fake geometry is correct
        lengths, radii = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                                   factor_r_a=1e-6, delta=1e-14,
                                                   method=2)
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            comps[-1].diam = 2.*radii[ii] * 1e4
            comps[-1].L = lengths[ii] * 1e4
            comps[-1].Ra = fake_r_a*1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1

        # check areas
        assert np.abs(comps[0](0.5).area()*1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area()*1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area()*1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        assert np.abs(comps[3](0.5).area()*1e-8 * fake_c_m - ctree[3].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs((comps[1](0.5).ri() - 1./ctree[1].g_c) / comps[1](0.5).ri()) < 1e-12
        assert np.abs((comps[1](0.5).ri()  - comps[1](1.).ri()) / comps[1](1.).ri()) < 1e-12
        assert np.abs((comps[2](0.5).ri() - 1./ctree[2].g_c) / comps[2](0.5).ri()) < 1e-12
        assert np.abs((comps[2](0.5).ri()  - comps[2](1.).ri()) / comps[2](1.).ri()) < 1e-12
        assert np.abs((comps[3](0.5).ri() - 1./ctree[3].g_c) / comps[3](0.5).ri()) < 1e-12
        assert np.abs((comps[3](0.5).ri()  - comps[3](1.).ri()) / comps[3](1.).ri()) < 1e-12


    def testImpedanceProperties2(self):
        fake_c_m = 1.
        fake_r_a = 100.*1e-6
        # create the two compartment model
        self.loadTwoCompartmentModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=2)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp, atol=1e-2)

        # create the three compartmental model
        self.loadThreeCompartmentModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=2)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5), (2,0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp)

        # create the T compartmental model
        self.loadTModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=2)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5), (2,0.5), (3,0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp)

        # create the multidend model
        self.loadMultiDendModel()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calcImpedanceMatrix()
        # create a neuron model
        sim_tree = createReducedModel(ctree, fake_c_m=fake_c_m,
                                                   fake_r_a=fake_r_a,
                                                   method=2)
        # sim_tree.initModel(pprint=True)
        z_mat_sim = sim_tree.calcImpedanceMatrix([(0,0.5), (1,0.5), (2,0.5), (3,0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp)


if __name__ == '__main__':
    tn = TestNeuron()
    # tn.testPassive(pplot=True)
    # tn.testActive()
    tn.testChannelRecording()

    # trn = TestReducedNeuron()
    # trn.testGeometry1()
    # trn.testImpedanceProperties1()
    # trn.testGeometry2()
    # trn.testImpedanceProperties2()
