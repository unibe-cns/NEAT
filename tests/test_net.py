import matplotlib.pyplot as pl
import numpy as np
import pytest

from neat import Kernel, NETNode, NET


class TestNET():
    def loadTree(self, reinitialize=0):
        if not hasattr(self, 'net') or reinitialize:
            alphas = np.array([1.]);
            gammas = np.array([1.])
            # define nodes
            node_r = NETNode(0, [0, 1, 2, 3, 4, 5], [], z_kernel=(alphas, gammas))
            node_s = NETNode(1, [0], [0], z_kernel=(alphas, gammas))
            node_b1 = NETNode(2, [1, 2, 3, 4, 5], [1], z_kernel=(alphas, gammas))
            node_b2 = NETNode(3, [2, 3, 4], [2], z_kernel=(alphas, gammas))
            node_l3 = NETNode(4, [3], [3], z_kernel=(alphas, gammas))
            node_l4 = NETNode(5, [4], [4], z_kernel=(alphas, gammas))
            node_l5 = NETNode(6, [5], [5], z_kernel=(alphas, gammas))
            # add nodes to tree
            self.net = NET()
            self.net.setRoot(node_r)
            self.net.addNodeWithParent(node_s, node_r)
            self.net.addNodeWithParent(node_b1, node_r)
            self.net.addNodeWithParent(node_b2, node_b1)
            self.net.addNodeWithParent(node_l3, node_b2)
            self.net.addNodeWithParent(node_l4, node_b2)
            self.net.addNodeWithParent(node_l5, node_b1)

    def testNodeFunctionalities(self):
        self.loadTree()
        node_r = self.net.root
        node_b = self.net[3]
        node_l = self.net[5]
        # test contains
        assert 1 in node_r
        assert 1 not in node_b and 2 in node_b
        assert 1 not in node_l and 4 in node_l

    def testKernels(self):
        # kernel 1
        a1 = np.array([1., 10.])
        c1 = np.array([2., 20.])
        k1 = Kernel((a1, c1))
        # kernel 2
        a2 = np.array([1., 10.])
        c2 = np.array([4., 40.])
        k2 = Kernel({'a': a2, 'c': c2})
        # kernel 3
        k3 = Kernel(1.)
        # kbar
        assert np.abs(k1.k_bar - 4.) < 1e-12
        assert np.abs(k2.k_bar - 8.) < 1e-12
        assert np.abs(k3.k_bar - 1.) < 1e-12
        # temporal kernel
        t_arr = np.array([0., np.infty])
        assert np.allclose(k1(t_arr), np.array([22., 0.]))
        assert np.allclose(k2(t_arr), np.array([44., 0.]))
        assert np.allclose(k3(t_arr), np.array([1., 0.]))
        # frequency kernel
        s_arr = np.array([0. * 1j, np.infty * 1j])
        assert np.allclose(k1.ft(s_arr), np.array([4. + 0j, np.nan * 1j]), equal_nan=True)
        assert np.allclose(k2.ft(s_arr), np.array([8. + 0j, np.nan * 1j]), equal_nan=True)
        assert np.allclose(k3.ft(s_arr), np.array([1. + 0j, np.nan * 1j]), equal_nan=True)
        # test addition
        k4 = k1 + k2
        assert np.abs(k4.k_bar - 12.) < 1e-12
        assert len(k4.a) == 2
        k5 = k1 + k3
        assert np.abs(k5.k_bar - 5.) < 1e-12
        assert len(k5.a) == 3
        # test subtraction
        k6 = k2 - k1
        assert len(k6.a) == 2
        assert np.allclose(k6.c, np.array([2., 20]))
        assert np.abs(k6.k_bar - 4.) < 1e-12
        k7 = k1 - k3
        assert len(k7.a) == 3
        assert np.allclose(k7.c, np.array([2., 20., -1.]))
        assert np.abs(k7.k_bar - 3.) < 1e-12

    def testBasic(self):
        self.loadTree()
        net = self.net
        assert net.getLocInds() == [0, 1, 2, 3, 4, 5]
        assert net.getLocInds(3) == [2, 3, 4]
        assert net.getLeafLocNode(0).index == 1
        assert net.getLeafLocNode(1).index == 2
        assert net.getLeafLocNode(2).index == 3
        assert net.getLeafLocNode(3).index == 4
        assert net.getLeafLocNode(4).index == 5
        assert net.getLeafLocNode(5).index == 6
        # create reduced net
        net_reduced = net.getReducedTree([4, 5])
        node_r = net_reduced[0]
        node_4 = net_reduced.getLeafLocNode(4)
        node_5 = net_reduced.getLeafLocNode(5)
        assert node_r.z_bar == (net[0].z_kernel + net[2].z_kernel).k_bar
        assert node_4.z_bar == (net[3].z_kernel + net[5].z_kernel).k_bar
        assert node_5.z_bar == net[6].z_bar
        # test Iz
        Izs = net.calcIZ([1, 3, 5])
        assert np.abs(Izs[(1, 3)] - .5) < 1e-12
        assert np.abs(Izs[(1, 5)] - .25) < 1e-12
        assert np.abs(Izs[(3, 5)] - .75) < 1e-12
        with pytest.raises(KeyError):
            Izs[(5, 3)]
        assert isinstance(net.calcIZ([4, 5]), float)
        # test impedance matrix calculation
        z_mat_control = np.array([[4., 2.], [2., 3.]])
        assert np.allclose(net_reduced.calcImpMat(), z_mat_control)

    def testCompartmentalization(self):
        self.loadTree()
        net = self.net
        comps = net.getCompartmentalization(Iz=.1)
        assert comps == [[1], [4], [5], [6]]
        comps = net.getCompartmentalization(Iz=.5)
        assert comps == [[1], [2]]
        comps = net.getCompartmentalization(Iz=1.)
        assert comps == [[3]]
        comps = net.getCompartmentalization(Iz=3.)
        assert comps == []
        comps = net.getCompartmentalization(Iz=5.)
        assert comps == []

    def testPlotting(self, pshow=0):
        self.loadTree()
        pl.figure('dendrograms')
        ax = pl.subplot(221)
        self.net.plotDendrogram(ax)
        ax = pl.subplot(222)
        self.net.plotDendrogram(ax,
                                plotargs={'lw': 2., 'color': 'DarkGrey'},
                                labelargs={'marker': 'o', 'ms': 6., 'c': 'r'},
                                textargs={'size': 'small'})
        ax = pl.subplot(223)
        self.net.plotDendrogram(ax,
                                plotargs={'lw': 2., 'color': 'DarkGrey'},
                                labelargs={-1: {'marker': 'o', 'ms': 6., 'c': 'r'},
                                           2: {'marker': 'o', 'ms': 10., 'c': 'y'}},
                                textargs={'size': 'small'})
        ax = pl.subplot(224)
        self.net.plotDendrogram(ax,
                                plotargs={'lw': 2., 'color': 'DarkGrey'},
                                labelargs={-1: {'marker': 'o', 'ms': 6., 'c': 'r'},
                                           2: {'marker': 'o', 'ms': 10., 'c': 'y'}},
                                textargs={'size': 'xx-small'},
                                nodelabels=None,
                                cs_comp={1: 0.05, 4: 0.35, 5: 0.75, 6: 0.95})
        if pshow:
            pl.show()


if __name__ == '__main__':
    tnet = TestNET()
    tnet.testNodeFunctionalities()
    tnet.testKernels()
    tnet.testBasic()
    tnet.testCompartmentalization()
    tnet.testPlotting(pshow=1)
