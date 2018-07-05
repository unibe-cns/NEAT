import numpy as np
import matplotlib.pyplot as pl

import pytest

from neat import SOVTree, GreensTree, GreensNode
import neat.tools.kernelextraction as ke


class TestGreensTree():
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
        self.tree = GreensTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(e_eq_target=-75., tau_m_target=10.)
        self.tree.setCompTree()

    def loadValidationTree(self):
        '''
        Load the T-tree morphology in memory

        5---1---4
        '''
        print '>>> loading validation tree <<<'
        fname = 'test_morphologies/sovvalidationtree.swc'
        self.tree = GreensTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(e_eq_target=-75., tau_m_target=10.)
        self.tree.setCompTree()

    def loadSOVTTree(self):
        '''
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        '''
        print '>>> loading T-tree <<<'
        fname = 'test_morphologies/Tsovtree.swc'
        self.sovtree = SOVTree(fname, types=[1,3,4])
        self.sovtree.fitLeakCurrent(e_eq_target=-75., tau_m_target=10.)
        self.sovtree.setCompTree()
        self.sovtree.calcSOVEquations()

    def loadSOVValidationTree(self):
        '''
        Load the T-tree morphology in memory

        5---1---4
        '''
        print '>>> loading validation tree <<<'
        fname = 'test_morphologies/sovvalidationtree.swc'
        self.sovtree = SOVTree(fname, types=[1,3,4])
        self.sovtree.fitLeakCurrent(e_eq_target=-75., tau_m_target=10.)
        self.sovtree.setCompTree()
        self.sovtree.calcSOVEquations()

    def testBasicProperties(self):
        self.loadTTree()
        # test Fourrier impedance matrix
        ft = ke.FourrierTools(np.arange(0.,100.,0.1))
        # set the impedances
        self.tree.setImpedance(ft.s)
        # sets of location
        locs_0 = [(6, .5), (8, .5)]
        locs_1 = [(1, .5), (4, .5), (4, 1.), (5, .5), (6, .5), (7, .5), (8, .5)]
        locs_2 = [(7, .5), (8, .5)]
        self.tree.storeLocs(locs_0, '0')
        self.tree.storeLocs(locs_1, '1')
        self.tree.storeLocs(locs_2, '2')
        # compute impedance matrices
        z_mat_0 = self.tree.calcImpedanceMatrix('0')[ft.ind_0s]
        z_mat_1 = self.tree.calcImpedanceMatrix('1')[ft.ind_0s]
        z_mat_2 = self.tree.calcImpedanceMatrix('2')[ft.ind_0s]
        # check complex steady state component zero
        assert np.allclose(z_mat_0.imag, np.zeros_like(z_mat_0.imag))
        assert np.allclose(z_mat_1.imag, np.zeros_like(z_mat_1.imag))
        assert np.allclose(z_mat_2.imag, np.zeros_like(z_mat_2.imag))
        # check symmetry
        assert np.allclose(z_mat_0, z_mat_0.T)
        assert np.allclose(z_mat_1, z_mat_1.T)
        assert np.allclose(z_mat_2, z_mat_2.T)
        # check symmetry directly
        assert np.allclose(self.tree.calcZF(locs_0[0], locs_0[1]),
                           self.tree.calcZF(locs_0[1], locs_0[0]))
        assert np.allclose(self.tree.calcZF(locs_1[0], locs_1[3]),
                           self.tree.calcZF(locs_1[3], locs_1[0]))
        assert np.allclose(self.tree.calcZF(locs_1[2], locs_1[5]),
                           self.tree.calcZF(locs_1[5], locs_1[2]))
        # check transitivity
        z_14_ = self.tree.calcZF(locs_1[1], locs_1[3]) * \
                self.tree.calcZF(locs_1[3], locs_1[4]) / \
                self.tree.calcZF(locs_1[3], locs_1[3])
        z_14 = self.tree.calcZF(locs_1[1], locs_1[4])
        assert np.allclose(z_14, z_14_)
        z_06_ = self.tree.calcZF(locs_1[0], locs_1[5]) * \
                self.tree.calcZF(locs_1[5], locs_1[6]) / \
                self.tree.calcZF(locs_1[5], locs_1[5])
        z_06 = self.tree.calcZF(locs_1[0], locs_1[6])
        assert np.allclose(z_06, z_06_)
        z_46_ = self.tree.calcZF(locs_1[4], locs_1[2]) * \
                self.tree.calcZF(locs_1[2], locs_1[6]) / \
                self.tree.calcZF(locs_1[2], locs_1[2])
        z_46 = self.tree.calcZF(locs_1[4], locs_1[6])
        assert np.allclose(z_46, z_46_)
        z_n15_ = self.tree.calcZF(locs_1[1], locs_1[3]) * \
                self.tree.calcZF(locs_1[3], locs_1[5]) / \
                self.tree.calcZF(locs_1[3], locs_1[3])
        z_15 = self.tree.calcZF(locs_1[1], locs_1[5])
        assert not np.allclose(z_15, z_n15_)

    def testValues(self):
        # load trees
        self.loadTTree()
        self.loadSOVTTree()
        # test Fourrier impedance matrix
        ft = ke.FourrierTools(np.arange(0.,100.,0.1))
        # set the impedances
        self.tree.setImpedance(ft.s)
        # sets of location
        # sets of location
        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (6, .5), (7, .5), (8, .5)]
        self.tree.storeLocs(locs, 'locs')
        self.sovtree.storeLocs(locs, 'locs')
        # compute impedance matrices with both methods
        z_sov = self.sovtree.calcImpedanceMatrix(name='locs', eps=1e-10)
        z_gf = self.tree.calcImpedanceMatrix('locs')[ft.ind_0s].real
        assert np.allclose(z_gf, z_sov, atol=5e-1)
        zf_sov = self.sovtree.calcImpedanceMatrix(name='locs', eps=1e-10, freqs=ft.s)
        zf_gf = self.tree.calcImpedanceMatrix('locs')
        assert np.allclose(zf_gf, zf_sov, atol=5e-1)

        # load trees
        self.loadValidationTree()
        self.loadSOVValidationTree()
        # test Fourrier impedance matrix
        ft = ke.FourrierTools(np.arange(0.,100.,0.1))
        # set the impedances
        self.tree.setImpedance(ft.s)
        # sets of location
        # sets of location
        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (5, 1.)]
        self.tree.storeLocs(locs, 'locs')
        self.sovtree.storeLocs(locs, 'locs')
        # compute impedance matrices with both methods
        z_sov = self.sovtree.calcImpedanceMatrix(name='locs', eps=1e-10)
        z_gf = self.tree.calcImpedanceMatrix('locs')[ft.ind_0s].real
        assert np.allclose(z_gf, z_sov, atol=5e-1)
        zf_sov = self.sovtree.calcImpedanceMatrix(name='locs', eps=1e-10, freqs=ft.s)
        zf_gf = self.tree.calcImpedanceMatrix('locs')
        assert np.allclose(zf_gf, zf_sov, atol=5e-1)

        # print z_sov
        # print z_gf

        # self.tree.treetype = 'computational'
        # for node in self.tree:
        #     print node
        #     if not self.tree.isRoot(node):
        #         print 'z_p', node.z_proximal[ft.ind_0s]
        #         print 'z_d', node.z_distal[ft.ind_0s]
        #         print 'z_c', node.z_c[ft.ind_0s]
        #         print 'z_a', node.z_a
        #         print 'z_m', node.z_m[ft.ind_0s]
        #         print 'gamma', node.gamma[ft.ind_0s]
        #     else:
        #         # pass
        #         print node.z_soma[ft.ind_0s]


        # import morphologyReader as morphR
        # distr = {'L': {'type': 'fit', 'param': [-65., 10.], 'E': -65.}}
        # fname = 'test_morphologies/Tsovtree.swc'
        # gt = morphR.greensTree(fname, ionc_distr=distr, soma_distr=distr, cnodesdistr='all')
        # # gt.set_changenodes()
        # gt.set_impedance(ft.s)

        # for node in gt.tree.get_nodes():
        #     if morphR.is_changenode(node):
        #         imp = node.get_content()['impedance']
        #         print node
        #         if node._index != 1:
        #             print 'z_p', imp.z_0[ft.ind_0s]
        #             print 'z_d', imp.z_1[ft.ind_0s]
        #             print 'z_c', imp.z_c[ft.ind_0s]
        #             print 'z_a', imp.z_a
        #             print 'z_m', imp.z_m[ft.ind_0s]
        #             print 'gamma', imp.gamma[ft.ind_0s]
        #         else:
        #             # pass
        #             print imp.z_soma[ft.ind_0s]




if __name__ == '__main__':
    tgt = TestGreensTree()
    # tgt.testBasicProperties()
    tgt.testValues()