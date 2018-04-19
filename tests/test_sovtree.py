import numpy as np
import matplotlib.pyplot as pl

import pytest
from neat import SOVTree, SOVNode


# import morphologyReader as morphR

class TestSOVTree():
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

    def loadValidationTree(self):
        '''
        Load the T-tree morphology in memory

        5---1---4
        '''
        print '>>> loading validation tree <<<'
        fname = 'test_morphologies/sovvalidationtree.swc'
        self.tree = SOVTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(e_eq_target=-75., tau_m_target=10.)
        self.tree.setCompTree()

    def testSOVCalculation(self):
        # validate the calculation on analytical model
        self.loadValidationTree()
        # do SOV calculation
        self.tree.calcSOVEquations()
        alphas, gammas = self.tree.getSOVMatrices([(1, 0.5)])
        # compute time scales analytically
        self.tree.treetype = 'computational'
        lambda_m_test = np.sqrt(self.tree[4].R_sov / \
                        (2.*self.tree[4].g_m*self.tree[4].r_a))
        tau_m_test = self.tree[4].c_m / self.tree[4].g_m * 1e3
        alphas_test = \
            (1. + \
            (np.pi * np.arange(20) * lambda_m_test / \
            (self.tree[4].L_sov + self.tree[5].L_sov))**2) / \
            tau_m_test
        # compare analytical and computed time scales
        assert np.allclose(alphas[:20], alphas_test)
        # compute the spatial mode functions analytically
        # import matplotlib.pyplot as pl
        # self.tree.distributeLocsUniform(dx=4., name='NET_eval')
        # alphas, gammas = self.tree.getSOVMatrices(self.tree.getLocs(name='NET_eval'))
        # for kk in range(5):
        #     print 'tau_' + str(kk) + ' =', -1./alphas[kk].real
        #     pl.plot(range(gammas.shape[1]), gammas[kk,:])
        #     pl.plot(range(gammas.shape[1]), g)
        # pl.show()
        ## TODO

        # test basic identities
        self.loadTTree()
        self.tree.calcSOVEquations(maxspace_freq=500)
        # sets of location
        locs_0 = [(6, .5), (8, .5)]
        locs_1 = [(1, .5), (4, .5), (4, 1.), (5, .5), (6, .5), (7, .5), (8, .5)]
        locs_2 = [(7, .5), (8, .5)]
        self.tree.storeLocs(locs_0, '0')
        self.tree.storeLocs(locs_1, '1')
        self.tree.storeLocs(locs_2, '2')
        # test mode importance
        imp_a = self.tree.getModeImportance(locs=locs_0)
        imp_b = self.tree.getModeImportance(name='0')
        imp_c = self.tree.getModeImportance(
                            sov_data=self.tree.getSOVMatrices(locs=locs_0))
        imp_d = self.tree.getModeImportance(
                            sov_data=self.tree.getSOVMatrices(name='0'))
        assert np.allclose(imp_a, imp_b)
        assert np.allclose(imp_a, imp_c)
        assert np.allclose(imp_a, imp_d)
        assert np.abs(1. - np.max(imp_a)) < 1e-12
        with pytest.raises(IOError):
            self.tree.getModeImportance()
        # test important modes
        imp_2 = self.tree.getModeImportance(name='2')
        assert not np.allclose(imp_a, imp_2)
        # test impedance matrix
        z_mat_a = self.tree.calcImpedanceMatrix(
                        sov_data=self.tree.getImportantModes(name='1', eps=1e-10))
        z_mat_b = self.tree.calcImpedanceMatrix(name='1', eps=1e-10)
        assert np.allclose(z_mat_a, z_mat_b)
        assert np.allclose(z_mat_a - z_mat_a.T, np.zeros(z_mat_a.shape))
        for ii, z_row in enumerate(z_mat_a):
            assert np.argmax(z_row) == ii

        # import matplotlib.pyplot as pl
        # self.tree.distributeLocsUniform(dx=4., name='NET_eval')
        # print [str(loc) for loc in self.tree.getLocs(name='NET_eval')]
        # z_m = self.tree.calcImpedanceMatrix(name='NET_eval', eps=1e-10)
        # pl.imshow(z_m, origin='lower', interpolation='none')
        # alphas, gammas = self.tree.getSOVMatrices(self.tree.getLocs(name='NET_eval'))
        # for kk in range(5):
        #     print 'tau_' + str(kk) + ' =', -1./alphas[kk].real
        #     pl.plot(range(gammas.shape[1]), gammas[kk,:])
        # pl.show()

        # import morphologyReader as morphR


    def testNETDerivation(self):
        # initialize
        self.loadValidationTree()
        self.tree.calcSOVEquations()
        # construct the NET
        net = self.tree.constructNET()
        # print str(net)
        # initialize
        self.loadTTree()
        self.tree.calcSOVEquations()
        # construct the NET
        net = self.tree.constructNET(dg=20)
        # print str(net)



if __name__ == '__main__':
    tsov = TestSOVTree()
    tsov.testSOVCalculation()
    tsov.testNETDerivation()
