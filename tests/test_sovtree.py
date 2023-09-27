import numpy as np
import matplotlib.pyplot as pl
import os

import pytest
from neat import SOVTree, SOVNode, Kernel, GreensTree
import neat.tools.kernelextraction as ke


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_morphologies'))


class TestSOVTree():
    def loadTTree(self):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Tsovtree.swc')
        self.tree = SOVTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(-75., 10.)
        self.tree.setCompTree()

    def testStringRepresentation(self):
        self.loadTTree()
        self.tree.calcSOVEquations()

        assert str(self.tree) == f">>> SOVTree\n"\
            "    SomaSOVNode 1, Parent: None\n" \
            "    SOVNode 4, Parent: 1\n" \
            "    SOVNode 5, Parent: 4\n" \
            "    SOVNode 6, Parent: 5\n" \
            "    SOVNode 7, Parent: 4\n" \
            "    SOVNode 8, Parent: 7"

        assert repr(self.tree) == "[" \
            "\"{'node index': 1, 'parent index': -1, 'content': '{}', 'xyz': array([0., 0., 0.]), 'R': 10.0, 'swc_type': 1, 'currents': {'L': [100.0, -75.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'e_eq': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 4, 'parent index': 1, 'content': '{}', 'xyz': array([100.,   0.,   0.]), 'R': 1.0, 'swc_type': 4, 'currents': {'L': [100.0, -75.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'e_eq': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 5, 'parent index': 4, 'content': '{}', 'xyz': array([100. ,  50.5,   0. ]), 'R': 1.0, 'swc_type': 4, 'currents': {'L': [100.0, -75.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'e_eq': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 6, 'parent index': 5, 'content': '{}', 'xyz': array([100., 101.,   0.]), 'R': 0.5, 'swc_type': 4, 'currents': {'L': [100.0, -75.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'e_eq': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 7, 'parent index': 4, 'content': '{}', 'xyz': array([100. , -49.5,   0. ]), 'R': 1.0, 'swc_type': 4, 'currents': {'L': [100.0, -75.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'e_eq': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 8, 'parent index': 7, 'content': '{}', 'xyz': array([100., -99.,   0.]), 'R': 0.5, 'swc_type': 4, 'currents': {'L': [100.0, -75.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'e_eq': -75.0, 'conc_eps': {}}\"" \
        "]{'channel_storage': [], 'maxspace_freq': 500.0}"

    def loadValidationTree(self):
        """
        Load the T-tree morphology in memory

        5---1---4
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'sovvalidationtree.swc')
        self.tree = SOVTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(-75., 10.)
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
        imp_a = self.tree.getModeImportance(locarg=locs_0)
        imp_b = self.tree.getModeImportance(locarg='0')
        imp_c = self.tree.getModeImportance(
                            sov_data=self.tree.getSOVMatrices(locarg=locs_0))
        imp_d = self.tree.getModeImportance(
                            sov_data=self.tree.getSOVMatrices(locarg='0'))
        assert np.allclose(imp_a, imp_b)
        assert np.allclose(imp_a, imp_c)
        assert np.allclose(imp_a, imp_d)
        assert np.abs(1. - np.max(imp_a)) < 1e-12
        with pytest.raises(IOError):
            self.tree.getModeImportance()
        # test important modes
        imp_2 = self.tree.getModeImportance(locarg='2')
        assert not np.allclose(imp_a, imp_2)
        # test impedance matrix
        z_mat_a = self.tree.calcImpedanceMatrix(
                        sov_data=self.tree.getImportantModes(locarg='1', eps=1e-10))
        z_mat_b = self.tree.calcImpedanceMatrix(locarg='1', eps=1e-10)
        assert np.allclose(z_mat_a, z_mat_b)
        assert np.allclose(z_mat_a - z_mat_a.T, np.zeros(z_mat_a.shape))
        for ii, z_row in enumerate(z_mat_a):
            assert np.argmax(z_row) == ii
        # test Fourrier impedance matrix
        ft = ke.FourrierTools(np.arange(0.,100.,0.1))
        z_mat_ft = self.tree.calcImpedanceMatrix(locarg='1', eps=1e-10, freqs=ft.s)
        assert np.allclose(z_mat_ft[ft.ind_0s,:,:].real, \
                           z_mat_a, atol=1e-1) # check steady state
        assert np.allclose(z_mat_ft - np.transpose(z_mat_ft, axes=(0,2,1)), \
                           np.zeros(z_mat_ft.shape)) # check symmetry
        assert np.allclose(z_mat_ft[:ft.ind_0s,:,:].real, \
                           z_mat_ft[ft.ind_0s+1:,:,:][::-1,:,:].real) # check real part even
        assert np.allclose(z_mat_ft[:ft.ind_0s,:,:].imag, \
                          -z_mat_ft[ft.ind_0s+1:,:,:][::-1,:,:].imag) # check imaginary part odd

    def loadBall(self):
        """
        Load point neuron model
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball.swc')
        self.btree = SOVTree(fname, types=[1,3,4])
        self.btree.fitLeakCurrent(-75., 10.)
        self.btree.setCompTree()

    def testSingleCompartment(self):
        self.loadBall()
        # for validation
        greenstree = self.btree.__copy__(new_tree=GreensTree())
        greenstree.setCompTree()
        greenstree.setImpedance(np.array([0.]))
        z_inp = greenstree.calcImpedanceMatrix([(1.,0.5)])

        self.btree.calcSOVEquations(maxspace_freq=500)
        alphas, gammas = self.btree.getSOVMatrices(locarg=[(1.,.5)])
        z_inp_sov = self.btree.calcImpedanceMatrix(locarg=[(1.,.5)])

        assert alphas.shape[0] == 1
        assert gammas.shape == (1,1)
        assert np.abs(1./np.abs(alphas[0]) - 10.) < 1e-10

        g_m = self.btree[1].getGTot(self.btree.channel_storage)
        g_s = g_m  * 4.*np.pi*(self.btree[1].R*1e-4)**2

        assert np.abs(gammas[0,0]**2/np.abs(alphas[0]) - 1./g_s) < 1e-10
        assert np.abs(z_inp_sov - 1./g_s) < 1e-10

    def testNETDerivation(self):
        # initialize
        self.loadValidationTree()
        self.tree.calcSOVEquations()
        # construct the NET
        net = self.tree.constructNET()
        # initialize
        self.loadTTree()
        self.tree.calcSOVEquations()
        # construct the NET
        net = self.tree.constructNET(dz=20.)
        # contruct the NET with linear terms
        net, lin_terms = self.tree.constructNET(dz=20., add_lin_terms=True)
        # check if correct
        alphas, gammas = self.tree.getImportantModes(locarg='net eval',
                                                eps=1e-4, sort_type='timescale')
        for ii, lin_term in lin_terms.items():
            z_k_trans = net.getReducedTree([0,ii]).getRoot().z_kernel + lin_term
            assert np.abs(z_k_trans.k_bar - Kernel((alphas, gammas[:,0]*gammas[:,ii])).k_bar) < 1e-8


if __name__ == '__main__':
    tsov = TestSOVTree()
    tsov.testStringRepresentation()
    # tsov.testSOVCalculation()
    # tsov.testSingleCompartment()
    # tsov.testNETDerivation()
