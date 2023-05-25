import numpy as np
import matplotlib.pyplot as pl

import os
import itertools

import pytest

from neat import SOVTree, GreensTree, GreensTreeTime, NeuronSimTree, GreensNode
import neat.tools.kernelextraction as ke

import channelcollection_for_tests as channelcollection
import channel_installer
channel_installer.load_or_install_neuron_testchannels()

MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_morphologies'))


class TestGreensTree():
    def loadTTree(self):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Tsovtree.swc')
        self.tree = GreensTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(-75., 10.)
        self.tree.setCompTree()

    def loadValidationTree(self):
        """
        Load the T-tree morphology in memory

        5---1---4
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'sovvalidationtree.swc')
        self.tree = GreensTree(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(-75., 10.)
        self.tree.setCompTree()

    def loadSOVTTree(self):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Tsovtree.swc')
        self.sovtree = SOVTree(fname, types=[1,3,4])
        self.sovtree.fitLeakCurrent(-75., 10.)
        self.sovtree.setCompTree()
        self.sovtree.calcSOVEquations()

    def loadSOVValidationTree(self):
        """
        Load the T-tree morphology in memory

        5---1---4
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'sovvalidationtree.swc')
        self.sovtree = SOVTree(fname, types=[1,3,4])
        self.sovtree.fitLeakCurrent(-75., 10.)
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
        zf = self.tree.calcZF(*locs_0)
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
        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (6, .5), (7, .5), (8, .5)]
        self.tree.storeLocs(locs, 'locs')
        self.sovtree.storeLocs(locs, 'locs')
        # compute impedance matrices with both methods
        z_sov = self.sovtree.calcImpedanceMatrix(locarg='locs', eps=1e-10)
        z_gf = self.tree.calcImpedanceMatrix('locs')[ft.ind_0s].real
        assert np.allclose(z_gf, z_sov, atol=5e-1)
        z_gf2 = self.tree.calcImpedanceMatrix('locs', explicit_method=False)[ft.ind_0s].real
        assert np.allclose(z_gf2, z_gf, atol=5e-6)
        zf_sov = self.sovtree.calcImpedanceMatrix(locarg='locs', eps=1e-10, freqs=ft.s)
        zf_gf = self.tree.calcImpedanceMatrix('locs')
        assert np.allclose(zf_gf, zf_sov, atol=5e-1)
        zf_gf2 = self.tree.calcImpedanceMatrix('locs', explicit_method=False)
        assert np.allclose(zf_gf2, zf_gf, atol=5e-6)

        # load trees
        self.loadValidationTree()
        self.loadSOVValidationTree()
        # test Fourrier impedance matrix
        ft = ke.FourrierTools(np.arange(0.,100.,0.1))
        # set the impedances
        self.tree.setImpedance(ft.s)
        # set of locations
        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (5, 1.)]
        self.tree.storeLocs(locs, 'locs')
        self.sovtree.storeLocs(locs, 'locs')
        # compute impedance matrices with both methods
        z_sov = self.sovtree.calcImpedanceMatrix(locarg='locs', eps=1e-10)
        z_gf = self.tree.calcImpedanceMatrix('locs')[ft.ind_0s].real
        assert np.allclose(z_gf, z_sov, atol=5e-1)
        z_gf2 = self.tree.calcImpedanceMatrix('locs', explicit_method=False)[ft.ind_0s].real
        assert np.allclose(z_gf2, z_gf, atol=5e-6)
        zf_sov = self.sovtree.calcImpedanceMatrix(locarg='locs', eps=1e-10, freqs=ft.s)
        zf_gf = self.tree.calcImpedanceMatrix('locs')
        assert np.allclose(zf_gf, zf_sov, atol=5e-1)
        zf_gf2 = self.tree.calcImpedanceMatrix('locs', explicit_method=False)
        assert np.allclose(zf_gf2, zf_gf, atol=5e-6)


class TestGreensTreeTime():
    def __init__(self):
        self.dt = 0.025
        self.tmax = 100.
        # for frequency derivation
        self.ft = ke.FourrierTools(np.arange(0., self.tmax, self.dt))

    def loadTTree(self):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Tsovtree.swc')
        self.tree = GreensTreeTime(fname, types=[1,3,4])
        self.tree.fitLeakCurrent(-75., 10.)
        # set equilibirum potententials
        self.tree.setEEq(-75.)
        self.tree.setCompTree()

    def loadAxonTree(self):
        '''
        Parameters taken from a BBP SST model for a subset of ion channels
        '''
        self.tree = GreensTreeTime(
            file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball_and_axon.swc'),
            types=[1,2,3,4],
        )
        # capacitance and axial resistance
        self.tree.setPhysiology(1.0, 100./1e6)
        # ion channels
        k_chan = channelcollection.SKv3_1()
        self.tree.addCurrent(k_chan,  0.653374 * 1e6, -85., node_arg=[self.tree[1]])
        self.tree.addCurrent(k_chan,  0.196957 * 1e6, -85., node_arg="axonal")
        na_chan = channelcollection.NaTa_t()
        self.tree.addCurrent(na_chan, 0.001 * 1e6, 50., node_arg=[self.tree[1]])
        self.tree.addCurrent(na_chan, 3.418459 * 1e6, 50., node_arg="axonal")
        ca_chan = channelcollection.Ca_HVA()
        self.tree.addCurrent(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[self.tree[1]])
        self.tree.addCurrent(ca_chan, 0.000138 * 1e6, 132.4579341637009, node_arg="axonal")
        self.tree.fitLeakCurrent(-75., 10.)
        # set equilibirum potententials
        self.tree.setEEq(-75.)
        self.tree.setCompTree()

    def loadBall(self):
        '''
        Load point neuron model
        '''
        self.tree = GreensTreeTime(file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball.swc'))
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

    def testPassiveKernels(self, pplot=False):
        self.loadTTree()
        greens_tree = self.tree.__copy__(new_tree=GreensTree())
        sim_tree = self.tree.__copy__(new_tree=NeuronSimTree())

        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (6, .5), (7, .5), (8, .5)]
        self.tree.setImpedance(self.ft)

        zt_mat_gtt = self.tree.calcImpulseResponseMatrix(locs)

        # compute impedance matrix with Green's function
        greens_tree.setImpedance(self.ft.s)
        zf_mat_gtf = greens_tree.calcImpedanceMatrix(locs)
        # convert impedance matrix to time domain
        zt_mat_expl = np.zeros((len(self.ft.t), len(locs), len(locs)))
        for (ii, jj) in itertools.product(list(range(len(locs))), list(range(len(locs)))):
            zt_mat_expl[:,ii,jj] = self.ft.ftInv(zf_mat_gtf[:,ii,jj])[1].real * 1e-3
        # simulate the temporal matrix
        tk, zt_mat_sim = sim_tree.calcImpedanceKernelMatrix(locs)

        assert np.allclose(
            zt_mat_expl[int(1.5/self.dt):,:,:],
            zt_mat_gtt[int(1.5/self.dt):,:,:],
            atol=.25
        )
        assert np.allclose(
            zt_mat_sim[int(1.5/self.dt):,:,:],
            zt_mat_gtt[int(1.5/self.dt):,:,:],
            atol=.15
        )

        if pplot:
            colours = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
            pl.figure("zt", figsize=(12,4))
            ax0 = pl.subplot(131)
            ax1 = pl.subplot(132)

            kk = 0
            for (ii, loc1), (jj, loc2) in itertools.product(enumerate(locs), enumerate(locs)):
                ax = ax0 if ii == jj else ax1
                ax.plot(self.ft.t, zt_mat_gtt[:,ii,jj], c=colours[kk%len(colours)], ls="-", lw=.7)
                ax.plot(self.ft.t, zt_mat_expl[:,ii,jj], c=colours[kk%len(colours)], ls="--", lw=2)
                ax.plot(self.ft.t, zt_mat_sim[:,ii,jj], c=colours[kk%len(colours)], ls=":", lw=3)
                kk += 1

        # test derivatives
        zt_mat, dz_dt_mat = self.tree.calcImpulseResponseMatrix(locs,
            compute_time_derivative=True
        )
        dz_dt_second_order = (zt_mat[2:] - zt_mat[:-2]) / (2. * self.dt)

        print(np.max(np.abs(
            dz_dt_mat[int(1.5/self.dt+1):-1,:,:] - \
            dz_dt_second_order[int(1.5/self.dt):,:,:]
        )))

        assert np.allclose(
            dz_dt_mat[int(1.5/self.dt+1):-1,:,:],
            dz_dt_second_order[int(1.5/self.dt):,:,:],
            atol=.15
        )

        if pplot:
            ax2 = pl.subplot(133)

            kk = 0
            for (ii, loc1), (jj, loc2) in itertools.product(enumerate(locs), enumerate(locs)):
                ax2.plot(self.ft.t[1:-1], dz_dt_second_order[:,ii,jj], c=colours[kk%len(colours)], ls="-", lw=.7)
                ax2.plot(self.ft.t[1:-1], dz_dt_mat[1:-1,ii,jj], c=colours[kk%len(colours)], ls="--", lw=2)
                kk += 1

            pl.show()


    def testActiveKernels(self, pplot=True):
        self.loadAxonTree()
        greens_tree = self.tree.__copy__(new_tree=GreensTree())
        sim_tree = self.tree.__copy__(new_tree=NeuronSimTree())

        locs = [(1, .5), (5, .95)]
        self.tree.setImpedance(self.ft)

        zt_mat_gtt = self.tree.calcImpulseResponseMatrix(locs)

        # compute impedance matrix with Green's function
        greens_tree.setImpedance(self.ft.s)
        zf_mat_gtf = greens_tree.calcImpedanceMatrix(locs)
        # convert impedance matrix to time domain
        zt_mat_expl = np.zeros((len(self.ft.t), len(locs), len(locs)))
        for (ii, jj) in itertools.product(list(range(len(locs))), list(range(len(locs)))):
            zt_mat_expl[:,ii,jj] = self.ft.ftInv(zf_mat_gtf[:,ii,jj])[1].real * 1e-3
        # simulate the temporal matrix
        tk, zt_mat_sim = sim_tree.calcImpedanceKernelMatrix(locs)

        assert np.allclose(
            zt_mat_expl[int(1.5/self.dt):,:,:],
            zt_mat_gtt[int(1.5/self.dt):,:,:],
            atol=.9
        )
        assert np.allclose(
            zt_mat_sim[int(1.5/self.dt):,:,:],
            zt_mat_gtt[int(1.5/self.dt):,:,:],
            atol=.5
        )

        if pplot:
            colours = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
            pl.figure("zt_inp")
            ax0 = pl.gca()
            pl.figure("zt_trans")
            ax1 = pl.gca()

            kk = 0
            for (ii, loc1), (jj, loc2) in itertools.product(enumerate(locs), enumerate(locs)):
                ax = ax0 if ii == jj else ax1
                ax.plot(self.ft.t, zt_mat_gtt[:,ii,jj], c=colours[kk%len(colours)], ls="-", lw=.7)
                ax.plot(self.ft.t, zt_mat_expl[:,ii,jj], c=colours[kk%len(colours)], ls="--", lw=2)
                ax.plot(self.ft.t, zt_mat_sim[:,ii,jj], c=colours[kk%len(colours)], ls=":", lw=3)
                kk += 1

            pl.show()

    def testChannelResponses(self, pplot=True):
        self.loadAxonTree()
        self.tree.setImpedance(self.ft)
        sim_tree = self.tree.__copy__(new_tree=NeuronSimTree())

        locs = [(1, .5), (5, .5)]
        idxs_out = [0,1]
        idx_in = 1

        crt_01 = [
            self.tree.calcChannelResponseT(locs[idx_in], locs[idx_out], compute_time_derivative=0) \
            for idx_out in idxs_out
        ]
        crt_mat = self.tree.calcChannelResponseMatrix(locs)

        # simulate
        i_amp = 0.001 # nA
        dt_pulse = 0.1 # ms
        delay_pulse = 10. # ms
        tmax = self.tmax + delay_pulse
        sim_tree.initModel(dt=self.dt, t_calibrate=10., factor_lambda=100)
        sim_tree.addIClamp(locs[idx_in], i_amp, delay_pulse, dt_pulse)
        sim_tree.storeLocs(locs, 'rec locs', warn=False)
        # simulate
        res = sim_tree.run(tmax, record_from_channels=True)
        sim_tree.deleteModel()

        # compute state variable deflections
        for channel_name in self.tree.getChannelsInTree():

            if pplot:
                pl.figure(channel_name)
                axes = [pl.subplot(121), pl.subplot(122)]
                axes[0].set_title("rec loc 0")
                axes[1].set_title("rec loc 1")

            for svar_name in list(crt_01[0][channel_name].keys()):
                for idx_out in idxs_out:
                    q_sim = res['chan'][channel_name][svar_name][idx_out]
                    # q_sim = q_sim[int((delay_pulse+dt_pulse) / self.dt):] - q_sim[0]
                    q_sim = q_sim - q_sim[0]
                    q_sim /= (i_amp * dt_pulse)
                    t_sim = res['t']#[int((delay_pulse+dt_pulse) / self.dt):] - (delay_pulse+dt_pulse)

                    q_calc = crt_mat[idx_out][channel_name][svar_name][:,idx_in]

                    print(len(t_sim), len(self.ft.t))

                    q_calc_ = crt_01[idx_out][channel_name][svar_name]
                    assert np.allclose(q_calc, q_calc_)

                    if pplot:
                        axes[idx_out].plot(self.ft.t, q_calc, c='r', ls='-', lw=1.)
                        axes[idx_out].plot(self.ft.t, q_calc_, c='b', ls=':', lw=3.)
                        axes[idx_out].plot(t_sim, q_sim, c='k', ls='--', lw=2.)

        if pplot:
            pl.show()


    def testExponentialDerivative(self, pplot=True): pass


if __name__ == '__main__':
    # tgt = TestGreensTree()
    # tgt.testBasicProperties()
    # tgt.testValues()

    tgtt = TestGreensTreeTime()
    # tgtt.testPassiveKernels(pplot=True)
    # tgtt.testActiveKernels(pplot=True)
    tgtt.testChannelResponses()

