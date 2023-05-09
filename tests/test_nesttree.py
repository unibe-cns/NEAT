import numpy as np
import matplotlib.pyplot as pl

import os
import pytest

try:
    import nest
    import nest.lib.hl_api_exceptions as nestexceptions
except ImportError as e:
    pytest.skip("NEST not installed", allow_module_level=True)

from neat import PhysTree
from neat import CompartmentNode, CompartmentTree
from neat import CompartmentFitter, createReducedNeuronModel
from neat import NestCompartmentNode, NestCompartmentTree, loadNestModel

import channelcollection_for_tests as channelcollection
import channel_installer
channel_installer.load_or_install_neuron_testchannels()
channel_installer.load_or_install_nest_testchannels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    'test_morphologies'
))


class TestNest:
    def loadTwoCompartmentModel(self):
        # simple two compartment model
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(root=pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.addNodeWithParent(cnode, pnode)

        for ii, cn in enumerate(self.ctree):
            cn.loc_ind = ii

    def testModelConstruction(self):
        loadNestModel("default")
        with pytest.raises(nestexceptions.NESTErrors.DynamicModuleManagementError):
            loadNestModel("default")

        self.loadTwoCompartmentModel()

        nct = self.ctree.__copy__(new_tree=NestCompartmentTree())
        cm_model = nct.initModel("default", 1)

        compartments_info = cm_model.compartments
        assert compartments_info[0]["comp_idx"] == 0
        assert compartments_info[0]["parent_idx"] == -1
        assert compartments_info[1]["comp_idx"] == 1
        assert compartments_info[1]["parent_idx"] == 0

    def loadBall(self):
        '''
        Load point neuron model
        '''
        self.tree = PhysTree(file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball.swc'))
        # capacitance and axial resistance
        self.tree.setPhysiology(0.8, 100./1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()
        self.tree.addCurrent(k_chan, 0.766*1e6, -85.)
        na_chan = channelcollection.Na_Ta()
        self.tree.addCurrent(na_chan, 1.71*1e6, 50.)
        # k_chan = channelcollection.Kv3_1()
        # self.tree.addCurrent(k_chan, 0.766*1e6, -85.)
        # na_chan = channelcollection.Na_Ta()
        # self.tree.addCurrent(na_chan, 1.5*1e6, 50.)
        # fit leak current
        self.tree.fitLeakCurrent(-75., 10.)
        # set equilibirum potententials
        self.tree.setEEq(-75.)
        # set computational tree
        self.tree.setCompTree()

        cfit = CompartmentFitter(self.tree,
            save_cache=False, recompute_cache=True
        )
        self.ctree = cfit.fitModel([(1,0.5)])

    def testSingleCompNestNeuronComparison(self, pplot=False):
        dt = .001
        nest.ResetKernel()
        nest.SetKernelStatus(dict(resolution=dt))

        self.loadBall()

        csimtree_neuron = createReducedNeuronModel(self.ctree)
        csimtree_neuron.initModel(dt=dt, t_calibrate=200.)
        csimtree_neuron.storeLocs([(0, .5)], name='rec locs')
        csimtree_neuron.addDoubleExpSynapse((0,.5), .2, 3., 0.)
        csimtree_neuron.setSpikeTrain(0, 0.001, [20., 23., 40.])
        res_neuron = csimtree_neuron.run(200.)

        csimtree_nest = self.ctree.__copy__(new_tree=NestCompartmentTree())
        nestmodel = csimtree_nest.initModel("multichannel_test", 1)
        # inputs
        nestmodel.receptors = [{
            "comp_idx": 0,
            "receptor_type": "AMPA",
            "params": {"e_AMPA": 0., "tau_r_AMPA": .2, "tau_d_AMPA": 3.},
        }]
        sg = nest.Create('spike_generator', 1, {'spike_times': [220., 223., 240.]})
        nest.Connect(sg, nestmodel,
            syn_spec={
                'synapse_model': 'static_synapse',
                'weight': 0.001,
                'delay': 3*dt,
                'receptor_type': 0,
            }
        )
        # voltage recording
        mm = nest.Create('multimeter', 1,
            {'record_from': ["v_comp0"], 'interval': dt}
        )
        nest.Connect(mm, nestmodel)
        # simulate
        nest.Simulate(400.)
        res_nest = nest.GetStatus(mm, 'events')[0]

        idx0 = int(200./dt)
        res_nest['times'] = res_nest['times'][idx0:] - res_nest['times'][idx0]
        res_nest['v_comp0'] = res_nest['v_comp0'][idx0:]
        v0 = res_nest['v_comp0'][0]

        assert np.allclose(
            res_nest['v_comp0'],
            res_neuron['v_m'][0][:-2],
            atol=1.
        )

        if pplot:
            pl.plot(res_neuron['t'], res_neuron['v_m'][0], 'rx-')
            pl.plot(res_nest['times'], res_nest['v_comp0'], 'bo--')
            pl.show()

    def loadAxonTree(self):
        '''
        Parameters taken from a BBP SST model for a subset of ion channels
        '''
        tree = PhysTree(
            file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball_and_axon.swc'),
            types=[1,2,3,4],
        )
        # capacitance and axial resistance
        tree.setPhysiology(1.0, 100./1e6)
        # ion channels
        k_chan = channelcollection.SKv3_1()
        tree.addCurrent(k_chan,  0.653374 * 1e6, -85., node_arg=[tree[1]])
        tree.addCurrent(k_chan,  0.196957 * 1e6, -85., node_arg="axonal")
        na_chan = channelcollection.NaTa_t()
        tree.addCurrent(na_chan, 3.418459 * 1e6, 50., node_arg="axonal")
        # ca_chan = channelcollection.Ca_HVA()
        # tree.addCurrent(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        # tree.addCurrent(ca_chan, 0.000138 * 1e6, 132.4579341637009, node_arg="axonal")
        # sk_chan = channelcollection.SK_E2()
        # tree.addCurrent(sk_chan, 0.653374 * 1e6, -85., node_arg=[tree[1]])
        # tree.addCurrent(sk_chan, 0.196957 * 1e6, -85., node_arg="axonal")
        # passive leak current
        tree.setLeakCurrent(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])
        tree.setLeakCurrent(0.000094 * 1e6, -79.315740, node_arg="axonal")

        # simplify
        locs = [(1,.5), (4.,0.5), (5,0.5)]
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        self.ctree = cfit.fitModel(locs)

    def testAxonNestNeuronComparison(self, pplot=False):
        dt = .001
        nest.ResetKernel()
        nest.SetKernelStatus(dict(resolution=dt))

        self.loadAxonTree()

        csimtree_neuron = createReducedNeuronModel(self.ctree)
        csimtree_neuron.initModel(dt=dt, t_calibrate=200.)
        csimtree_neuron.storeLocs([(0, .5), (1, .5), (2., .5)], name='rec locs')
        csimtree_neuron.addDoubleExpSynapse((0,.5), .2, 3., 0.)
        csimtree_neuron.setSpikeTrain(0, 0.001, [20., 23., 40.])
        res_neuron = csimtree_neuron.run(200.)

        csimtree_nest = self.ctree.__copy__(new_tree=NestCompartmentTree())
        nestmodel = csimtree_nest.initModel("multichannel_test", 1)
        # inputs
        nestmodel.receptors = [{
            "comp_idx": 0,
            "receptor_type": "AMPA",
            "params": {"e_AMPA": 0., "tau_r_AMPA": .2, "tau_d_AMPA": 3.},
        }]
        sg = nest.Create('spike_generator', 1, {'spike_times': [220., 223., 240.]})
        nest.Connect(sg, nestmodel,
            syn_spec={
                'synapse_model': 'static_synapse',
                'weight': 0.001,
                'delay': dt,
                'receptor_type': 0,
            }
        )
        # voltage recording
        mm = nest.Create('multimeter', 1,
            {'record_from': ["v_comp0", "v_comp1", "v_comp2"], 'interval': dt}
        )
        nest.Connect(mm, nestmodel)
        # simulate
        nest.Simulate(400.)
        res_nest = nest.GetStatus(mm, 'events')[0]

        idx0 = int(200./dt)
        res_nest['times'] = res_nest['times'][idx0:] - res_nest['times'][idx0]
        res_nest['v_comp0'] = res_nest['v_comp0'][idx0:]
        res_nest['v_comp1'] = res_nest['v_comp1'][idx0:]
        res_nest['v_comp2'] = res_nest['v_comp2'][idx0:]
        v0 = res_nest['v_comp0'][0]

        # assert np.allclose(
        #     res_nest['v_comp0'],
        #     res_neuron['v_m'][0][2:],
        #     atol=1.
        # )

        if pplot:
            pl.figure(figsize=(15,6))
            ax = pl.subplot(131)
            ax.plot(res_neuron['t'], res_neuron['v_m'][0], 'rx-')
            ax.plot(res_nest['times'], res_nest['v_comp0'], 'bo--')
            ax = pl.subplot(132)
            ax.plot(res_neuron['t'], res_neuron['v_m'][1], 'rx-')
            ax.plot(res_nest['times'], res_nest['v_comp1'], 'bo--')
            ax = pl.subplot(133)
            ax.plot(res_neuron['t'], res_neuron['v_m'][2], 'rx-')
            ax.plot(res_nest['times'], res_nest['v_comp2'], 'bo--')
            pl.show()

    def loadTTree(self):
        '''
        Parameters taken from a BBP SST model for a subset of ion channels
        '''
        tree = PhysTree(
            file_n=os.path.join(MORPHOLOGIES_PATH_PREFIX, 'Ttree_segments.swc'),
            types=[1,2,3,4],
        )
        # capacitance and axial resistance
        tree.setPhysiology(1.0, 100./1e6)
        # ion channels
        k_chan = channelcollection.SKv3_1()
        tree.addCurrent(k_chan,  0.653374 * 1e6, -85., node_arg=[tree[1]])
        na_chan = channelcollection.NaTa_t()
        tree.addCurrent(na_chan, 3.418459 * 1e6, 50., node_arg=[tree[1]])
        ca_chan = channelcollection.Ca_HVA()
        tree.addCurrent(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        # passive leak current
        tree.fitLeakCurrent(-70., 15.)

        # simplify
        locs = [(n.index, .5) for n in tree]
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        self.ctree = cfit.fitModel(locs)

    def testDendNestNeuronComparison(self, pplot=False):
        dt = .1
        t0 = 0.
        t1 = 200.
        idx0 = int(t0/dt)
        nest.ResetKernel()
        nest.SetKernelStatus(dict(resolution=dt))

        self.loadTTree()

        rec_idx = 7
        dend_idx = 9

        clocs = [(ii, .5) for ii in range(len(self.ctree))]
        print(clocs)
        print(self.ctree)
        csimtree_neuron = createReducedNeuronModel(self.ctree)
        csimtree_neuron.initModel(dt=dt, t_calibrate=0.)
        csimtree_neuron.storeLocs(clocs, name='rec locs')
        csimtree_neuron.addDoubleExpSynapse(clocs[0], .2, 3., 0.)
        csimtree_neuron.setSpikeTrain(0, 0.005, [t1 + 20., t1 + 23., t1 + 40.])
        csimtree_neuron.addDoubleExpSynapse(clocs[dend_idx], .2, 3., 0.)
        csimtree_neuron.setSpikeTrain(1, 0.005, [t1 + 70., t1 + 74., t1 + 85.])
        res_neuron = csimtree_neuron.run(400, record_from_channels=True)

        print(res_neuron['chan'].keys())

        res_neuron['t'] = res_neuron['t'][idx0:] - res_neuron['t'][idx0]
        res_neuron['v_m'] = res_neuron['v_m'][:,idx0:]

        print(res_neuron['v_m'].shape)

        csimtree_nest = self.ctree.__copy__(new_tree=NestCompartmentTree())
        nestmodel = csimtree_nest.initModel("multichannel_test", 1)
        nestmodel.V_init = -75.
        # inputs
        nestmodel.receptors = [{
            "comp_idx": 0,
            "receptor_type": "AMPA",
            "params": {"e_AMPA": 0., "tau_r_AMPA": .2, "tau_d_AMPA": 3.},
        },
        {
            "comp_idx": dend_idx,
            "receptor_type": "AMPA",
            "params": {"e_AMPA": 0., "tau_r_AMPA": .2, "tau_d_AMPA": 3.},
        }]
        sg = nest.Create('spike_generator', 1, {'spike_times': [t1 + 20., t1 + 23., t1 + 40.]})
        nest.Connect(sg, nestmodel,
            syn_spec={
                'synapse_model': 'static_synapse',
                'weight': 0.005,
                'delay': dt,
                'receptor_type': 0,
            }
        )
        sg_ = nest.Create('spike_generator', 1, {'spike_times': [t1 + 70., t1 + 74., t1 + 85.]})
        nest.Connect(sg_, nestmodel,
            syn_spec={
                'synapse_model': 'static_synapse',
                'weight': 0.005,
                'delay': dt,
                'receptor_type': 1,
            }
        )
        # voltage recording
        mm = nest.Create('multimeter', 1, {
            'record_from': [f"v_comp{ii}" for ii in range(len(self.ctree))] + \
                           [f"h_Ca_HVA{ii}"for ii in range(len(self.ctree))] + \
                           [f"m_Ca_HVA{ii}"for ii in range(len(self.ctree))] + \
                           [f"h_NaTa_t{ii}"for ii in range(len(self.ctree))] + \
                           [f"m_NaTa_t{ii}"for ii in range(len(self.ctree))],
            'interval': dt
        })
        nest.Connect(mm, nestmodel)
        # simulate
        nest.Simulate(400.)
        res_nest = nest.GetStatus(mm, 'events')[0]

        print("!!!", len(res_nest[f'v_comp{0}']))

        res_nest['times'] = res_nest['times'][idx0:] - res_nest['times'][idx0]
        for ii in range(len(self.ctree)):
            res_nest[f'v_comp{ii}'] = res_nest[f'v_comp{ii}'][idx0:]

        print("!!!", len(res_nest[f'v_comp{0}']))

        for ii in range(len(self.ctree)):
            v0 = res_nest[f'v_comp{ii}'][0]
            v_maxdiff = np.max(np.abs(res_nest[f'v_comp{ii}'] - res_neuron['v_m'][ii,2:]))
            v_meandiff = np.mean(np.abs(res_nest[f'v_comp{ii}'] - res_neuron['v_m'][ii,2:]))
            print(v_maxdiff, "    ", v_meandiff)
            # assert v_maxdiff < 3. and v_meandiff < 0.004

        if pplot:
            pl.figure('v', figsize=(15,6))
            ax = pl.subplot(131)
            ax.plot(res_neuron['t'], res_neuron['v_m'][0], 'r-')
            ax.plot(res_nest['times'], res_nest['v_comp0'], 'b--')
            ax = pl.subplot(132)
            ax.plot(res_neuron['t'], res_neuron['v_m'][rec_idx], 'r-')
            ax.plot(res_nest['times'], res_nest[f'v_comp{rec_idx}'], 'b--')
            ax = pl.subplot(133)
            ax.plot(res_neuron['t'], res_neuron['v_m'][dend_idx], 'r-')
            ax.plot(res_nest['times'], res_nest[f'v_comp{dend_idx}'], 'b--')

            pl.figure('Ca_HVA', figsize=(15,6))
            ax = pl.subplot(131)
            ax.plot(res_neuron['t'], res_neuron['chan']['Ca_HVA']['m'][0], 'r-')
            ax.plot(res_neuron['t'], res_neuron['chan']['Ca_HVA']['h'][0], 'g-')
            ax.plot(res_nest['times'], res_nest[f'm_Ca_HVA0'], 'r--', lw=2)
            ax.plot(res_nest['times'], res_nest[f'h_Ca_HVA0'], 'g--', lw=2)
            ax = pl.subplot(132)
            ax.plot(res_neuron['t'], res_neuron['chan']['Ca_HVA']['m'][rec_idx], 'r-')
            ax.plot(res_neuron['t'], res_neuron['chan']['Ca_HVA']['h'][rec_idx], 'g-')
            ax.plot(res_nest['times'], res_nest[f'm_Ca_HVA{rec_idx}'], 'r--', lw=2)
            ax.plot(res_nest['times'], res_nest[f'h_Ca_HVA{rec_idx}'], 'g--', lw=2)
            ax = pl.subplot(133)
            ax.plot(res_neuron['t'], res_neuron['chan']['Ca_HVA']['m'][dend_idx], 'r-')
            ax.plot(res_neuron['t'], res_neuron['chan']['Ca_HVA']['h'][dend_idx], 'g-')
            ax.plot(res_nest['times'], res_nest[f'm_Ca_HVA{dend_idx}'], 'r--', lw=2)
            ax.plot(res_nest['times'], res_nest[f'h_Ca_HVA{dend_idx}'], 'g--', lw=2)

            pl.figure('NaTa_t', figsize=(15,6))
            ax = pl.subplot(131)
            ax.plot(res_neuron['t'], res_neuron['chan']['NaTa_t']['m'][0], 'r-')
            ax.plot(res_neuron['t'], res_neuron['chan']['NaTa_t']['h'][0], 'g-')
            ax.plot(res_nest['times'], res_nest[f'm_NaTa_t0'], 'r--', lw=2)
            ax.plot(res_nest['times'], res_nest[f'h_NaTa_t0'], 'g--', lw=2)
            ax = pl.subplot(132)
            ax.plot(res_neuron['t'], res_neuron['chan']['NaTa_t']['m'][rec_idx], 'r-')
            ax.plot(res_neuron['t'], res_neuron['chan']['NaTa_t']['h'][rec_idx], 'g-')
            ax.plot(res_nest['times'], res_nest[f'm_NaTa_t{rec_idx}'], 'r--', lw=2)
            ax.plot(res_nest['times'], res_nest[f'h_NaTa_t{rec_idx}'], 'g--', lw=2)
            ax = pl.subplot(133)
            ax.plot(res_neuron['t'], res_neuron['chan']['NaTa_t']['m'][dend_idx], 'r-')
            ax.plot(res_neuron['t'], res_neuron['chan']['NaTa_t']['h'][dend_idx], 'g-')
            ax.plot(res_nest['times'], res_nest[f'm_NaTa_t{dend_idx}'], 'r--', lw=2)
            ax.plot(res_nest['times'], res_nest[f'h_NaTa_t{dend_idx}'], 'g--', lw=2)
            pl.show()


if __name__ == "__main__":
    tn = TestNest()
    # tn.testModelConstruction()
    # tn.testSingleCompNestNeuronComparison(pplot=True)
    # tn.testMultiCompNestNeuronComparison(pplot=True)
    tn.testDendNestNeuronComparison(pplot=True)

    # ca_act()
