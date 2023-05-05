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
        # k_chan = channelcollection.Kv3_1()
        # self.tree.addCurrent(k_chan, 0.766*1e6, -85.)
        # na_chan = channelcollection.Na_Ta()
        # self.tree.addCurrent(na_chan, 1.71*1e6, 50.)
        k_chan = channelcollection.Kv3_1()
        self.tree.addCurrent(k_chan, 0.766*1e6, -85.)
        na_chan = channelcollection.Na_Ta()
        self.tree.addCurrent(na_chan, 1.5*1e6, 50.)
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

    def testNestNeuronComparison(self, pplot=False):
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
            res_neuron['v_m'][0][:-2] ,
            atol=1.
        )

        if pplot:
            pl.plot(res_neuron['t'], res_neuron['v_m'][0], 'rx-')
            pl.plot(res_nest['times'], res_nest['v_comp0'], 'bo--')
            pl.show()


if __name__ == "__main__":
    tn = TestNest()
    tn.testModelConstruction()
    tn.testNestNeuronComparison(pplot=True)