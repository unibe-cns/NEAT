import numpy as np
import matplotlib.pyplot as pl
import os

import pytest
import copy

from neat import PhysTree, PhysNode

import channelcollection_for_tests as channelcollection


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    'test_morphologies'
))


class TestPhysTree():
    def loadTree(self, reinitialize=0, segments=False):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        if not hasattr(self, 'tree') or reinitialize:
            fname = 'Ttree_segments.swc' if segments else 'Ttree.swc'
            self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, fname), types=[1,3,4])

    def testStringRepresentation(self):
        self.loadTree()

        # gmax as potential as float
        e_rev = 100.
        g_max = 100.
        channel = channelcollection.TestChannel2()
        self.tree.addCurrent(channel, g_max, e_rev)

        assert str(self.tree) == ">>> PhysTree\n" \
            "    PhysNode 1, Parent: None --- r_a = 0.0001 MOhm*cm, c_m = 1.0 uF/cm^2, v_ep = -75.0 mV, (g_TestChannel2 = 100.0 uS/cm^2, e_TestChannel2 = 100.0 mV)\n" \
            "    PhysNode 4, Parent: 1 --- r_a = 0.0001 MOhm*cm, c_m = 1.0 uF/cm^2, v_ep = -75.0 mV, (g_TestChannel2 = 100.0 uS/cm^2, e_TestChannel2 = 100.0 mV)\n" \
            "    PhysNode 5, Parent: 4 --- r_a = 0.0001 MOhm*cm, c_m = 1.0 uF/cm^2, v_ep = -75.0 mV, (g_TestChannel2 = 100.0 uS/cm^2, e_TestChannel2 = 100.0 mV)\n" \
            "    PhysNode 6, Parent: 5 --- r_a = 0.0001 MOhm*cm, c_m = 1.0 uF/cm^2, v_ep = -75.0 mV, (g_TestChannel2 = 100.0 uS/cm^2, e_TestChannel2 = 100.0 mV)\n" \
            "    PhysNode 7, Parent: 4 --- r_a = 0.0001 MOhm*cm, c_m = 1.0 uF/cm^2, v_ep = -75.0 mV, (g_TestChannel2 = 100.0 uS/cm^2, e_TestChannel2 = 100.0 mV)\n" \
            "    PhysNode 8, Parent: 7 --- r_a = 0.0001 MOhm*cm, c_m = 1.0 uF/cm^2, v_ep = -75.0 mV, (g_TestChannel2 = 100.0 uS/cm^2, e_TestChannel2 = 100.0 mV)"

        assert repr(self.tree) == "['PhysTree', " \
            "\"{'node index': 1, 'parent index': -1, 'content': '{}', 'xyz': array([0., 0., 0.]), 'R': 10.0, 'swc_type': 1, 'currents': {'TestChannel2': [100.0, 100.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'v_ep': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 4, 'parent index': 1, 'content': '{}', 'xyz': array([100.,   0.,   0.]), 'R': 1.0, 'swc_type': 4, 'currents': {'TestChannel2': [100.0, 100.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'v_ep': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 5, 'parent index': 4, 'content': '{}', 'xyz': array([100.,  50.,   0.]), 'R': 1.0, 'swc_type': 4, 'currents': {'TestChannel2': [100.0, 100.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'v_ep': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 6, 'parent index': 5, 'content': '{}', 'xyz': array([100., 100.,   0.]), 'R': 0.5, 'swc_type': 4, 'currents': {'TestChannel2': [100.0, 100.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'v_ep': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 7, 'parent index': 4, 'content': '{}', 'xyz': array([100., -50.,   0.]), 'R': 1.0, 'swc_type': 4, 'currents': {'TestChannel2': [100.0, 100.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'v_ep': -75.0, 'conc_eps': {}}\", " \
            "\"{'node index': 8, 'parent index': 7, 'content': '{}', 'xyz': array([ 100., -100.,    0.]), 'R': 0.5, 'swc_type': 4, 'currents': {'TestChannel2': [100.0, 100.0]}, 'concmechs': {}, 'c_m': 1.0, 'r_a': 0.0001, 'g_shunt': 0.0, 'v_ep': -75.0, 'conc_eps': {}}\"" \
        "]"\
        "{'channel_storage': ['TestChannel2']}"



    def testLeakDistr(self):
        self.loadTree(reinitialize=1)
        with pytest.raises(AssertionError):
            self.tree.fitLeakCurrent(-75., -10.)
        # test simple distribution
        self.tree.fitLeakCurrent(-75., 10.)
        for node in self.tree:
            assert np.abs(node.c_m - 1.0) < 1e-9
            assert np.abs(node.currents['L'][0] - 1. / (10.*1e-3)) < 1e-9
            assert np.abs(node.v_ep + 75.) < 1e-9
        # create complex distribution
        tau_distr = lambda x: x + 100.
        for node in self.tree:
            d2s = self.tree.pathLength({'node': node.index, 'x': 1.}, (1., 0.5))
            node.fitLeakCurrent(self.tree.channel_storage,
                                e_eq_target=-75., tau_m_target=tau_distr(d2s))
            assert np.abs(node.c_m - 1.0) < 1e-9
            assert np.abs(node.currents['L'][0] - 1. / (tau_distr(d2s)*1e-3)) < \
                   1e-9
            assert np.abs(node.v_ep + 75.) < 1e-9

    def testPhysiologySetting(self):
        self.loadTree(reinitialize=1)
        d2s = {1:0., 4:50., 5:125., 6:175., 7:125., 8:175.}
        # passive parameters as float
        c_m = 1.; r_a = 100.*1e-6
        self.tree.setPhysiology(c_m, r_a)
        for node in self.tree:
            assert np.abs(node.c_m - c_m) < 1e-10
            assert np.abs(node.r_a - r_a) < 1e-10
        # passive parameters as function
        c_m = lambda x: .5*x + 1.
        r_a = lambda x: np.exp(0.01*x) * 100*1e-6
        self.tree.setPhysiology(c_m, r_a)
        for node in self.tree:
            assert np.abs(node.c_m - c_m(d2s[node.index])) < 1e-10
            assert np.abs(node.r_a - r_a(d2s[node.index])) < 1e-10
        # passive parameters as incomplete dict
        r_a = 100.*1e-6
        c_m = {1:1., 4:1.2}
        with pytest.raises(KeyError):
            self.tree.setPhysiology(c_m, r_a)
        # passive parameters as complete dict
        c_m.update({5:1.1, 6:0.9, 7:0.8, 8:1.})
        self.tree.setPhysiology(c_m, r_a)
        for node in self.tree:
            assert np.abs(node.c_m - c_m[node.index]) < 1e-10

        # equilibrium potential as float
        e_eq = -75.
        self.tree.setVEP(e_eq)
        for node in self.tree:
            assert np.abs(node.v_ep - e_eq) < 1e-10
        # equilibrium potential as dict
        e_eq = {1:-75., 4:-74., 5:-73., 6:-72., 7:-71., 8:-70.}
        self.tree.setVEP(e_eq)
        for node in self.tree:
            assert np.abs(node.v_ep - e_eq[node.index]) < 1e-10
        # equilibrium potential as function
        e_eq = lambda x: -70. + 0.1*x
        self.tree.setVEP(e_eq)
        for node in self.tree:
            assert np.abs(node.v_ep - e_eq(d2s[node.index])) < 1e-10
        # as wrong type
        with pytest.raises(TypeError):
            self.tree.setVEP([])
            self.tree.setPhysiology([], [])

        # leak as float
        g_l, e_l = 100., -75.
        self.tree.setLeakCurrent(g_l, e_l)
        for node in self.tree:
            g, e = node.currents['L']
            assert np.abs(g - g_l) < 1e-10
            assert np.abs(e - e_l) < 1e-10
        # equilibrium potential as dict
        g_l = {1:101., 4:103., 5:105., 6:107., 7:108., 8:109.}
        e_l = {1:-75., 4:-74., 5:-73., 6:-72., 7:-71., 8:-70.}
        self.tree.setLeakCurrent(g_l, e_l)
        for node in self.tree:
            g, e = node.currents['L']
            assert np.abs(g - g_l[node.index]) < 1e-10
            assert np.abs(e - e_l[node.index]) < 1e-10
        # equilibrium potential as function
        g_l = lambda x: 100. + 0.05*x
        e_l = lambda x: -70. + 0.05*x
        self.tree.setLeakCurrent(g_l, e_l)
        for node in self.tree:
            g, e = node.currents['L']
            assert np.abs(g - g_l(d2s[node.index])) < 1e-10
            assert np.abs(e - e_l(d2s[node.index])) < 1e-10
        # as wrong type
        with pytest.raises(TypeError):
            self.tree.setLeakCurrent([])

        # gmax as potential as float
        e_rev = 100.
        g_max = 100.
        channel = channelcollection.TestChannel2()
        self.tree.addCurrent(channel, g_max, e_rev)
        for node in self.tree:
            g_m = node.currents['TestChannel2'][0]
            assert np.abs(g_m - g_max) < 1e-10
        # equilibrium potential as dict
        g_max = {1:101., 4:103., 5:104., 6:106., 7:107., 8:110.}
        self.tree.addCurrent(channel, g_max, e_rev)
        for node in self.tree:
            g_m = node.currents['TestChannel2'][0]
            assert np.abs(g_m - g_max[node.index]) < 1e-10
        # equilibrium potential as function
        g_max = lambda x: 100. + 0.005 * x**2
        self.tree.addCurrent(channel, g_max, e_rev)
        for node in self.tree:
            g_m = node.currents['TestChannel2'][0]
            assert np.abs(g_m - g_max(d2s[node.index])) < 1e-10
        # test is channel is stored
        assert isinstance(self.tree.channel_storage[channel.__class__.__name__],
                          channelcollection.TestChannel2)
        # check if error is thrown if an ionchannel is not give
        with pytest.raises(IOError):
            self.tree.addCurrent('TestChannel2', g_max, e_rev)

    def testMembraneFunctions(self):
        self.loadTree(reinitialize=1)
        self.tree.setPhysiology(1., 100*1e-6)
        # passive parameters
        c_m = 1.; r_a = 100.*1e-6; e_eq = -75.
        self.tree.setPhysiology(c_m, r_a)
        self.tree.setVEP(e_eq)
        # channel
        p_open =  .9 * .3**3 * .5**2 + .1 * .4**2 * .6**1 # TestChannel2
        g_chan, e_chan = 100., 100.
        channel = channelcollection.TestChannel2()
        self.tree.addCurrent(channel, g_chan, e_chan)
        # fit the leak current
        self.tree.fitLeakCurrent(-30., 10.)

        # test if fit was correct
        for node in self.tree:
            tau_mem = c_m / (node.currents['L'][0] + g_chan*p_open) * 1e3
            assert np.abs(tau_mem - 10.) < 1e-10
            e_eq = (node.currents['L'][0]*node.currents['L'][1] + \
                    g_chan*p_open*e_chan) / (node.currents['L'][0] + g_chan*p_open)
            assert np.abs(e_eq - (-30.)) < 1e-10

        # test if warning is raised for impossible to reach time scale
        with pytest.warns(UserWarning):
            tree = copy.deepcopy(self.tree)
            tree.fitLeakCurrent(-30., 100000.)

        # total membrane conductance
        g_pas = self.tree[1].currents['L'][0] + g_chan*p_open
        i_pas = self.tree[1].currents['L'][0] * (-30. - self.tree[1].currents['L'][1]) + \
                g_chan*p_open * (-30. - e_chan)
        i_pas_ = self.tree[1].getITot(self.tree.channel_storage)
        # check that total current is zero at equilibrium
        assert np.abs(i_pas) < 1e-10
        assert np.abs(i_pas_) < 1e-10
        # make passive membrane
        tree = copy.deepcopy(self.tree)
        tree.asPassiveMembrane()
        # test if fit was correct
        for node in tree:
            assert np.abs(node.currents['L'][0] - g_pas) < 1e-10
            assert np.abs(node.currents['L'][1] - (-30.)) < 1e-10
        # test if channels storage is empty
        assert len(tree.channel_storage) == 0
        # test if computational root was removed
        assert tree._computational_root is None

        # test partial passification
        tree = copy.deepcopy(self.tree)
        # channel
        g_chan1, e_chan1 = 50., -100.
        channel1 = channelcollection.TestChannel()
        tree.addCurrent(channel1, g_chan, e_chan)
        # passify channel 2
        tree.asPassiveMembrane(channel_names=["TestChannel2"])
        for node in tree:
            assert set(node.currents.keys()) == {"TestChannel", "L"}
            assert np.abs(node.currents['L'][0] - g_pas) < 1e-10
            assert np.abs(node.getITot(tree.channel_storage)) < 1e-10


    def testCompTree(self):
        self.loadTree(reinitialize=1, segments=True)

        # capacitance axial resistance constant
        c_m = 1.; r_a = 100.*1e-6
        self.tree.setPhysiology(c_m, r_a)
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        assert [n.index for n in self.tree] == [1,8,10,12]
        # capacitance and axial resistance change
        c_m = lambda x: 1. if x < 200. else 1.6
        r_a = lambda x: 1. if x < 300. else 1.6
        self.tree.setPhysiology(c_m, r_a)
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        assert [n.index for n in self.tree] == [1,5,6,8,10,12]
        # leak current changes
        g_l = lambda x: 100. if x < 400. else 160.
        self.tree.setLeakCurrent(g_l, -75.)
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        assert [n.index for n in self.tree] == [1,5,6,7,8,10,12]
        # leak current & reversal change
        g_l = 100.
        e_l = {ind: -75. for ind in [1,4,5,6,7,8,11,12]}
        e_l.update({ind: -55. for ind in [9,10]})
        self.tree.setLeakCurrent(g_l, e_l)
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        assert [n.index for n in self.tree] == [1,5,6,8,10,12]
        # leak current & reversal change
        g_l = 100.
        e_l = {ind: -75. for ind in [1,4,5,6,7,8,10,11,12]}
        e_l.update({9: -55.})
        self.tree.setLeakCurrent(g_l, e_l)
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        assert [n.index for n in self.tree] == [1,5,6,8,9,10,12]
        # shunt
        self.tree.treetype = 'original'
        self.tree[7].g_shunt = 1.
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        assert [n.index for n in self.tree] == [1,5,6,7,8,9,10,12]


if __name__ == '__main__':
    tphys = TestPhysTree()
    # tphys.testStringRepresentation()
    # tphys.testLeakDistr()
    # tphys.testPhysiologySetting()
    # tphys.testMembraneFunctions()
    tphys.testCompTree()
