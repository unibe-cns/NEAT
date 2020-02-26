import numpy as np
import matplotlib.pyplot as pl

import pytest
import random
import copy

from neat import MorphLoc
from neat import PhysTree
from neat import CompartmentFitter
from neat.channels import channelcollection
import neat.tools.fittools.compartmentfitter as compartmentfitter


class TestCompartmentFitter():
    def loadTTree(self):
        '''
        Load the T-tree model

          6--5--4--7--8
                |
                |
                1
        '''
        print('>>> loading T-tree <<<')
        fname = 'test_morphologies/Tsovtree.swc'
        self.tree = PhysTree(fname, types=[1,3,4])
        self.tree.setPhysiology(0.8, 100./1e6)
        self.tree.fitLeakCurrent(-75., 10.)
        self.tree.setCompTree()

    def loadBallAndStick(self):
        '''
        Load the ball and stick model

        1--4
        '''
        self.tree = PhysTree(file_n='test_morphologies/ball_and_stick.swc')
        self.tree.setPhysiology(0.8, 100./1e6)
        self.tree.setLeakCurrent(100., -75.)
        self.tree.setCompTree()

    def loadBall(self):
        '''
        Load point neuron model
        '''
        self.tree = PhysTree(file_n='test_morphologies/ball.swc')
        # capacitance and axial resistance
        self.tree.setPhysiology(0.8, 100./1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()
        self.tree.addCurrent(k_chan, 0.766*1e6, -85.)
        na_chan = channelcollection.Na_Ta()
        self.tree.addCurrent(na_chan, 1.71*1e6, 50.)
        # fit leak current
        self.tree.fitLeakCurrent(-75., 10.)
        # set computational tree
        self.tree.setCompTree()

    def testTreeStructure(self):
        self.loadTTree()
        cm = CompartmentFitter(self.tree)
        # set of locations
        fit_locs1 = [(1,.5), (4,.5), (5,.5)] # no bifurcations
        fit_locs2 = [(1,.5), (4,.5), (5,.5), (8,.5)] # w bifurcation, should be added
        fit_locs3 = [(1,.5), (4,1.), (5,.5), (8,.5)] # w bifurcation, already added

        # test fit_locs1, no bifurcation are added
        # input paradigm 1
        cm.setReducedTree(fit_locs1, extend_w_bifurc=True)
        fl1_a = cm.tree.getLocs('fit locs')
        cm.setReducedTree(fit_locs1, extend_w_bifurc=False)
        fl1_b = cm.tree.getLocs('fit locs')
        assert len(fl1_a) == len(fl1_b)
        for fla, flb in zip(fl1_a, fl1_b): assert fla == flb
        # input paradigm 2
        cm.tree.storeLocs(fit_locs1, 'fl1')
        cm.setReducedTree('fl1', extend_w_bifurc=True)
        fl1_a = cm.tree.getLocs('fit locs')
        assert len(fl1_a) == len(fl1_b)
        for fla, flb in zip(fl1_a, fl1_b): assert fla == flb
        # test tree structure
        assert len(cm.ctree) == 3
        for cn in cm.ctree: assert len(cn.child_nodes) <= 1

        # test fit_locs2, a bifurcation should be added
        with pytest.warns(UserWarning):
            cm.setReducedTree(fit_locs2, extend_w_bifurc=False)
        fl2_b = cm.tree.getLocs('fit locs')
        cm.setReducedTree(fit_locs2, extend_w_bifurc=True)
        fl2_a = cm.tree.getLocs('fit locs')
        assert len(fl2_a) == len(fl2_b) + 1
        for fla, flb in zip(fl2_a, fl2_b): assert fla == flb
        assert fl2_a[-1] == (4,1.)
        # test tree structure
        assert len(cm.ctree) == 5
        for cn in cm.ctree:
            assert len(cn.child_nodes) <= 1 if cn.loc_ind != 4 else \
                   len(cn.child_nodes) == 2

        # test fit_locs2, no bifurcation should be added as it is already present
        cm.setReducedTree(fit_locs3, extend_w_bifurc=True)
        fl3 = cm.tree.getLocs('fit locs')
        for fl_, fl3 in zip(fit_locs3, fl3): assert fl_ == fl3
        # test tree structure
        assert len(cm.ctree) == 4
        for cn in cm.ctree:
            assert len(cn.child_nodes) <= 1 if cn.loc_ind != 1 else \
                   len(cn.child_nodes) == 2

    def _checkChannels(self, tree, channel_names):
        assert isinstance(tree, compartmentfitter.FitTreeGF)
        assert set(tree.channel_storage.keys()) == set(channel_names)
        for node in tree:
            assert set(node.currents.keys()) == set(channel_names + ['L'])

    def testCreateTreeGF(self):
        self.loadBall()
        cm = CompartmentFitter(self.tree)

        # create tree with only 'L'
        tree_pas = cm.createTreeGF()
        self._checkChannels(tree_pas, [])
        # create tree with only 'Na_Ta'
        tree_na = cm.createTreeGF(['Na_Ta'])
        self._checkChannels(tree_na, ['Na_Ta'])
        # create tree with only 'Kv3_1'
        tree_k = cm.createTreeGF(['Kv3_1'])
        self._checkChannels(tree_k, ['Kv3_1'])
        # create tree with all channels
        tree_all = cm.createTreeGF(['Na_Ta', 'Kv3_1'])
        self._checkChannels(tree_all, ['Na_Ta', 'Kv3_1'])






if __name__ == '__main__':
    tcf = TestCompartmentFitter()
    # tcf.testTreeStructure()
    tcf.testCreateTreeGF()
