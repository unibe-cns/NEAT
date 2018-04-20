import numpy as np
import matplotlib.pyplot as pl

import pytest

from neat import MorphTree, MorphNode, MorphLoc


class TestMorphTree():
    def loadTree(self, reinitialize=0):
        '''
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1

        Standard associated computational tree is:

          6-----4-----8
                |
                |
                1

        '''
        if not hasattr(self, 'tree') or reinitialize:
            print '>>> loading T-tree <<<'
            fname = 'test_morphologies/Ttree.swc'
            self.tree = MorphTree(fname, types=[1,3,4])

    def testLocEquality(self):
        self.loadTree()
        loc1 = MorphLoc((4,.5), self.tree)
        loc2 = MorphLoc((4,.5), self.tree)
        loc3 = MorphLoc((4,.7), self.tree)
        assert loc1 == (4,.5) and (4,.5) == loc1
        assert loc1 == {'node': 4, 'x':.5} and {'node': 4, 'x':.5} == loc1
        assert loc1 == loc2 and loc2 == loc1
        assert loc1 != (4,.7) and (4,.7) != loc1
        assert loc1 != {'node': 5, 'x':.5} and {'node': 5, 'x':.5} != loc1
        assert loc1 != loc3 and loc3 != loc1
        loc4 = MorphLoc((1,.5), self.tree)
        assert loc4 == (1,.8) and loc4 != (4,.5)

    def testIteration(self):
        self.loadTree()
        indices = [node.index for node in self.tree]
        assert 2 not in indices and 3 not in indices
        assert set(indices) == set([1,4,5,6,7,8])
        indices = [node.index for node in self.tree.nodes]
        assert 2 not in indices and 3 not in indices
        assert set(indices) == set([1,4,5,6,7,8])

    def testStructure(self):
        self.loadTree()
        # check root
        root = self.tree[1]
        assert root.parent_node == None
        assert len(root.child_nodes) == 1 \
               and set([node.index for node in root.child_nodes]) == set([4])
        # check bifurcation node
        bifur = self.tree[4]
        assert bifur.parent_node == root
        assert len(bifur.child_nodes) == 2 \
               and set([node.index for node in bifur.child_nodes]) == set([5,7])
        # check branch nodes
        bn1 = self.tree[5]
        assert bn1.parent_node == bifur
        assert len(bn1.child_nodes) == 1 and bn1.child_nodes[0].index == 6
        bn2 = self.tree[7]
        assert bn2.parent_node == bifur
        assert len(bn2.child_nodes) == 1 and bn2.child_nodes[0].index == 8
        # check leaf nodes
        leaf1 = self.tree[6]
        assert leaf1.parent_node == bn1
        assert len(leaf1.child_nodes) == 0
        leaf2 = self.tree[8]
        assert leaf2.parent_node == bn2
        assert len(leaf2.child_nodes) == 0

    def testGeometry(self):
        self.loadTree()
        # check root
        root = self.tree[1]
        assert np.allclose(root.xyz, np.array([0.,0.,0.]))
        assert np.allclose(root.R, 10.)
        assert np.allclose(root.L, 0.)
        assert root.swc_type == 1
        # check bifurcation node
        bifur = self.tree[4]
        assert np.allclose(bifur.xyz, np.array([100.,0.,0.]))
        assert np.allclose(bifur.R, 1.)
        assert np.allclose(bifur.L, 100.)
        assert bifur.swc_type == 4
        # check branch nodes
        bn1 = self.tree[5]
        assert np.allclose(bn1.xyz, np.array([100.,50.,0.]))
        assert np.allclose(bn1.R, 1.)
        assert np.allclose(bn1.L, 50.)
        assert bn1.swc_type == 4
        bn2 = self.tree[7]
        assert np.allclose(bn2.xyz, np.array([100.,-50.,0.]))
        assert np.allclose(bn2.R, 1.)
        assert np.allclose(bn2.L, 50.)
        assert bn2.swc_type == 4
        # check leaf nodes
        leaf1 = self.tree[6]
        assert np.allclose(leaf1.xyz, np.array([100.,100.,0.]))
        assert np.allclose(leaf1.R, .5)
        assert np.allclose(leaf1.L, 50.)
        assert leaf1.swc_type == 4
        leaf2 = self.tree[8]
        assert np.allclose(leaf2.xyz, np.array([100.,-100.,0.]))
        assert np.allclose(leaf2.R, .5)
        assert np.allclose(leaf2.L, 50.)
        assert leaf2.swc_type == 4

    def testComptree(self):
        self.loadTree(reinitialize=1)
        # check exception when treetype is invalid
        with pytest.raises(ValueError):
            self.tree.treetype = 'bla'
        # check exception when computational tree has not been set
        with pytest.raises(ValueError):
            self.tree.treetype = 'computational'
        # initialize the comptree
        self.tree.setCompTree()
        # set computational tree as primary tree
        self.tree.treetype = 'computational'
        # check root node
        root = self.tree[1]
        assert root.parent_node == None
        assert len(root.child_nodes) == 1 \
               and set([node.index for node in root.child_nodes]) == set([4])
        assert np.allclose(root.xyz, np.array([0.,0.,0.]))
        assert np.allclose(root.R, 10.)
        assert np.allclose(root.L, 0.)
        assert root.swc_type == 1
        # check bifurcation node
        bifur = self.tree[4]
        assert bifur.parent_node == root
        assert len(bifur.child_nodes) == 2 \
               and set([node.index for node in bifur.child_nodes]) == set([6,8])
        assert np.allclose(bifur.xyz, np.array([100.,0.,0.]))
        assert np.allclose(bifur.R, 1.)
        assert np.allclose(bifur.L, 100.)
        assert bifur.swc_type == 4
        # check leaf nodes
        leaf1 = self.tree[6]
        assert leaf1.parent_node == bifur
        assert len(leaf1.child_nodes) == 0
        assert np.allclose(leaf1.xyz, np.array([100.,100.,0.]))
        assert np.allclose(leaf1.R, .75)
        assert np.allclose(leaf1.L, 100.)
        assert leaf1.swc_type == 4
        leaf2 = self.tree[8]
        assert leaf2.parent_node == bifur
        assert len(leaf2.child_nodes) == 0
        assert np.allclose(leaf2.xyz, np.array([100.,-100.,0.]))
        assert np.allclose(leaf2.R, .75)
        assert np.allclose(leaf2.L, 100.)
        # test nodes getter
        assert len(self.tree.nodes) == 4
        leafs = self.tree.leafs
        assert np.allclose([leafs[0].L, leafs[1].L], [100., 100.])
        self.tree.treetype = 'original'
        assert len(self.tree.nodes) == 6
        leafs = self.tree.leafs
        assert np.allclose([leafs[0].L, leafs[1].L], [50., 50.])
        # remove the computational tree
        self.tree.removeComptree()
        with pytest.raises(ValueError):
            self.tree.treetype = 'computational'
        for node in self.tree:
            assert not node.used_in_comptree

    def testInputArgConversion(self):
        self.loadTree()
        nodes = self.tree._convertNodeArgToNodes(None)
        assert self.tree.nodes == nodes
        nodes = self.tree._convertNodeArgToNodes(self.tree[4])
        assert self.tree.gatherNodes(self.tree[4]) == nodes
        nodes = self.tree._convertNodeArgToNodes('apical')
        assert self.tree.getNodesInApicalSubtree() == nodes
        nodes = self.tree._convertNodeArgToNodes('basal')
        assert self.tree.getNodesInBasalSubtree() == nodes
        nodes = self.tree._convertNodeArgToNodes('axonal')
        assert self.tree.getNodesInAxonalSubtree() == nodes
        nodes_ = [self.tree[5], self.tree[7]]
        nodes = self.tree._convertNodeArgToNodes(nodes_)
        assert nodes_ == nodes
        with pytest.raises(ValueError):
            self.tree._convertNodeArgToNodes('wrong arg')
            self.tree._convertNodeArgToNodes(5)
        # with the computational tree
        self.tree.setCompTree(set_as_primary_tree=1)
        nodes = self.tree._convertNodeArgToNodes(None)
        assert self.tree.nodes == nodes
        nodes = self.tree._convertNodeArgToNodes(self.tree[4])
        assert self.tree.gatherNodes(self.tree[4]) == nodes
        nodes = self.tree._convertNodeArgToNodes('apical')
        assert self.tree.getNodesInApicalSubtree() == nodes
        nodes = self.tree._convertNodeArgToNodes('basal')
        assert self.tree.getNodesInBasalSubtree() == nodes
        nodes = self.tree._convertNodeArgToNodes('axonal')
        assert self.tree.getNodesInAxonalSubtree() == nodes
        nodes__ = [self.tree[6], self.tree[8]]
        nodes = self.tree._convertNodeArgToNodes(nodes_)
        assert nodes_ != nodes
        assert nodes__ == nodes


    def testLocFunctionality(self):
        self.loadTree()
        # locs in the original tree
        self.tree.treetype = 'original'
        locs = [MorphLoc({'node': 1, 'x': .5}, self.tree),
                MorphLoc({'node': 4, 'x': .5}, self.tree),
                MorphLoc({'node': 5, 'x': .5}, self.tree),
                MorphLoc({'node': 7, 'x': .5}, self.tree),
                MorphLoc({'node': 6, 'x': .5}, self.tree),
                MorphLoc({'node': 8, 'x': .5}, self.tree)]
        # set the computational tree
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        for loc in locs: loc._setComputationalLoc()
        assert locs[0].comp_loc == locs[0].loc
        assert locs[1].comp_loc == locs[1].loc
        assert locs[2].comp_loc == {'node': 6, 'x': .25}
        assert locs[3].comp_loc == {'node': 8, 'x': .25}
        assert locs[4].comp_loc == {'node': 6, 'x': .75}
        assert locs[5].comp_loc == {'node': 8, 'x': .75}

    def testPathLength(self):
        self.loadTree()
        # lengths in the original tree
        self.tree.treetype = 'original'
        # test lengths
        L = self.tree.pathLength({'node': 1, 'x': 1.}, {'node': 1, 'x': .4})
        assert np.allclose(L, 0.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 1, 'x': .4})
        assert np.allclose(L, 100.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 1, 'x': .6})
        assert np.allclose(L, 100.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 4, 'x': 1.})
        assert np.allclose(L, 0.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 5, 'x': 0.})
        assert np.allclose(L, 0.)
        L = self.tree.pathLength({'node': 4, 'x': .5}, {'node': 5, 'x': .2})
        assert np.allclose(L, 60.)
        L = self.tree.pathLength({'node': 5, 'x': .2}, {'node': 4, 'x': .5})
        assert np.allclose(L, 60.)
        L = self.tree.pathLength({'node': 5, 'x': .2}, {'node': 7, 'x': .2})
        assert np.allclose(L, 20.)
        L = self.tree.pathLength({'node': 8, 'x': .2}, {'node': 4, 'x': .5})
        assert np.allclose(L, 110.)
        # lengths in the original tree
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        # test lengths
        L = self.tree.pathLength({'node': 1, 'x': 1.}, {'node': 1, 'x': .4})
        assert np.allclose(L, 0.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 1, 'x': .4})
        assert np.allclose(L, 100.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 1, 'x': .6})
        assert np.allclose(L, 100.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 4, 'x': 1.})
        assert np.allclose(L, 0.)
        L = self.tree.pathLength({'node': 4, 'x': 1.}, {'node': 5, 'x': 0.})
        assert np.allclose(L, 0.)
        L = self.tree.pathLength({'node': 4, 'x': .5}, {'node': 5, 'x': .2})
        assert np.allclose(L, 60.)
        L = self.tree.pathLength({'node': 5, 'x': .2}, {'node': 4, 'x': .5})
        assert np.allclose(L, 60.)
        L = self.tree.pathLength({'node': 5, 'x': .2}, {'node': 7, 'x': .2})
        assert np.allclose(L, 20.)
        L = self.tree.pathLength({'node': 8, 'x': .2}, {'node': 4, 'x': .5})
        assert np.allclose(L, 110.)

    def testLocStorageRetrievalLookup(self):
        self.loadTree()
        locs = [(1,.5), (1, 1.), (4, 1.), (4, .5), (5, .5),
               (5, 1.), (6, .5), (6, 1.), (8, .5), (8, 1.)]
        with pytest.raises(ValueError):
            self.tree.storeLocs(locs, 'testlocs')
        locs = [(1,.5), (4, 1.), (4, .5), (5, .5),
               (5, 1.), (6, .5), (6, 1.), (8, .5), (8, 1.)]
        self.tree.storeLocs(locs, 'testlocs')
        # test retrieval
        with pytest.raises(KeyError):
            self.tree.getLocs('wronglocs')
        locs_ = [(loc['node'], loc['x'])
                    for loc in self.tree.getLocs('testlocs')]
        assert locs_ == locs
        # test getting node indices
        with pytest.raises(KeyError):
            self.tree.getNodeIndices('wronglocs')
        assert [1,4,4,5,5,6,6,8,8] == self.tree.getNodeIndices('testlocs').tolist()
        # test get x-coords
        with pytest.raises(KeyError):
            self.tree.getXCoords('wronglocs')
        assert np.allclose([.5,1.,.5,.5,1.,.5,1.,.5,1.],
                           self.tree.getXCoords('testlocs'))
        # test get locinds on non-empty node
        locinds = self.tree.getLocindsOnNode('testlocs', self.tree[4])
        assert locinds == [2,1]
        # test get locinds on empty node
        locinds = self.tree.getLocindsOnNode('testlocs', self.tree[7])
        assert locinds == []
        with pytest.raises(KeyError):
            self.tree.getLocindsOnNode('wronglocs', self.tree[7])
        # test locinds on path
        path = self.tree.pathBetweenNodes(self.tree[6], self.tree[8])
        locinds = self.tree.getLocindsOnPath('testlocs',
                            self.tree[6], self.tree[8], xstart=.5, xstop=1.)
        assert locinds == [5,4,3,1,7,8]
        path = self.tree.pathBetweenNodes(self.tree[7], self.tree[7])
        locinds = self.tree.getLocindsOnPath('testlocs',
                            self.tree[7], self.tree[7], xstart=0., xstop=1.)
        assert locinds == []
        path = self.tree.pathBetweenNodes(self.tree[4], self.tree[4])
        locinds = self.tree.getLocindsOnPath('testlocs',
                            self.tree[4], self.tree[4], xstart=0., xstop=.9)
        assert locinds == [2]
        path = self.tree.pathBetweenNodes(self.tree[1], self.tree[1])
        locinds = self.tree.getLocindsOnPath('testlocs',
                            self.tree[1], self.tree[1], xstart=0., xstop=.9)
        assert locinds == [0]
        path = self.tree.pathBetweenNodes(self.tree[4], self.tree[1])
        locinds = self.tree.getLocindsOnPath('testlocs',
                            self.tree[4], self.tree[1], xstart=.9, xstop=.9)
        assert locinds == [2,0]
        locinds = self.tree.getLocindsOnPath('testlocs',
                            self.tree[7], self.tree[7], xstart=.1, xstop=.9)
        assert locinds == []
        # test locinds on branch
        branch = self.tree.pathBetweenNodes(self.tree[4], self.tree[5])
        locinds = self.tree.getLocindsOnNodes('testlocs', branch)
        assert locinds == [2,1,3,4]
        # test locinds on basal/apical/axonal subtrees
        locinds = self.tree.getLocindsOnNodes('testlocs', 'basal')
        assert locinds == [0]
        locinds = self.tree.getLocindsOnNodes('testlocs', 'axonal')
        assert locinds == [0]
        locinds = self.tree.getLocindsOnNodes('testlocs', 'apical')
        assert locinds == [0,2,1,3,4,5,6,7,8]
        # test locinds on subtree
        nodes = self.tree.getNodesInSubtree(self.tree[5], self.tree[4])
        locinds = self.tree.getLocindsOnNodes('testlocs', nodes)
        assert locinds == [2,1,3,4,5,6]
        # find the nearest locs
        locs = [(1,.5), (4, .5), (5, .4), (5, 1.),
                (6, .5), (6, 1.), (8, .5), (8, 1.)]
        self.tree.storeLocs(locs, 'testlocs2')
        locinds = self.tree.getNearestLocinds([(7, .5), (1,.6), (6, .1), (6, .7)],
                            'testlocs2', direction=0)
        assert locinds == [2, 0, 3, 4]
        locinds = self.tree.getNearestLocinds([(7, .5), (1,.6), (6, .1), (6, .7)],
                            'testlocs2', direction=1)
        assert locinds == [2, 0, 3, 4]
        locinds = self.tree.getNearestLocinds([(7, .5), (1,.6), (6, .1), (6, .7)],
                            'testlocs2', direction=2)
        assert locinds == [6, 0, 4, 5]
        # find the leaf locs
        locinds = self.tree.getLeafLocinds('testlocs2')
        assert locinds == [5,7]

    def testDistances(self):
        self.loadTree()
        locs = [(1,.5), (4, 1.), (5, .5), (6, .5), (6, 1.)]
        self.tree.storeLocs(locs, 'testlocs')
        # compute the distances to soma
        d2s = self.tree.distancesToSoma('testlocs')
        assert np.allclose(d2s, np.array([0.,100.,125.,175.,200.]))
        d2b = self.tree.distancesToBifurcation('testlocs')
        assert np.allclose(d2b, np.array([0.,100.,25.,75.,100.]))

    def testLocDistribution(self):
        self.loadTree()
        # check comptree resetting
        self.tree.setCompTree()
        self.tree.treetype = 'computational'
        self.tree.distributeLocsOnNodes(np.array([90.,140.,190.]), [])
        assert self.tree.treetype == 'computational'
        self.tree.treetype = 'original'
        # test loc distribution on nodes
        locs = self.tree.distributeLocsOnNodes(np.array([90.,140.,190.]),
                                                  [self.tree[6], self.tree[8]])
        assert locs[0] == {'node': 6, 'x': 4./5.} \
                and locs[1] == {'node': 8, 'x': 4./5.} \
                and len(locs) == 2
        locs = self.tree.distributeLocsOnNodes(np.array([190.,190.]),
                                                  [self.tree[6]])
        assert locs[0] == locs[1] and locs[0] == {'node': 6, 'x': 4./5.}
        locs = self.tree.distributeLocsOnNodes(np.array([]),
                                                  [self.tree[6]])
        assert len(locs) == 0
        locs = self.tree.distributeLocsOnNodes(np.array([100.]),
                                                  [self.tree[4], self.tree[5]])
        assert locs[0] == {'node': 4, 'x': 1.}
        # derived loc distribution functions
        locs = self.tree.distributeLocsOnNodes(np.array([150.,100.]))
        assert locs[0] == {'node': 4, 'x': 1.} \
                and locs[1] == {'node': 5, 'x': 1.} \
                and locs[2] == {'node': 7, 'x': 1.}
        nodes = self.tree.pathBetweenNodes(self.tree[5], self.tree[6])
        locs = self.tree.distributeLocsOnNodes(np.array([50.,120.,170.]),
                                                  node_arg=nodes)
        assert locs[0] == {'node': 5, 'x': 2./5.} \
                and locs[1] == {'node': 6, 'x': 2./5.}
        locs = self.tree.distributeLocsOnNodes(np.array([70.,120.]),
                                                  node_arg='basal')
        assert len(locs) == 0
        locs = self.tree.distributeLocsOnNodes(np.array([70.,120.]),
                                                  node_arg='apical')
        assert locs[0] == {'node': 4, 'x': 7./10.} \
                and locs[1] == {'node': 5, 'x': 2./5.} \
                and locs[2] == {'node': 7, 'x': 2./5.}
        nodes = self.tree.getNodesInSubtree(self.tree[6], self.tree[4])
        locs = self.tree.distributeLocsOnNodes(np.array([70.,120]),
                                                  node_arg=nodes)
        assert locs[0] == {'node': 4, 'x': 7./10.} \
                and locs[1] == {'node': 5, 'x': 2./5.}
        # test uniform loc distribution
        locs = self.tree.distributeLocsUniform(50.)
        checklocs = [{'node': 1, 'x': .8},
                     {'node': 4, 'x': 1./2.}, {'node': 4, 'x': 1.},
                     {'node': 5, 'x': 1.}, {'node': 6, 'x': 1.},
                     {'node': 7, 'x': 1.}, {'node': 8, 'x': 1.}]
        for (loc, checkloc) in zip(locs, checklocs):
            assert loc == checkloc
        locs = self.tree.distributeLocsUniform(50., node_arg=self.tree[5])
        for loc in locs: print loc
        checklocs = [{'node': 5, 'x': 1.}, {'node': 6, 'x': 1.}]
        for (loc, checkloc) in zip(locs, checklocs):
            assert loc == checkloc
        # test random loc distribution
        locs = self.tree.distributeLocsRandom(10)
        assert len(locs) > 0
        for node in self.tree:
            assert 'tag' not in node.content
        locs = self.tree.distributeLocsRandom(10, node_arg='basal', add_soma=0)
        assert len(locs) == 0
        with pytest.raises(ValueError):
            self.tree.distributeLocsRandom(10, node_arg='bad type')

    def testTreeCreation(self):
        self.loadTree(self)
        locs = [(1,.5),(4,.5),(4,1.),(5,1.),(6,1.),(7,1.),(8,1.)]
        self.tree.storeLocs(locs, 'newtree_test')
        # create the new tree
        new_tree = self.tree.createNewTree('newtree_test')
        new_xyzs = [(0.,0.,0.),
                    (50.,0.,0.),
                    (100.,0.,0.),
                    (100.,50.,0.),
                    (100.,100.,0.),
                    (100.,-50.,0.),
                    (100.,-100.,0.)]
        new_inds = [1,4,5,6,7,8,9]
        for ii, new_node in enumerate(new_tree):
            assert np.allclose(new_xyzs[ii], new_node.xyz)
            assert new_inds[ii] == new_node.index

    def testPlotting(self, pshow=0):
        self.loadTree()
        self.tree.setCompTree()
        # create the x-axis
        self.tree.makeXAxis(50.)
        assert np.allclose(self.tree.xaxis,
                            np.array([0.,50.,100.,150.,200.,200.,250.]))
        # find x-values
        locs = [(4.,.7),(5,.9), (7,.8)]
        xvals = self.tree.getXValues(locs)
        assert np.allclose([50.,150.,200.], xvals)
        xvals = self.tree.getXValues([])
        assert len(xvals) == 0
        # plot on x-axis
        parr = np.array([1.,2.,3.,4.,5.,4.,3.])
        pl.figure('plot 1d test orig')
        # original tree
        lines = self.tree.plot1D(pl.gca(), parr, c='r')
        assert np.abs(self.tree.xaxis[-1] - 250.) < 1e-5
        assert len(lines) == len(self.tree.getLeafLocinds('xaxis'))
        assert np.allclose(lines[0].get_data()[1], parr[0:5])
        assert np.allclose(lines[1].get_data()[1], parr[5:])
        xarr = np.array([0.,50.,100.,150.,200.,200.,250.])
        assert np.allclose(lines[0].get_data()[0], xarr[0:5])
        assert np.allclose(lines[1].get_data()[0], xarr[5:])
        assert lines[0].get_color() == 'r'
        # update the lines
        parr_ = np.array([5.,4.,3.,2.,1.,2.,3.])
        self.tree.setLineData(lines, parr_)
        assert np.allclose(lines[0].get_data()[1], parr_[0:5])
        assert np.allclose(lines[1].get_data()[1], parr_[5:])
        # test x-axis coloring (just sees if function runs)
        self.tree.colorXAxis(pl.gca(), pl.get_cmap('jet'))
        # computational tree
        self.tree.treetype = 'computational'
        pl.figure('plot 1d test comp')
        lines = self.tree.plot1D(pl.gca(), parr, c='r')
        assert np.abs(self.tree.xaxis[-1] - 250.) < 1e-5
        assert len(lines) == len(self.tree.getLeafLocinds('xaxis'))
        assert np.allclose(lines[0].get_data()[1], parr[0:5])
        assert np.allclose(lines[1].get_data()[1], parr[5:])
        xarr = np.array([0.,50.,100.,150.,200.,200.,250.])
        assert np.allclose(lines[0].get_data()[0], xarr[0:5])
        assert np.allclose(lines[1].get_data()[0], xarr[5:])
        assert lines[0].get_color() == 'r'
        # test x-axis coloring (just sees if function runs)
        self.tree.colorXAxis(pl.gca(), pl.get_cmap('jet'))
        # a true distance from soma plot
        parr = np.array([1.,2.,3.,4.,5.,4.,3.])
        pl.figure('plot d2s test')
        lines = self.tree.plotTrueD2S(pl.gca(),parr)
        assert len(lines) == len(self.tree.getLeafLocinds('xaxis'))
        assert np.allclose(lines[0].get_data()[1], parr[0:5])
        assert np.allclose(lines[1].get_data()[1], parr[5:])
        assert np.allclose(lines[0].get_data()[0], self.tree.d2s['xaxis'][0:5])
        assert np.allclose(lines[1].get_data()[0], self.tree.d2s['xaxis'][5:])
        # test 2D plotting (just sees if function runs)
        pl.figure('plot morphology test')
        cs = {node.index: node.index for node in self.tree}
        marklocs = [(1,.5), (4,.8), (6,.2)]
        locargs = {'marker': 'o', 'mfc': 'r', 'ms': 6}
        marklabels = {0: r'soma', 1: r'$\mu$'}
        labelargs = {'fontsize': 8}
        self.tree.plot2DMorphology(pl.gca(), cs=cs, cmap=pl.get_cmap('summer'),
                                     plotargs={'lw':2},
                                     marklocs=marklocs, locargs=locargs,
                                     marklabels=marklabels, labelargs=labelargs,
                                     textargs={'fontsize': 24},
                                     cb_draw=True)
        if pshow: pl.show()

if __name__ == '__main__':
    tmt = TestMorphTree()
    # tmt.testPlotting(pshow=True)
    tmt.testComptree()

