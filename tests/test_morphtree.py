import numpy as np
import matplotlib.pyplot as pl
import os

import pytest

from neat import MorphTree, MorphNode, MorphLoc


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    'test_morphologies'
))


class TestMorphTree():
    def loadTree(self, reinitialize=0, segments=False):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1

        If `segments` is ``True``, tree is:

        12--11--8---9--10
                |
                7
                |
                6
                |
                5
                |
                4
                |
                1

        Standard associated computational tree is:

        12------8------10
                |
                1
        """
        if not hasattr(self, 'tree') or reinitialize:
            fname = 'Ttree_segments.swc' if segments else 'Ttree.swc'
            self.tree = MorphTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, fname), types=[1,3,4])

    def testStringRepresentation(self):
        self.loadTree()
        tree_str = f">>> MorphTree\n" \
            "    MorphNode 1, Parent: None --- xyz = [0.000, 0.000, 0.000] um, R = 10.00 um, swc_type = 1\n" \
            "    MorphNode 4, Parent: 1 --- xyz = [100.000, 0.000, 0.000] um, R = 1.00 um, swc_type = 4\n" \
            "    MorphNode 5, Parent: 4 --- xyz = [100.000, 50.000, 0.000] um, R = 1.00 um, swc_type = 4\n" \
            "    MorphNode 6, Parent: 5 --- xyz = [100.000, 100.000, 0.000] um, R = 0.50 um, swc_type = 4\n" \
            "    MorphNode 7, Parent: 4 --- xyz = [100.000, -50.000, 0.000] um, R = 1.00 um, swc_type = 4\n" \
            "    MorphNode 8, Parent: 7 --- xyz = [100.000, -100.000, 0.000] um, R = 0.50 um, swc_type = 4"
        assert str(self.tree) == tree_str

        repr_str = "['MorphTree', " \
            "\"{'node index': 1, 'parent index': -1, 'content': '{}', 'xyz': array([0., 0., 0.]), 'R': '10', 'swc_type': 1}\", " \
            "\"{'node index': 4, 'parent index': 1, 'content': '{}', 'xyz': array([100.,   0.,   0.]), 'R': '1', 'swc_type': 4}\", " \
            "\"{'node index': 5, 'parent index': 4, 'content': '{}', 'xyz': array([100.,  50.,   0.]), 'R': '1', 'swc_type': 4}\", " \
            "\"{'node index': 6, 'parent index': 5, 'content': '{}', 'xyz': array([100., 100.,   0.]), 'R': '0.5', 'swc_type': 4}\", " \
            "\"{'node index': 7, 'parent index': 4, 'content': '{}', 'xyz': array([100., -50.,   0.]), 'R': '1', 'swc_type': 4}\", " \
            "\"{'node index': 8, 'parent index': 7, 'content': '{}', 'xyz': array([ 100., -100.,    0.]), 'R': '0.5', 'swc_type': 4}\""\
        "]"
        assert repr(self.tree) == repr_str


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

    def testCompTree0(self):
        self.loadTree(reinitialize=1)
        # check exception when computational tree has not been set
        with pytest.raises(AttributeError):
            with self.tree.as_computational_tree:
                pass
        # initialize the comptree
        self.tree.set_comp_tree(eps=1.)
        # set computational tree as primary tree
        with self.tree.as_computational_tree:
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
        # work in the original tree
        assert len(self.tree.nodes) == 6
        leafs = self.tree.leafs
        assert np.allclose([leafs[0].L, leafs[1].L], [50., 50.])
        # remove the computational tree
        self.tree.remove_comp_tree()
        with pytest.raises(AttributeError):
            with self.tree.as_computational_tree:
                pass
        for node in self.tree:
            assert not node.used_in_comp_tree

    def testInputArgConversion(self):
        self.loadTree()
        nodes = self.tree.convert_node_arg_to_nodes(None)
        assert self.tree.nodes == nodes
        nodes = self.tree.convert_node_arg_to_nodes(self.tree[4])
        assert self.tree.gather_nodes(self.tree[4]) == nodes
        nodes = self.tree.convert_node_arg_to_nodes('apical')
        assert self.tree.get_nodes_in_apical_subtree() == nodes
        nodes = self.tree.convert_node_arg_to_nodes('basal')
        assert self.tree.get_nodes_in_basal_subtree() == nodes
        nodes = self.tree.convert_node_arg_to_nodes('axonal')
        assert self.tree.get_nodes_in_axonal_subtree() == nodes
        nodes_ = [self.tree[5], self.tree[7]]
        nodes = self.tree.convert_node_arg_to_nodes(nodes_)
        assert nodes_ == nodes
        with pytest.raises(IOError):
            self.tree.convert_node_arg_to_nodes('wrong arg')
        with pytest.raises(IOError):
            self.tree.convert_node_arg_to_nodes(5)
        # with the computational tree
        self.tree.set_comp_tree(eps=1.)
        with self.tree.as_computational_tree:
            nodes = self.tree.convert_node_arg_to_nodes(None)
            assert self.tree.nodes == nodes
            nodes = self.tree.convert_node_arg_to_nodes(self.tree[4])
            assert self.tree.gather_nodes(self.tree[4]) == nodes
            nodes = self.tree.convert_node_arg_to_nodes('apical')
            assert self.tree.get_nodes_in_apical_subtree() == nodes
            nodes = self.tree.convert_node_arg_to_nodes('basal')
            assert self.tree.get_nodes_in_basal_subtree() == nodes
            nodes = self.tree.convert_node_arg_to_nodes('axonal')
            assert self.tree.get_nodes_in_axonal_subtree() == nodes
            nodes__ = [self.tree[6], self.tree[8]]
            nodes = self.tree.convert_node_arg_to_nodes(nodes_)
            assert nodes_ != nodes
            assert nodes__ == nodes

    def testLocFunctionality(self):
        self.loadTree()
        locs = [MorphLoc({'node': 1, 'x': .5}, self.tree),
                MorphLoc({'node': 4, 'x': .5}, self.tree),
                MorphLoc({'node': 5, 'x': .5}, self.tree),
                MorphLoc({'node': 7, 'x': .5}, self.tree),
                MorphLoc({'node': 6, 'x': .5}, self.tree),
                MorphLoc({'node': 8, 'x': .5}, self.tree)]
        # set the computational tree
        self.tree.set_comp_tree(eps=1.)
        with self.tree.as_computational_tree:
            for loc in locs: loc._set_computational_loc()
            assert locs[0].comp_loc == locs[0].loc
            assert locs[1].comp_loc == locs[1].loc
            assert locs[2].comp_loc == {'node': 6, 'x': .25}
            assert locs[3].comp_loc == {'node': 8, 'x': .25}
            assert locs[4].comp_loc == {'node': 6, 'x': .75}
            assert locs[5].comp_loc == {'node': 8, 'x': .75}

    def testUniqueLocs(self):
        self.loadTree()
        locs = [
            (1.,.5),
            (4, 0.), (4, .1), (4, 1.),
                              (5, 0.), (5,.9), (5, 1.),
                                               (6, 0.), (6, 1.),
                              (7, 0.), (7, .5), (8, .5),
        ]
        unique_locs_groundtruth = [
            (1,.5),
                    (4, .1), (4, 1.),
                                      (5,.9), (5, 1.), (6, 1.),
                                      (7, .5), (8, .5)
        ]
        unique_locs_tested = self.tree.unique_locs(locs)

        for loc_tested, loc_groundtruth in zip(unique_locs_tested, unique_locs_groundtruth):
            assert loc_tested == loc_groundtruth

    def testPathLength(self):
        self.loadTree()
        # test lengths
        L = self.tree.path_length({'node': 1, 'x': 1.}, {'node': 1, 'x': .4})
        assert np.allclose(L, 0.)
        L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 1, 'x': .4})
        assert np.allclose(L, 100.)
        L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 1, 'x': .6})
        assert np.allclose(L, 100.)
        L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 4, 'x': 1.})
        assert np.allclose(L, 0.)
        L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 5, 'x': 0.})
        assert np.allclose(L, 0.)
        L = self.tree.path_length({'node': 4, 'x': .5}, {'node': 5, 'x': .2})
        assert np.allclose(L, 60.)
        L = self.tree.path_length({'node': 5, 'x': .2}, {'node': 4, 'x': .5})
        assert np.allclose(L, 60.)
        L = self.tree.path_length({'node': 5, 'x': .2}, {'node': 7, 'x': .2})
        assert np.allclose(L, 20.)
        L = self.tree.path_length({'node': 8, 'x': .2}, {'node': 4, 'x': .5})
        assert np.allclose(L, 110.)
        # lengths in the original tree
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            # test lengths
            L = self.tree.path_length({'node': 1, 'x': 1.}, {'node': 1, 'x': .4})
            assert np.allclose(L, 0.)
            L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 1, 'x': .4})
            assert np.allclose(L, 100.)
            L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 1, 'x': .6})
            assert np.allclose(L, 100.)
            L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 4, 'x': 1.})
            assert np.allclose(L, 0.)
            L = self.tree.path_length({'node': 4, 'x': 1.}, {'node': 5, 'x': 0.})
            assert np.allclose(L, 0.)
            L = self.tree.path_length({'node': 4, 'x': .5}, {'node': 5, 'x': .2})
            assert np.allclose(L, 60.)
            L = self.tree.path_length({'node': 5, 'x': .2}, {'node': 4, 'x': .5})
            assert np.allclose(L, 60.)
            L = self.tree.path_length({'node': 5, 'x': .2}, {'node': 7, 'x': .2})
            assert np.allclose(L, 20.)
            L = self.tree.path_length({'node': 8, 'x': .2}, {'node': 4, 'x': .5})
            assert np.allclose(L, 110.)

    def testLocStorageRetrievalLookup(self):
        self.loadTree()
        locs = [(1,.5), (1, 1.), (4, 1.), (4, .5), (5, .5),
               (5, 1.), (6, .5), (6, 1.), (8, .5), (8, 1.)]
        with pytest.warns(UserWarning):
            self.tree.store_locs(locs, 'testlocs')
        # with pytest.raises(ValueError):
        #     self.tree.store_locs(locs, 'testlocs')
        locs = [(1,.5), (4, 1.), (4, .5), (5, .5),
               (5, 1.), (6, .5), (6, 1.), (8, .5), (8, 1.)]
        self.tree.store_locs(locs, 'testlocs')
        # test retrieval
        with pytest.raises(KeyError):
            self.tree.get_locs('wronglocs')
        locs_ = [(loc['node'], loc['x'])
                    for loc in self.tree.get_locs('testlocs')]
        assert locs_ == locs
        # test getting node indices
        with pytest.raises(KeyError):
            self.tree.get_node_indices('wronglocs')
        assert [1,4,4,5,5,6,6,8,8] == self.tree.get_node_indices('testlocs').tolist()
        # test get x-coords
        with pytest.raises(KeyError):
            self.tree.get_x_coords('wronglocs')
        assert np.allclose([.5,1.,.5,.5,1.,.5,1.,.5,1.],
                           self.tree.get_x_coords('testlocs'))
        # test get locinds on non-empty node
        locinds = self.tree.get_loc_idxs_on_node('testlocs', self.tree[4])
        assert locinds == [2,1]
        # test get locinds on empty node
        locinds = self.tree.get_loc_idxs_on_node('testlocs', self.tree[7])
        assert locinds == []
        with pytest.raises(KeyError):
            self.tree.get_loc_idxs_on_node('wronglocs', self.tree[7])
        # test locinds on path
        path = self.tree.path_between_nodes(self.tree[6], self.tree[8])
        locinds = self.tree.get_loc_idxs_on_path('testlocs',
                            self.tree[6], self.tree[8], xstart=.5, xstop=1.)
        assert locinds == [5,4,3,1,7,8]
        path = self.tree.path_between_nodes(self.tree[7], self.tree[7])
        locinds = self.tree.get_loc_idxs_on_path('testlocs',
                            self.tree[7], self.tree[7], xstart=0., xstop=1.)
        assert locinds == []
        path = self.tree.path_between_nodes(self.tree[4], self.tree[4])
        locinds = self.tree.get_loc_idxs_on_path('testlocs',
                            self.tree[4], self.tree[4], xstart=0., xstop=.9)
        assert locinds == [2]
        path = self.tree.path_between_nodes(self.tree[1], self.tree[1])
        locinds = self.tree.get_loc_idxs_on_path('testlocs',
                            self.tree[1], self.tree[1], xstart=0., xstop=.9)
        assert locinds == [0]
        path = self.tree.path_between_nodes(self.tree[4], self.tree[1])
        locinds = self.tree.get_loc_idxs_on_path('testlocs',
                            self.tree[4], self.tree[1], xstart=.9, xstop=.9)
        assert locinds == [2,0]
        locinds = self.tree.get_loc_idxs_on_path('testlocs',
                            self.tree[7], self.tree[7], xstart=.1, xstop=.9)
        assert locinds == []
        # test locinds on branch
        branch = self.tree.path_between_nodes(self.tree[4], self.tree[5])
        locinds = self.tree.get_loc_idxs_on_nodes('testlocs', branch)
        assert locinds == [2,1,3,4]
        # test locinds on basal/apical/axonal subtrees
        locinds = self.tree.get_loc_idxs_on_nodes('testlocs', 'basal')
        assert locinds == []
        locinds = self.tree.get_loc_idxs_on_nodes('testlocs', 'axonal')
        assert locinds == []
        locinds = self.tree.get_loc_idxs_on_nodes('testlocs', 'apical')
        assert locinds == [2,1,3,4,5,6,7,8]
        # test locinds on subtree
        nodes = self.tree.get_nodes_in_subtree(self.tree[5], self.tree[4])
        locinds = self.tree.get_loc_idxs_on_nodes('testlocs', nodes)
        assert locinds == [2,1,3,4,5,6]
        # find the nearest locs
        locs = [(1,.5), (4, .5), (5, .4), (5, 1.),
                (6, .5), (6, 1.), (8, .5), (8, 1.)]
        self.tree.store_locs(locs, 'testlocs2')
        locinds = self.tree.get_nearest_loc_idxs([(7, .5), (1,.6), (6, .1), (6, .7)],
                            'testlocs2', direction=0)
        res = self.tree.get_nearest_loc_idxs([(6, .1)],
                            'testlocs2', direction=0)
        assert locinds == [2, 0, 3, 4]
        locinds = self.tree.get_nearest_loc_idxs([(7, .5), (1,.6), (6, .1), (6, .7)],
                            'testlocs2', direction=1)
        assert locinds == [2, 0, 3, 4]
        locinds = self.tree.get_nearest_loc_idxs([(7, .5), (1,.6), (6, .1), (6, .7)],
                            'testlocs2', direction=2)
        assert locinds == [6, 0, 4, 5]
        # find the leaf locs
        locinds = self.tree.get_leaf_loc_idxs('testlocs2')
        assert locinds == [5,7]
        # find the neartest locs
        locs = [(4, .1), (4, .4), (4, .7)]
        self.tree.store_locs(locs, 'testlocs3')
        locinds0 = self.tree.get_nearest_loc_idxs([(4, .4)],
                            'testlocs3', direction=0)
        locinds1 = self.tree.get_nearest_loc_idxs([(4, .4)],
                            'testlocs3', direction=1)
        locinds2 = self.tree.get_nearest_loc_idxs([(4, .4)],
                            'testlocs3', direction=2)
        assert locinds0[0] == 1
        assert locinds1[0] == 1
        assert locinds2[0] == 1

    def testNearestNeighbours(self):
        self.loadTree()

        locs1 = [(4,1.), (5,.5), (7,.5)]
        inds = self.tree.get_nearest_neighbour_loc_idxs((5,.7), locs1)
        assert set(inds) == {1}
        inds = self.tree.get_nearest_neighbour_loc_idxs((6,.7), locs1)
        assert set(inds) == {1}
        inds = self.tree.get_nearest_neighbour_loc_idxs((5,.5), locs1)
        assert set(inds) == {1}
        inds = self.tree.get_nearest_neighbour_loc_idxs((5,.4), locs1)
        assert set(inds) == {0,1}

        locs2 = [(4,.9), (5,.5), (7,.5)]
        inds = self.tree.get_nearest_neighbour_loc_idxs((5,.4), locs2)
        assert set(inds) == {0,1,2}

        locs3 = [(1,.5), (4,.9), (7,.5)]
        inds = self.tree.get_nearest_neighbour_loc_idxs((4,.8), locs3)
        assert set(inds) == {0,1}

        locs4 = [(5,.9), (6,.9), (7,.5), (8,.9)]
        inds = self.tree.get_nearest_neighbour_loc_idxs((1,.5), locs4)
        assert set(inds) == {0,2}

    def testDistances(self):
        self.loadTree()
        locs = [(1,.5), (4, 1.), (5, .5), (6, .5), (6, 1.)]
        self.tree.store_locs(locs, 'testlocs')
        # compute the distances to soma
        d2s = self.tree.distances_to_soma('testlocs')
        assert np.allclose(d2s, np.array([0.,100.,125.,175.,200.]))
        d2b = self.tree.distances_to_bifurcation('testlocs')
        assert np.allclose(d2b, np.array([0.,100.,25.,75.,100.]))

    def testLocDistribution(self):
        self.loadTree()
        # check comptree resetting
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            with pytest.raises(IOError):
                self.tree.distribute_locs_on_nodes(np.array([90.,140.,190.]), [])
        # test loc distribution on nodes
        locs = self.tree.distribute_locs_on_nodes(np.array([90.,140.,190.]),
                                                  [self.tree[6], self.tree[8]])
        assert locs[0] == {'node': 6, 'x': 4./5.} \
                and locs[1] == {'node': 8, 'x': 4./5.} \
                and len(locs) == 2
        locs = self.tree.distribute_locs_on_nodes(np.array([190.,190.]),
                                                  [self.tree[6]])
        assert locs[0] == locs[1] and locs[0] == {'node': 6, 'x': 4./5.}
        locs = self.tree.distribute_locs_on_nodes(np.array([]),
                                                  [self.tree[6]])
        assert len(locs) == 0
        locs = self.tree.distribute_locs_on_nodes(np.array([100.]),
                                                  [self.tree[4], self.tree[5]])
        assert locs[0] == {'node': 4, 'x': 1.}
        # derived loc distribution functions
        locs = self.tree.distribute_locs_on_nodes(np.array([150.,100.]))
        assert locs[0] == {'node': 4, 'x': 1.} \
                and locs[1] == {'node': 5, 'x': 1.} \
                and locs[2] == {'node': 7, 'x': 1.}
        nodes = self.tree.path_between_nodes(self.tree[5], self.tree[6])
        locs = self.tree.distribute_locs_on_nodes(np.array([50.,120.,170.]),
                                                  node_arg=nodes)
        assert locs[0] == {'node': 5, 'x': 2./5.} \
                and locs[1] == {'node': 6, 'x': 2./5.}
        locs = self.tree.distribute_locs_on_nodes(np.array([70.,120.]),
                                                  node_arg='basal')
        assert len(locs) == 0
        locs = self.tree.distribute_locs_on_nodes(np.array([70.,120.]),
                                                  node_arg='apical')
        assert locs[0] == {'node': 4, 'x': 7./10.} \
                and locs[1] == {'node': 5, 'x': 2./5.} \
                and locs[2] == {'node': 7, 'x': 2./5.}
        nodes = self.tree.get_nodes_in_subtree(self.tree[6], self.tree[4])
        locs = self.tree.distribute_locs_on_nodes(np.array([70.,120]),
                                                  node_arg=nodes)
        assert locs[0] == {'node': 4, 'x': 7./10.} \
                and locs[1] == {'node': 5, 'x': 2./5.}
        # test uniform loc distribution
        locs = self.tree.distribute_locs_uniform(50.)
        checklocs = [{'node': 1, 'x': .8},
                     {'node': 4, 'x': 1./2.}, {'node': 4, 'x': 1.},
                     {'node': 5, 'x': 1.}, {'node': 6, 'x': 1.},
                     {'node': 7, 'x': 1.}, {'node': 8, 'x': 1.}]
        for (loc, checkloc) in zip(locs, checklocs):
            assert loc == checkloc
        locs = self.tree.distribute_locs_uniform(50., node_arg=self.tree[5])
        checklocs = [{'node': 5, 'x': 1.}, {'node': 6, 'x': 1.}]
        for (loc, checkloc) in zip(locs, checklocs):
            assert loc == checkloc
        # test random loc distribution
        locs = self.tree.distribute_locs_random(10)
        assert len(locs) > 0
        for node in self.tree:
            assert 'tag' not in node.content
        locs = self.tree.distribute_locs_random(10, node_arg='basal', add_soma=0)
        assert len(locs) == 0
        with pytest.raises(IOError):
            self.tree.distribute_locs_random(10, node_arg='bad type')

    def testTreeCreation(self):
        self.loadTree(self)
        locs = [(1,.5),(4,.5),(4,1.),(5,1.),(6,1.),(7,1.),(8,1.)]
        self.tree.store_locs(locs, 'newtree_test')
        # create the new tree
        new_tree = self.tree.create_new_tree('newtree_test')
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
        self.tree.set_comp_tree()
        # create the x-axis
        self.tree.make_x_axis(50.)
        assert np.allclose(self.tree.xaxis,
                            np.array([0.,50.,100.,150.,200.,200.,250.]))
        # find x-values
        locs = [(4.,.7),(5,.9), (7,.8)]
        xvals = self.tree.get_x_values(locs)
        assert np.allclose([50.,150.,200.], xvals)
        xvals = self.tree.get_x_values([])
        assert len(xvals) == 0
        # plot on x-axis
        parr = np.array([1.,2.,3.,4.,5.,4.,3.])
        pl.figure('plot 1d test orig')
        # original tree
        lines = self.tree.plot_1d(pl.gca(), parr, c='r')
        assert np.abs(self.tree.xaxis[-1] - 250.) < 1e-5
        assert len(lines) == len(self.tree.get_leaf_loc_idxs('xaxis'))
        assert np.allclose(lines[0].get_data()[1], parr[0:5])
        assert np.allclose(lines[1].get_data()[1], parr[5:])
        xarr = np.array([0.,50.,100.,150.,200.,200.,250.])
        assert np.allclose(lines[0].get_data()[0], xarr[0:5])
        assert np.allclose(lines[1].get_data()[0], xarr[5:])
        assert lines[0].get_color() == 'r'
        # update the lines
        parr_ = np.array([5.,4.,3.,2.,1.,2.,3.])
        self.tree.set_line_data(lines, parr_)
        assert np.allclose(lines[0].get_data()[1], parr_[0:5])
        assert np.allclose(lines[1].get_data()[1], parr_[5:])
        # test x-axis coloring (just sees if function runs)
        self.tree.color_x_axis(pl.gca(), pl.get_cmap('jet'))
        # computational tree
        with self.tree.as_computational_tree:
            pl.figure('plot 1d test comp')
            lines = self.tree.plot_1d(pl.gca(), parr, c='r')
            assert np.abs(self.tree.xaxis[-1] - 250.) < 1e-5
            assert len(lines) == len(self.tree.get_leaf_loc_idxs('xaxis'))
            assert np.allclose(lines[0].get_data()[1], parr[0:5])
            assert np.allclose(lines[1].get_data()[1], parr[5:])
            xarr = np.array([0.,50.,100.,150.,200.,200.,250.])
            assert np.allclose(lines[0].get_data()[0], xarr[0:5])
            assert np.allclose(lines[1].get_data()[0], xarr[5:])
            assert lines[0].get_color() == 'r'
            # test x-axis coloring (just sees if function runs)
            self.tree.color_x_axis(pl.gca(), pl.get_cmap('jet'))
            # a true distance from soma plot
            parr = np.array([1.,2.,3.,4.,5.,4.,3.])
            pl.figure('plot d2s test')
            lines = self.tree.plot_true_d2s(pl.gca(),parr)
            assert len(lines) == len(self.tree.get_leaf_loc_idxs('xaxis'))
            assert np.allclose(lines[0].get_data()[1], parr[0:5])
            assert np.allclose(lines[1].get_data()[1], parr[5:])
            assert np.allclose(lines[0].get_data()[0], self.tree.d2s['xaxis'][0:5])
            assert np.allclose(lines[1].get_data()[0], self.tree.d2s['xaxis'][5:])
            # test 2D plotting (just sees if function runs)
            pl.figure('plot morphology test')
            cs = {node.index: node.index for node in self.tree}
            marklocs = [(1,.5), (4,.8), (6,.2)]
            loc_args = {'marker': 'o', 'mfc': 'r', 'ms': 6}
            marklabels = {0: r'soma', 1: r'$\mu$'}
            labelargs = {'fontsize': 8}
            self.tree.plot_2d_morphology(pl.gca(), cs=cs, cmap=pl.get_cmap('summer'),
                                        plotargs={'lw':2},
                                        marklocs=marklocs, loc_args=loc_args,
                                        marklabels=marklabels, labelargs=labelargs,
                                        textargs={'fontsize': 24},
                                        cb_draw=True)
            if pshow: pl.show()

    def testCompTree(self):
        self.loadTree(reinitialize=1, segments=True)
        self.tree.set_comp_tree()
        # check whether amount of nodes is correct
        assert len(self.tree.nodes) == 10
        assert len([node for node in self.tree]) == 10
        # get nodes in original tree
        ns_orig = self.tree.nodes
        with self.tree.as_computational_tree:
            assert len(self.tree.nodes) == 4
            assert len([node for node in self.tree]) == 4
            # check whether computational nodes are correct
            for node in self.tree: assert node.index in [1,8,10,12]
            # check whether returned nodes from `MorphTree.convert_node_arg_to_nodes`
            # are correct
            ns_comp = self.tree.convert_node_arg_to_nodes(ns_orig)
            assert [n.index for n in ns_comp] == [1,8,10,12]
            ns_comp = self.tree.convert_node_arg_to_nodes('apical')
            assert [n.index for n in ns_comp] == [8,10,12]
            ns_comp = self.tree.convert_node_arg_to_nodes('basal')
            assert [n.index for n in ns_comp] == []
            ns_comp = [self.tree.convert_node_arg_to_nodes(n)[0] for n in ns_orig]
            assert [n.index for n in ns_comp] == [1,8,8,8,8,8,10,10,12,12]
            # assert whether returned nodes have correct structure
            assert self.tree[5] == None
            assert self.tree[6] == None
            assert self.tree[7] == None
            assert self.tree[9] == None
            assert self.tree[11] == None
            assert self.tree[1].parent_node == None
            assert self.tree[8].parent_node.index == 1
            assert self.tree[10].parent_node.index == 8
            assert self.tree[12].parent_node.index == 8
            assert self.tree[1].child_nodes[0].index == 8
            assert [cn.index for cn in self.tree[8].child_nodes] == [10,12]
            assert self.tree[10].child_nodes == []
            assert self.tree[12].child_nodes == []

        # change radius of a node and check whether computational tree is correct
        # (i) below threshold
        self.tree[5].R += 1e-12
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            for node in self.tree: assert node.index in [1,8,10,12]

        # (i1) above threshold, one node
        self.tree[5].R += 1e-2
        self.tree.set_comp_tree(eps=1e-9)
        with self.tree.as_computational_tree:
            for node in self.tree: assert node.index in [1,4,5,8,10,12]

        # (i2) above threshold, two consecutive nodes
        self.tree[4].R += 1e-2
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            for node in self.tree: assert node.index in [1,5,8,10,12]


    def testOnePointSoma(self):
        mtree1 = MorphTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'onepoint_soma.swc'))
        mtree3 = MorphTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'threepoint_soma.swc'))

        # one point soma is internally converted to three-point soma, so we test
        # equivalence of both representations
        for ii in range(1,6):
            n1 = mtree1.__getitem__(ii, skip_inds=[])
            n3 = mtree3.__getitem__(ii, skip_inds=[])
            xyz1, R1, swc_type1 = n1.xyz, n1.R, n1.swc_type
            xyz3, R3, swc_type3 = n3.xyz, n3.R, n3.swc_type

            assert np.allclose(xyz1, xyz3)
            assert np.allclose(R1, R3)
            assert swc_type1 == swc_type3


    def testThreePointSoma(self):
        mtree = MorphTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'threepoint_soma.swc'))

        for n, idx in zip(mtree, [1,4,5]):
            assert n.index == idx

        idxs = [1,2,3,4,5]
        for ii, n in enumerate(mtree.__iter__(skip_inds=[])):
            assert n.index == idxs[ii]

        s_surface = 2. * np.pi * 5. * 5. + \
                    2. * np.pi * 5. * 5.
        s_radius = np.sqrt(s_surface / (4.*np.pi))

        assert mtree[1].R == s_radius

    def testMultiCylinderSoma(self):
        mtree = MorphTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'multicylinder_soma.swc'))

        for n, idx in zip(mtree, [1,7,8,9,10]):
            assert n.index == idx

        idxs = [1,2,3,7,8,9,10]
        for ii, n in enumerate(mtree.__iter__(skip_inds=[])):
            assert n.index == idxs[ii]

        s_radius = np.sqrt(2.*np.pi*10. * 5. / (4.*np.pi))

        assert mtree[1].R == s_radius

        assert np.allclose(mtree[1].xyz, np.array([0.,2.5,0.]))

    def testWrongSoma(self):
        with pytest.raises(ValueError):
            MorphTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, 'wrong_soma.swc'))

    def testCopyConstruct(self):
        self.loadTree(reinitialize=True, segments=True)
        tree1 = MorphTree(self.tree)

        self.tree.set_comp_tree()
        tree2 = MorphTree(self.tree)

        assert tree1._computational_root is None
        assert tree2._computational_root is not None
        
        assert repr(tree2) == repr(self.tree)
        assert repr(tree2) == repr(tree1)
        with tree2.as_computational_tree:
            with self.tree.as_computational_tree:
                assert repr(tree2) == repr(self.tree)
                assert repr(tree2) != repr(tree1)

        tree1.set_comp_tree()
        assert repr(tree1) == repr(tree2)
        with tree2.as_computational_tree:
            with tree1.as_computational_tree:
                assert repr(tree2) == repr(tree1)


if __name__ == '__main__':
    tmt = TestMorphTree()
    # tmt.testStringRepresentation()
    # tmt.testPlotting(pshow=True)
    # tmt.testCompTree0()
    # tmt.testInputArgConversion()
    # tmt.testLocFunctionality()
    # tmt.testUniqueLocs()
    # tmt.testLocStorageRetrievalLookup()
    # tmt.testNearestNeighbours()
    # tmt.testCompTree()
    # tmt.testMultiCylinderSoma()
    # tmt.testOnePointSoma()
    # tmt.testThreePointSoma()
    # tmt.testWrongSoma()
    tmt.testCopyConstruct()

