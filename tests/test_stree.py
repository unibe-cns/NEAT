# -*- coding: utf-8 -*-
#
# test_stree.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

from neat import STree, SNode

import pytest


class TestSTree():
    def create_tree(self, reinitialize=0):
        # Create a simple tree structure

        #  2     3
        #   \   /
        #    \ /
        #     1
        #     |
        #     |
        #     0
        # create the four nodes
        node0 = SNode(0)
        node1 = SNode(1)
        node2 = SNode(2)
        node3 = SNode(3)
        self.nodelist = [node0, node1, node2, node3]
        # create the tree and set its nodes
        self.tree = STree()
        self.tree.set_root(node0)
        self.tree.add_node_with_parent(node1, node0)
        self.tree.add_node_with_parent(node2, node1)
        self.tree.add_node_with_parent(node3, node1)

    def create_tree2(self, reinitialize=0):
        # Create a simple tree structure

        #   4     5     6
        #    \   /     /
        #     \ /     /
        #      2     3
        #       \   /
        #        \ /
        #         1
        #         |
        #         |
        #         0
        # create the nodes
        node0 = SNode(0)
        node1 = SNode(1)
        node2 = SNode(2)
        node3 = SNode(3)
        node4 = SNode(4)
        node5 = SNode(5)
        node6 = SNode(6)
        self.nodelist = [node0, node1, node2, node3, node4, node5, node6]
        # create the tree and set its nodes
        self.tree = STree()
        self.tree.set_root(node0)
        self.tree.add_node_with_parent(node1, node0)
        self.tree.add_node_with_parent(node2, node1)
        self.tree.add_node_with_parent(node3, node1)
        self.tree.add_node_with_parent(node4, node2)
        self.tree.add_node_with_parent(node5, node2)
        self.tree.add_node_with_parent(node6, node3)

    def test_copy_create(self):
        self.create_tree2()
        new_tree = STree(self.tree)
        assert len(new_tree) == len(self.tree)

        for n, n_ in zip(new_tree, self.tree):
            assert n.index == n_.index

        # test copy of empty tree
        empty_tree = STree()
        empty_tree_copy = STree(empty_tree)
        assert len(empty_tree_copy) == 0

    def test_getitem(self):
        self.create_tree()
        for ii in range(4):
            assert self.tree[ii].index == ii
        assert self.tree[4] == None

    def test_iter(self):
        self.create_tree()
        # full iteration
        nodeset = set([node for node in self.tree])
        assert nodeset == set(self.nodelist)
        # partial iteration
        nodeset = set([node for node in self.tree.__iter__(self.nodelist[1])])
        assert nodeset == set(self.nodelist[1:])

    def test_string_representations(self):
        self.create_tree()

        node_str = "SNode 0, Parent: None"
        node_repr = \
            "{'node index': 0, 'parent index': -1, 'content': '{}'}"
        assert str(self.tree[0]) == node_str
        assert repr(self.tree[0]) == node_repr

        tree_str = ">>> STree\n" \
            "    SNode 0, Parent: None\n" \
            "    SNode 1, Parent: 0\n" \
            "    SNode 2, Parent: 1\n" \
            "    SNode 3, Parent: 1"
        tree_repr = "['STree', "\
            "\"{'node index': 0, 'parent index': -1, 'content': '{}'}\", " \
            "\"{'node index': 1, 'parent index': 0, 'content': '{}'}\", " \
            "\"{'node index': 2, 'parent index': 1, 'content': '{}'}\", " \
            "\"{'node index': 3, 'parent index': 1, 'content': '{}'}\"]"
        assert tree_str == str(self.tree)
        assert tree_repr == repr(self.tree)

    def test_hashing(self):
        self.create_tree()

        tree_repr = "['STree', "\
            "\"{'node index': 0, 'parent index': -1, 'content': '{}'}\", " \
            "\"{'node index': 1, 'parent index': 0, 'content': '{}'}\", " \
            "\"{'node index': 2, 'parent index': 1, 'content': '{}'}\", " \
            "\"{'node index': 3, 'parent index': 1, 'content': '{}'}\"]"

        # we check whether the hashes generated with the standard hash function
        # are consistent
        assert hash(self.tree) == hash(tree_repr)
        # we check whether the hash generated with the unique_hash function is
        # correct, as this hash should be the same in every session
        assert self.tree.unique_hash() == 'd2a693df13fd87b89b4ecb4166713cb9dcd90a13743bcfc8311a3d3d75e854e9'

    def test_node_counting(self):
        self.create_tree()
        assert len(self.tree) == 4
        assert self.tree.__len__(self.tree[1]) == 3
        # test empty tree
        empty_tree = STree()
        assert len(empty_tree) == 0

    def test_get_set_nodes(self):
        self.create_tree()
        assert self.tree.nodes == self.nodelist
        with pytest.raises(AttributeError):
            self.tree.nodes = [SNode(15)]

    def test_get_set_leafs(self):
        self.create_tree()
        assert self.tree.leafs == [self.nodelist[2], self.nodelist[3]]
        with pytest.raises(AttributeError):
            self.tree.leafs = [SNode(15)]

    def test_root_leaf_check(self):
        self.create_tree()
        assert self.tree.is_root(self.nodelist[0]) == True
        assert self.tree.is_root(self.nodelist[1]) == False
        assert self.tree.is_leaf(self.nodelist[1]) == False
        assert self.tree.is_leaf(self.nodelist[2]) == True

    def test_insertion_removal(self):
        self.create_tree()
        # test node insertion
        newnode = SNode(15)
        self.tree.insert_node(newnode, self.nodelist[1], self.nodelist[2:3])
        assert newnode in self.nodelist[1].child_nodes
        assert newnode.parent_node == self.nodelist[1]
        assert self.nodelist[2] in newnode.child_nodes
        assert self.nodelist[2].parent_node == newnode
        assert self.nodelist[3] not in newnode.child_nodes
        # test rearranging indices
        self.tree.reset_indices()
        assert [node.index for node in self.tree] == list(range(5))
        # test node removal
        self.tree.remove_single_node(newnode)
        assert set(self.nodelist[1].child_nodes) == set([self.nodelist[2],
                                                         self.nodelist[3]])
        assert self.nodelist[2].parent_node == self.nodelist[1]
        assert self.nodelist[3].parent_node == self.nodelist[1]
        self.tree.reset_indices()
        assert [node.index for node in self.tree] == list(range(4))
        # limit case 1: insert and remove root
        newroot = SNode(15)
        self.tree.insert_node(newroot, None)
        assert self.tree.root == newroot
        assert newroot.child_nodes == self.nodelist[0:1]
        assert self.nodelist[0].parent_node == newroot
        assert newroot.parent_node == None
        with pytest.raises(ValueError):
            self.tree.remove_single_node(newroot)
        # add a node with a given index
        self.tree.add_node_with_parent_from_index(4, self.tree[3])
        assert isinstance(self.tree[4], SNode)
        with pytest.raises(ValueError):
            self.tree.add_node_with_parent_from_index(3, self.tree[3])
        with pytest.warns(UserWarning):
            self.tree.add_node_with_parent_from_index(5, None)
        # reinitialize original tree
        self.create_tree(reinitialize=1)

    def test_degree_order_depth_node(self):
        self.create_tree()
        assert self.tree.order_of_node(self.nodelist[0]) == -1
        assert self.tree.order_of_node(self.nodelist[1]) == 0
        assert self.tree.order_of_node(self.nodelist[2]) == 0

        assert self.tree.degree_of_node(self.nodelist[1]) == 2
        assert self.tree.degree_of_node(self.nodelist[2]) == 1

        assert self.tree.depth_of_node(self.nodelist[0]) == 0
        assert self.tree.depth_of_node(self.nodelist[1]) == 1
        assert self.tree.depth_of_node(self.nodelist[2]) == 2

    def test_paths(self):
        self.create_tree()
        # paths to root
        assert self.tree.path_to_root(self.nodelist[0]) == \
                    self.nodelist[0:1]
        assert self.tree.path_to_root(self.nodelist[2]) == \
                    self.nodelist[0:3][::-1]
        # paths from node to node
        assert self.tree.path_between_nodes(self.nodelist[2], self.nodelist[3]) == \
                    [self.nodelist[2], self.nodelist[1], self.nodelist[3]]
        assert self.tree.path_between_nodes(self.nodelist[2], self.nodelist[1]) == \
                    [self.nodelist[2], self.nodelist[1]]
        assert self.tree.path_between_nodes(self.nodelist[1], self.nodelist[2]) == \
                    [self.nodelist[1], self.nodelist[2]]
        assert self.tree.path_between_nodes(self.nodelist[2], self.nodelist[2]) == \
                    [self.nodelist[2]]
        # path from node to node in a depth-first ordering
        assert self.tree.path_between_nodes_depth_first(self.nodelist[2], self.nodelist[3]) == \
                    [self.nodelist[1], self.nodelist[2], self.nodelist[3]]
        assert self.tree.path_between_nodes_depth_first(self.nodelist[2], self.nodelist[1]) == \
                    [self.nodelist[1], self.nodelist[2]]
        assert self.tree.path_between_nodes_depth_first(self.nodelist[1], self.nodelist[2]) == \
                    [self.nodelist[1], self.nodelist[2]]
        assert self.tree.path_between_nodes_depth_first(self.nodelist[2], self.nodelist[2]) == \
                    [self.nodelist[2]]

    def test_sister_leafs(self):
        self.create_tree()
        # normal case
        bnode, sister_leafs, corresponding_children = \
                                    self.tree.sister_leafs(self.nodelist[2])
        assert bnode == self.nodelist[1]
        assert sister_leafs == self.nodelist[2:]
        assert corresponding_children == self.nodelist[2:]
        # node is bifurcation case
        bnode, sister_leafs, corresponding_children = \
                                    self.tree.sister_leafs(self.nodelist[1])
        assert bnode == self.nodelist[0]
        assert sister_leafs == [self.nodelist[1]]
        assert corresponding_children == [self.nodelist[1]]

    def test_bifurcation_search_to_root(self):
        self.create_tree()
        # normal case
        bnode, cnode = self.tree.find_bifurcation_node_to_root(self.nodelist[2])
        assert bnode == self.nodelist[1]
        assert cnode == self.nodelist[2]
        # node is bifurcation node case
        bnode, cnode = self.tree.find_bifurcation_node_to_root(self.nodelist[1])
        assert bnode == self.nodelist[0]
        assert cnode == self.nodelist[1]
        # node is root node case
        bnode, cnode = self.tree.find_bifurcation_node_to_root(self.nodelist[0])
        assert bnode == self.nodelist[0]
        assert cnode == None

    def test_nodes_in_subtree(self):
        self.create_tree()
        rn = self.tree[2]
        rr = self.tree[1]
        nodes1 = self.tree.get_nodes_in_subtree(rn, subtree_root=rr)
        assert [nn.index for nn in nodes1] == [1,2]
        nodes2 = self.tree.get_nodes_in_subtree(rn)
        assert [nn.index for nn in nodes2] == [0,1,2,3]
        nodes3 = self.tree.get_nodes_in_subtree(rn, subtree_root=rn)
        assert len(nodes3) == 1 and nodes3[0].index == 2

    def test_bifurcation_nodes(self):
        self.create_tree()
        nodes = [self.tree[3]]
        bnodes = self.tree.find_in_between_bifurcation_nodes(nodes)
        assert bnodes == [self.tree[0]]
        nodes = [self.tree[2], self.tree[3]]
        bnodes = self.tree.find_in_between_bifurcation_nodes(nodes)
        assert bnodes == [self.tree[0], self.tree[1]]
        # more complex tree
        self.create_tree2()
        nodes = [self.tree[4], self.tree[5]]
        bnodes = self.tree.find_in_between_bifurcation_nodes(nodes)
        assert bnodes == [self.tree[0], self.tree[2]]
        self.create_tree2()
        nodes = [self.tree[4], self.tree[5], self.tree[6]]
        bnodes = self.tree.find_in_between_bifurcation_nodes(nodes)
        assert bnodes == [self.tree[0], self.tree[1], self.tree[2]]



if __name__ == '__main__':
    tst = TestSTree()
    tst.test_string_representations()
    tst.test_hashing()
    tst.test_paths()
    tst.test_bifurcation_nodes()
    tst.test_degree_order_depth_node()