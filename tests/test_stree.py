from neat import STree, SNode

import pytest


class TestSTree():
    def createTree(self, reinitialize=0):
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
        self.tree.setRoot(node0)
        self.tree.addNodeWithParent(node1, node0)
        self.tree.addNodeWithParent(node2, node1)
        self.tree.addNodeWithParent(node3, node1)

    def createTree2(self, reinitialize=0):
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
        self.tree.setRoot(node0)
        self.tree.addNodeWithParent(node1, node0)
        self.tree.addNodeWithParent(node2, node1)
        self.tree.addNodeWithParent(node3, node1)
        self.tree.addNodeWithParent(node4, node2)
        self.tree.addNodeWithParent(node5, node2)
        self.tree.addNodeWithParent(node6, node3)

    def testGetitem(self):
        self.createTree()
        for ii in range(4):
            assert self.tree[ii].index == ii
        assert self.tree[4] == None

    def testIter(self):
        self.createTree()
        # full iteration
        nodeset = set([node for node in self.tree])
        assert nodeset == set(self.nodelist)
        # partial iteration
        nodeset = set([node for node in self.tree.__iter__(self.nodelist[1])])
        assert nodeset == set(self.nodelist[1:])

    def testNodeCounting(self):
        self.createTree()
        assert len(self.tree) == 4
        assert self.tree.__len__(self.tree[1]) == 3
        # test empty tree
        empty_tree = STree()
        assert len(empty_tree) == 0

    def testGetSetNodes(self):
        self.createTree()
        assert self.tree.nodes == self.nodelist
        with pytest.raises(AttributeError):
            self.tree.nodes = [SNode(15)]

    def testGetSetLeafs(self):
        self.createTree()
        assert self.tree.leafs == [self.nodelist[2], self.nodelist[3]]
        with pytest.raises(AttributeError):
            self.tree.leafs = [SNode(15)]

    def testRootLeafCheck(self):
        self.createTree()
        assert self.tree.isRoot(self.nodelist[0]) == True
        assert self.tree.isRoot(self.nodelist[1]) == False
        assert self.tree.isLeaf(self.nodelist[1]) == False
        assert self.tree.isLeaf(self.nodelist[2]) == True

    def testInsertionRemoval(self):
        self.createTree()
        # test node insertion
        newnode = SNode(15)
        self.tree.insertNode(newnode, self.nodelist[1], self.nodelist[2:3])
        assert newnode in self.nodelist[1].child_nodes
        assert newnode.parent_node == self.nodelist[1]
        assert self.nodelist[2] in newnode.child_nodes
        assert self.nodelist[2].parent_node == newnode
        assert self.nodelist[3] not in newnode.child_nodes
        # test rearranging indices
        self.tree.resetIndices()
        assert [node.index for node in self.tree] == list(range(5))
        # test node removal
        self.tree.removeSingleNode(newnode)
        assert set(self.nodelist[1].child_nodes) == set([self.nodelist[2],
                                                         self.nodelist[3]])
        assert self.nodelist[2].parent_node == self.nodelist[1]
        assert self.nodelist[3].parent_node == self.nodelist[1]
        self.tree.resetIndices()
        assert [node.index for node in self.tree] == list(range(4))
        # limit case 1: insert and remove root
        newroot = SNode(15)
        self.tree.insertNode(newroot, None)
        assert self.tree.root == newroot
        assert newroot.child_nodes == self.nodelist[0:1]
        assert self.nodelist[0].parent_node == newroot
        assert newroot.parent_node == None
        with pytest.raises(ValueError):
            self.tree.removeSingleNode(newroot)
        # add a node with a given index
        self.tree.addNodeWithParentFromIndex(4, self.tree[3])
        assert isinstance(self.tree[4], SNode)
        with pytest.raises(ValueError):
            self.tree.addNodeWithParentFromIndex(3, self.tree[3])
        with pytest.warns(UserWarning):
            self.tree.addNodeWithParentFromIndex(5, None)
        # reinitialize original tree
        self.createTree(reinitialize=1)

    def testDegreeOrderDepthNode(self):
        self.createTree()
        assert self.tree.orderOfNode(self.nodelist[0]) == -1
        assert self.tree.orderOfNode(self.nodelist[1]) == 0
        assert self.tree.orderOfNode(self.nodelist[2]) == 0

        assert self.tree.degreeOfNode(self.nodelist[1]) == 2
        assert self.tree.degreeOfNode(self.nodelist[2]) == 1

        assert self.tree.depthOfNode(self.nodelist[0]) == 0
        assert self.tree.depthOfNode(self.nodelist[1]) == 1
        assert self.tree.depthOfNode(self.nodelist[2]) == 2

    def testPaths(self):
        self.createTree()
        # paths to root
        assert self.tree.pathToRoot(self.nodelist[0]) == \
                    self.nodelist[0:1]
        assert self.tree.pathToRoot(self.nodelist[2]) == \
                    self.nodelist[0:3][::-1]
        # paths from node to node
        assert self.tree.pathBetweenNodes(self.nodelist[2], self.nodelist[3]) == \
                    [self.nodelist[2], self.nodelist[1], self.nodelist[3]]
        assert self.tree.pathBetweenNodes(self.nodelist[2], self.nodelist[1]) == \
                    [self.nodelist[2], self.nodelist[1]]
        assert self.tree.pathBetweenNodes(self.nodelist[1], self.nodelist[2]) == \
                    [self.nodelist[1], self.nodelist[2]]
        assert self.tree.pathBetweenNodes(self.nodelist[2], self.nodelist[2]) == \
                    [self.nodelist[2]]
        # path from node to node in a depth-first ordering
        assert self.tree.pathBetweenNodesDepthFirst(self.nodelist[2], self.nodelist[3]) == \
                    [self.nodelist[1], self.nodelist[2], self.nodelist[3]]
        assert self.tree.pathBetweenNodesDepthFirst(self.nodelist[2], self.nodelist[1]) == \
                    [self.nodelist[1], self.nodelist[2]]
        assert self.tree.pathBetweenNodesDepthFirst(self.nodelist[1], self.nodelist[2]) == \
                    [self.nodelist[1], self.nodelist[2]]
        assert self.tree.pathBetweenNodesDepthFirst(self.nodelist[2], self.nodelist[2]) == \
                    [self.nodelist[2]]

    def testSisterLeafs(self):
        self.createTree()
        # normal case
        bnode, sisterLeafs, corresponding_children = \
                                    self.tree.sisterLeafs(self.nodelist[2])
        assert bnode == self.nodelist[1]
        assert sisterLeafs == self.nodelist[2:]
        assert corresponding_children == self.nodelist[2:]
        # node is bifurcation case
        bnode, sisterLeafs, corresponding_children = \
                                    self.tree.sisterLeafs(self.nodelist[1])
        assert bnode == self.nodelist[0]
        assert sisterLeafs == [self.nodelist[1]]
        assert corresponding_children == [self.nodelist[1]]

    def testUpBifurcationSearch(self):
        self.createTree()
        # normal case
        bnode, cnode = self.tree.upBifurcationNode(self.nodelist[2])
        assert bnode == self.nodelist[1]
        assert cnode == self.nodelist[2]
        # node is bifurcation node case
        bnode, cnode = self.tree.upBifurcationNode(self.nodelist[1])
        assert bnode == self.nodelist[0]
        assert cnode == self.nodelist[1]
        # node is root node case
        bnode, cnode = self.tree.upBifurcationNode(self.nodelist[0])
        assert bnode == self.nodelist[0]
        assert cnode == None

    def testNodesInSubtree(self):
        self.createTree()
        rn = self.tree[2]
        rr = self.tree[1]
        nodes1 = self.tree.getNodesInSubtree(rn, subtree_root=rr)
        assert [nn.index for nn in nodes1] == [1,2]
        nodes2 = self.tree.getNodesInSubtree(rn)
        assert [nn.index for nn in nodes2] == [0,1,2,3]
        nodes3 = self.tree.getNodesInSubtree(rn, subtree_root=rn)
        assert len(nodes3) == 1 and nodes3[0].index == 2

    def testBifurcationNodes(self):
        self.createTree()
        nodes = [self.tree[3]]
        bnodes = self.tree.getBifurcationNodes(nodes)
        assert bnodes == [self.tree[0]]
        nodes = [self.tree[2], self.tree[3]]
        bnodes = self.tree.getBifurcationNodes(nodes)
        assert bnodes == [self.tree[0], self.tree[1]]
        # more complex tree
        self.createTree2()
        nodes = [self.tree[4], self.tree[5]]
        bnodes = self.tree.getBifurcationNodes(nodes)
        assert bnodes == [self.tree[0], self.tree[2]]
        self.createTree2()
        nodes = [self.tree[4], self.tree[5], self.tree[6]]
        bnodes = self.tree.getBifurcationNodes(nodes)
        assert bnodes == [self.tree[0], self.tree[1], self.tree[2]]



if __name__ == '__main__':
    tst = TestSTree()
    tst.testPaths()
    tst.testBifurcationNodes()
    tst.testDegreeOrderDepthNode()