"""
File contains:

    - :class:`GreensNode`
    - :class:`SomaGreensNode`
    - :class:`GreensTree`

Author: W. Wybo
"""

import numpy as np

from phystree import PhysNode, PhysTree


class GreensNode(PhysNode):
    def __init__(self, index, p3d):
        super(GreensNode, self).__init__(index, p3d)
        self.R_ = self.R * 1e-4 # convert to cm
        self.L_ = self.L * 1e-4 # convert to cm

    def calcMembranceImpedance(self, freqs):
        # TODO
        return None

    def setImpedance(self, freqs):
        self.counter = 0
        self.z_m = self.calcMembraneImpedance(freqs)
        self.z_a = self.r_a / (np.pi * self.R_**2)
        self.gamma = np.sqrt(self.z_a / self.z_m)
        self.z_c = self.z_a / self.gammas

    def setImpedanceDistal(self):
        '''
        Set the boundary condition at the distal end of the segment
        '''
        if len(self.child_nodes) == 0:
            self.z_distal = np.infty*np.ones(len(self.freqs))
        else:
            self.z_distal = 1. / np.sum([1. / cnode.collapseBranch() \
                                         for cnode in self.child_nodes])

    def setImpedanceProximal(self):
        '''
        Set the boundary condition at the proximal end of the segment
        '''
        # child nodes of parent node without the current node
        sister_nodes = self.pnode.child_nodes[:].remove(self)
        # compute the impedance
        val = 0.
        if self.parent_node is not None:
            val += 1. / self.parent_node.collapseBranchToLeaf()
        for snode in self.sister_nodes:
            val += 1. / snode.collapseBranchToRoot()
        self.z_proximal = 1. / val

    def collapseBranchToLeaf(self):
        return self.z_c * (self.z_proximal * np.cosh(self.gamma * self.L_) + \
                           self.z_c * np.sinh(self.gamma * self.L_)) / \
                          (self.z_proximal * np.sinh(self.gamma * self.L_) +
                           self.z_c * np.cosh(self.gamma * self.L_))

    def collapseBranchToRoot(self):
        if self.z_distal[0] == np.infty:
            zr = self.z_c / tanh(self.gamma*self.L_)
        else:
            zr = self.z_c * (self.z_distal * np.cosh(self.gamma * self.L_) +
                             self.z_c * np.sinh(self.gamma * self.L_)) / \
                            (self.z_distal * np.sinh(self.gamma * self.L_) +
                             self.z_c * np.cosh(self.gamma * self.L_))
        return zr

    def calcZF(self, x1, x2):
        if x1 > x2:
            return None
        else:
            return None


    # def set_greensfunctions(self):
    #     # transfer impedance between ends of segment
    #     if self.z_1[0] == np.infty:
    #         self.z_trans = 1. / ((1./self.z_0) * np.cosh(self.gamma*self.length) + \
    #                     1./self.z_c * np.sinh(self.gamma*self.length))
    #         self.z_in = self.collapse_branch0()
    #     else:
    #         self.z_trans = 1. / ((1./self.z_0 + 1./self.z_1) * np.cosh(self.gamma*self.length) + \
    #                     (self.z_c/(self.z_0*self.z_1) + 1./self.z_c) * np.sinh(self.gamma*self.length))
    #         self.z_in = 1./(1./self.z_1 + 1./self.collapse_branch0())


class SomaGreensNode(GreensNode):
    def calcMembranceImpedance(self, freqs):
        # TODO
        return None

    def setImpedance(self, freqs):
        self.z_soma = calcMembranceImpedance(freqs)

    def collapseBranchToLeaf():
        return self.z_soma


class GreensTree(PhysTree):
    '''
    Class that computes the Green's function in the Fourrier domain of a given
    neuronal morphology (Koch, 1985). This three defines a special
    :class:`SomaGreensNode` as a derived class from :class:`GreensNode` as some
    functions required for Green's function calculation are different and thus
    overwritten.

    The calculation proceeds on the computational tree (see docstring of
    :class:`MorphNode`). Thus it makes no sense to look for Green's function
    related quantities in the original tree.
    '''
    def __init__(self, file_n=None, types=[1,3,4]):
        super(GreensTree, self).__init__(file_n=file_n, types=types)
        self.freqs = None

    def createCorrespondingNode(self, node_index, p3d):
        '''
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        '''
        if node_index == 1:
            return SomaGreensNode(node_index, p3d)
        else:
            return GreensNode(node_index, p3d)

    @computationalTreetypeDecorator
    def setImpedance(self, freqs):
        self.freqs = freqs
        # set the node specific impedances
        for node in self: node.setImpedance(freqs)
        # recursion
        self._impedanceFromLeaf(leafs[0], leafs[1:], pprint=pprint)
        self._impedanceFromRoot(self.root)
        # clean
        for node in self: node.counter = 0

    def _impedanceFromLeaf(self, node, leafs, pprint=False):
        if pprint:
            print 'Forward sweep: ' + str(node)
        pnode = node.parent_node
        # log how many times recursion has passed at node
        if not self.isLeaf(node):
            node.counter += 1
        # if the number of childnodes of node is equal to the amount of times
        # the recursion has passed node, the distal impedance can be set. Otherwise
        # we start a new recursion at another leaf.
        if node.counter == len(node.child_nodes):
            node.setImpedanceDistal()
            if not self.isRoot(node):
                self._impedanceFromLeaf(pnode, leafs, pprint=pprint)
        elif len(leafs) > 0:
                self._SOVFromLeaf(leafs[0], leafs[1:], pprint=pprint)

    def _impedanceFromRoot(self, node):
        if node != self.root:
            node.setImpedanceProximal()
        for cnode in node.child_nodes:
            self._impedanceFromRoot(node)

    def calcZF(self, loc1, loc2):
        # cast to morphlocs
        loc1 = self.morphLoc(loc1, self)
        loc2 = self.morphLoc(loc2, self)
        # the path between the nodes
        path = self.pathBetweenNodes(self[loc1['node']], self[loc2['node']])
        # compuate the kernel
        z_f = np.ones_like(self.freqs)
        if len(path) == 1:
            # both locations are on same node
            zf *= path[0].calcZF(loc1['x'], loc2['x'])
        else:
            # diferent cases whether path goes to or from root
            if path[1] == self[loc1['node']].parent_node:
                zf *= path[0].calcZF(loc1['x'], 0.)
            else:
                zf *= path[0].calcZF(loc1['x'], 1.)
            if path[-2] == self[loc2['node']].parent_node:
                zf *= path[0].calcZF(loc1['x'], 0.)
            else:
                zf *= path[0].calcZF(loc1['x'], 1.)




