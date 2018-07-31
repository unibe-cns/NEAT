"""
File contains:

    - :class:`GreensNode`
    - :class:`SomaGreensNode`
    - :class:`GreensTree`

Author: W. Wybo
"""

import numpy as np

import copy

import morphtree
from morphtree import MorphLoc
from phystree import PhysNode, PhysTree
from neat.channels import channelcollection



class GreensNode(PhysNode):
    def __init__(self, index, p3d):
        super(GreensNode, self).__init__(index, p3d)

    def rescaleLengthRadius(self):
        self.R_ = self.R * 1e-4 # convert to cm
        self.L_ = self.L * 1e-4 # convert to cm

    def calcMembraneImpedance(self, freqs, channel_storage=None):
        '''
        Compute the impedance of the membrane at the node

        Parameters
        ----------
        freqs: `np.ndarray` (``dtype=complex``, ``ndim=1``)
            The frequencies at which the impedance is to be evaluated
        channel_storage: dict of ion channels (optional)
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized

        Returns
        -------
        `np.ndarray` (``dtype=complex``, ``ndim=1``)
            The membrane impedance
        '''
        g_m_aux = self.c_m * freqs + self.currents['L'][0]
        for channel_name in set(self.currents.keys()) - set('L'):
            g, e = self.currents[channel_name]
            # create the ionchannel object
            if channel_storage is not None:
                channel = channel_storage[channel_name]
            else:
                channel = eval('channelcollection.' + channel_name + '()')
            # add channel contribution to membrane impedance
            g_m_aux -= g * (e - self.e_eq) * \
                       channel.computeLinear(self.e_eq, freqs)
            g_m_aux += g * channel.computePOpen(self.e_eq)

        return 1. / (2. * np.pi * self.R_ * g_m_aux)

    def setImpedance(self, freqs):
        self.counter = 0
        self.z_m = self.calcMembraneImpedance(freqs)
        self.z_a = self.r_a / (np.pi * self.R_**2)
        self.gamma = np.sqrt(self.z_a / self.z_m)
        self.z_c = self.z_a / self.gamma

    def setImpedanceDistal(self):
        '''
        Set the boundary condition at the distal end of the segment
        '''
        if len(self.child_nodes) == 0:
            self.z_distal = np.infty*np.ones(len(self.z_m))
        else:
            self.z_distal = 1. / np.sum([1. / cnode.collapseBranchToRoot() \
                                         for cnode in self.child_nodes], 0)

    def setImpedanceProximal(self):
        '''
        Set the boundary condition at the proximal end of the segment
        '''
        # child nodes of parent node without the current node
        sister_nodes = copy.copy(self.parent_node.child_nodes[:])
        sister_nodes.remove(self)
        # compute the impedance
        val = 0.
        if self.parent_node is not None:
            val += 1. / self.parent_node.collapseBranchToLeaf()
        for snode in sister_nodes:
            val += 1. / snode.collapseBranchToRoot()
        self.z_proximal = 1. / val

    def collapseBranchToLeaf(self):
        return self.z_c * (self.z_proximal * np.cosh(self.gamma * self.L_) + \
                           self.z_c * np.sinh(self.gamma * self.L_)) / \
                          (self.z_proximal * np.sinh(self.gamma * self.L_) +
                           self.z_c * np.cosh(self.gamma * self.L_))

    def collapseBranchToRoot(self):
        # if self.z_distal[0] == np.infty:
        #     zr = self.z_c / tanh(self.gamma*self.L_)
        # else:
        zr = self.z_c * (np.cosh(self.gamma * self.L_) +
                         self.z_c / self.z_distal * np.sinh(self.gamma * self.L_)) / \
                        (np.sinh(self.gamma * self.L_) +
                         self.z_c / self.z_distal * np.cosh(self.gamma * self.L_))
        return zr

    def setImpedanceArrays(self):
        self.gammaL = self.gamma * self.L_
        self.z_cp = self.z_c / self.z_proximal
        self.z_cd = self.z_c / self.z_distal
        self.wrongskian = np.cosh(self.gammaL) / self.z_c * \
                           (self.z_cp + self.z_cd + \
                            (1. + self.z_cp * self.z_cd) * np.tanh(self.gammaL))
        self.z_00 = (self.z_cd * np.sinh(self.gammaL) + np.cosh(self.gammaL)) / \
                    self.wrongskian
        self.z_11 = (self.z_cp * np.sinh(self.gammaL) + np.cosh(self.gammaL)) / \
                    self.wrongskian
        self.z_01 = 1. / self.wrongskian

    def calcZF(self, x1, x2):
        if x1 <  1e-3 and x2 < 1e-3:
            return self.z_00
        elif x1 > 1.-1e-3 and x2 > 1.-1e-3:
            return self.z_11
        elif (x1 < 1e-3 and x2 > 1.-1e-3) or (x1 > 1.-1e-3 and x2 < 1e-3):
            return self.z_01
        elif x1 < x2:
            return (self.z_cp * np.sinh(self.gammaL*x1) + np.cosh(self.gammaL*x1)) * \
                   (self.z_cd * np.sinh(self.gammaL*(1.-x2)) + np.cosh(self.gammaL*(1.-x2))) / \
                   self.wrongskian
        else:
            return (self.z_cp * np.sinh(self.gammaL*x2) + np.cosh(self.gammaL*x2)) * \
                   (self.z_cd * np.sinh(self.gammaL*(1.-x1)) + np.cosh(self.gammaL*(1.-x1))) / \
                   self.wrongskian


# G = np.cosh(gamma*x) * one_minus_tanhtanh(gamma*L, gamma*D) * \
#                                 (tanh(gamma*x) + z_0/z_c) * (z_c*tanh(gamma*(L-D)) + z_1) / \
#                                 ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L)) * np.cosh(gamma*D)

# G = np.cosh(gamma*D) * one_minus_tanhtanh(gamma*L, gamma*x) * \
#                                 (tanh(gamma*D) + z_0/z_c) * (z_c*tanh(gamma*(L-x)) + z_1) / \
#                                 ((z_0+z_1)/z_c + (1. + (z_0/z_c)*(z_1/z_c))*tanh(gamma*L)) * np.cosh(gamma*x)


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
    def calcMembraneImpedance(self, freqs):
        z_m = super(SomaGreensNode, self).calcMembraneImpedance(freqs)
        # rescale for soma surface instead of cylinder radius
        return z_m / (2. * self.R_)

    def setImpedance(self, freqs):
        self.counter = 0
        self.z_soma = self.calcMembraneImpedance(freqs)

    def collapseBranchToLeaf(self):
        return self.z_soma

    def setImpedanceArrays(self):
        val = 1. / self.z_soma
        for node in self.child_nodes:
            val += 1. / node.collapseBranchToRoot()
        self.z_in = 1. / val

    def calcZF(self, x1, x2):
        return self.z_in


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

    def createCorrespondingNode(self, node_index, p3d=None):
        '''
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
        node_index: `int`
            index of the new node
        '''
        if node_index == 1:
            return SomaGreensNode(node_index, p3d)
        else:
            return GreensNode(node_index, p3d)

    @morphtree.computationalTreetypeDecorator
    def setImpedance(self, freqs, pprint=False):
        '''
        Set the boundary impedances for each node in the tree

        Parameters
        ----------
        freqs: `np.ndarray` (``dtype=complex``, ``ndim=1``)
            frequencies at which the impedances will be evaluated
        pprint: bool (default ``False``)
            whether or not to print info on the progression of the algorithm

        '''
        self.freqs = freqs
        # set the node specific impedances
        for node in self:
            node.rescaleLengthRadius()
            node.setImpedance(freqs)
        # recursion
        self._impedanceFromLeaf(self.leafs[0], self.leafs[1:], pprint=pprint)
        self._impedanceFromRoot(self.root)
        # clean
        for node in self:
            node.counter = 0
            node.setImpedanceArrays()

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
                self._impedanceFromLeaf(leafs[0], leafs[1:], pprint=pprint)

    def _impedanceFromRoot(self, node):
        if node != self.root:
            node.setImpedanceProximal()
        for cnode in node.child_nodes:
            self._impedanceFromRoot(cnode)

    @morphtree.computationalTreetypeDecorator
    def calcZF(self, loc1, loc2):
        '''
        Computes the transfer impedance between two locations for all frequencies
        in `self.freqs`.

        Parameters
        ----------
        loc1, loc2: dict, tuples or `:class:MorphLoc`
            Two locations between which the transfer impedance is computed

        Returns
        -------
        nd.ndarray (dtype = complex, ndim = 1)
            The transfer impedance as a function of frequency
        '''
        # cast to morphlocs
        loc1 = MorphLoc(loc1, self)
        loc2 = MorphLoc(loc2, self)
        # the path between the nodes
        path = self.pathBetweenNodes(self[loc1['node']], self[loc2['node']])
        # compute the kernel
        z_f = np.ones_like(self.freqs)
        if len(path) == 1:
            # both locations are on same node
            z_f *= path[0].calcZF(loc1['x'], loc2['x'])
        else:
            # different cases whether path goes to or from root
            if path[1] == self[loc1['node']].parent_node:
                z_f *= path[0].calcZF(loc1['x'], 0.)
            else:
                z_f *= path[0].calcZF(loc1['x'], 1.)
                z_f /= path[0].calcZF(1., 1.)
            if path[-2] == self[loc2['node']].parent_node:
                z_f *= path[-1].calcZF(loc2['x'], 0.)
            else:
                z_f *= path[-1].calcZF(loc2['x'], 1.)
                z_f /= path[-1].calcZF(1., 1.)
            # nodes within the path
            ll = 1
            for node in path[1:-1]:
                z_f /= node.calcZF(1., 1.)
                if path[ll-1] not in node.child_nodes or \
                   path[ll+1] not in node.child_nodes:
                    z_f *= node.calcZF(0., 1.)
                ll += 1

        return z_f

    @morphtree.computationalTreetypeDecorator
    def calcImpedanceMatrix(self, locarg):
        '''
        Computes the impedance matrix of a given set of locations for each
        frequency stored in `self.freqs`.

        Parameters
        ----------
        locarg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            impedance matrix is evaluated, if ``string``, specifies the
            name under which a set of location is stored

        Returns
        -------
        `np.ndarray` (``dtype = complex``, ``ndim = 3``)
            the impedance matrix, first dimension corresponds to the
            frequency, second and third dimensions contain the impedance
            matrix at that frequency
        '''
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
        elif isinstance(locarg, str):
            locs = self.getLocs(locarg)
        else:
            raise IOError('`locarg` should be list of locs or string')
        z_mat = np.zeros((len(self.freqs), len(locs), len(locs)), dtype=complex)
        for ii, loc0 in enumerate(locs):
            jj = 0
            while jj < ii:
                loc1 = locs[jj]
                z_f = self.calcZF(loc0, loc1)
                z_mat[:,ii,jj] = z_f
                z_mat[:,jj,ii] = z_f
                jj += 1
            loc1 = locs[ii]
            z_f = self.calcZF(loc0, loc1)
            z_mat[:,ii,ii] = z_f

        return z_mat




