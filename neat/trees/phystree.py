"""
File contains:

    - :class:`PhysNode`
    - :class:`PhysTree`

Author: W. Wybo
"""

import numpy as np

import warnings

from neat.channels import ionchannels as ionc

import morphtree
from morphtree import MorphNode, MorphTree


class PhysNode(MorphNode):
    def __init__(self, index, p3d):
        super(PhysNode, self).__init__(index, p3d)
        self.currents = {} #{name: (g_max (uS/cm^2), e_rev (mV))}

    def setPhysiology(self, c_m, r_a, g_shunt=0.):
        '''
        Set the physiological parameters of the current

        Parameters
        ----------
            c_m: float
                the membrance capacitance (uF/cm^2)
            r_a: float
                the axial current (MOhm*cm)
            g_shunt: float
                A point-like shunt, located at x=1 on the node. Defaults to 0.
        '''
        self.c_m = c_m # uF/cm^2
        self.r_a = r_a # MOhm*cm 
        self.g_shunt = g_shunt

    def addCurrent(self, current_type, g_max, e_rev):
        '''
        Add an ion channel current at this node. ('L' as `current_type`
        signifies the leak current)

        Parameters
        ----------
            current_type: string
                the name of the current
            g_max: float
                the conductance of the current (uS/cm^2)
            e_rev: float
                the reversal potential of the current (mV)
        '''
        self.currents[current_type] = (g_max, e_rev)

    def _setEEq(self, e_eq):
        '''
        Set the equilibrium potential at the node.

        Parameters
        ----------
            e_eq: float
                the equilibrium potential (mV)
        '''
        self.e_eq = e_eq

    def fitLeakCurrent(self, e_eq_target=-75., tau_m_target=10.):   
        gsum = 0.
        i_eq = 0.
        for channel_name in set(self.currents.keys()) - set('L'):
            # create the ionchannel object
            channel = eval('ionc.' + channel_name + '( \
                                    g=gs[\'' + channel_name + '\'], \
                                    e=es[\'' + channel_name + '\'], \
                                    V0=params[0])')
            i_chan = - channel.g0 * (e_eq_target - channel.e)
            gsum += channel.g0
            i_eq += i_chan
        if self.c_m / (tau_m_target*1e-3) < gsum:
            warnings.warn('Membrane time scale is chosen largen than \
                           possible, decreasing membrane time scale')
            tau_m_target = cm / (gsum+300.) 
        else:
            tau_m_target *= 1e-3
        g_l = self.c_m / tau_m_target - gsum
        e_l = e_eq_target - i_eq / g_l
        self.currents['L'] = (g_l, e_l)
        self.e_eq = e_eq_target

    def getGTot(self):
        if self.currents.keys() == ['L']:
            return self.currents['L'][0]
        else:
            raise Exception('Not implemented yet')


class PhysTree(MorphTree):
    def __init__(self, file_n=None, types=[1,3,4]):
        super(PhysTree, self).__init__(file_n=file_n, types=types)
        # set basic physiology parameters (c_m = 1.0 uF/cm^2 and 
        # r_a = 0.0001 MOhm*cm)
        for node in self:
            node.setPhysiology(1.0, 100./1e6)

    def createCorrespondingNode(self, node_index, p3d):
        '''
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        '''
        return PhysNode(node_index, p3d)

    def addCurrent(self, current_type, g_max_distr, e_rev, node_arg=None,
                        fill_tree=0, eval_type='pas'):
        '''
        Adds a channel to the morphology.

        Parameters
        ----------
            current_type: string
                The name of the channel type
            g_max_distr: float, dict or :func:`float -> float`
                If float, the maximal conductance is set to this value for all
                the nodes specified in `node_arg`. If it is a function, the input
                must specify the distance from the soma (micron) and the output 
                the ion channel density (uS/cm^2) at that distance. If it is a
                dict, keys are the node indices and values the ion channel 
                densities (uS/cm^2).
            node_arg:
                see documentation of :func:`MorphTree._convertNodeArgToNodes`.
                Defaults to None
            eval_type: {'pas', 'lin'}
                Specifies the way the ion channel is evaluated in calculations. 
                'pas' means that only the passive conductance at the local 
                equilibrium potential is incorporated, whereas 'lin' means that 
                the full semi-active channel is evaluated.
        '''
        for node in self._convertNodeArgToNodes(node_arg):
            # get the ion channel conductance
            if type(g_max_distr) == float:
                g_max = g_max_distr
            elif type(g_max_distr) == dict:
                g_max = g_max_distr[node.index]
            elif hasattr('__call__'):
                d2s = self.pathLength({'node': node.index, 'x': .5}, (1., 0.5))
                g_max = g_max_distr(d2s)
            else:
                raise TypeError('`g_max_distr` argument should be a float, dict \
                                or a callable')
            node.addCurrent(current_type, g_max, e_rev)

    def fitLeakCurrent(self, e_eq_target=-75., tau_m_target=10.):
        '''
        Fits the leak current to fix equilibrium potential and membrane time-
        scale.

        Parameters
        ----------
            e_eq_target: float
                The target reversal potential (mV). Defaults to -75 mV.
            tau_m_target: float
                The target membrane time-scale (ms). Defaults to 10 ms.
        '''
        assert tau_m_target > 0.
        for node in self:
            node.fitLeakCurrent(e_eq_target=e_eq_target, 
                                  tau_m_target=tau_m_target)

    def computeEquilibirumPotential(self):
        pass

    def setCompTree(self, eps=1e-8):
        comp_nodes = []
        for node in self.nodes[1:]:
            pnode = node.parent_node
            # check if parameters are the same
            if not( np.abs(node.r_a - pnode.r_a) < eps and \
                np.abs(node.c_m - pnode.c_m) < eps and \
                np.abs(node.R - pnode.R) < eps and \
                set(node.currents.keys()) == set(pnode.currents.keys()) and
                not sum([sum([np.abs(curr[0] - pnode.currents[key][0]) > eps,
                              np.abs(curr[1] - pnode.currents[key][1]) > eps]) 
                         for key, curr in node.currents.iteritems()])):
                comp_nodes.append(pnode)
        super(PhysTree, self).setCompTree(compnodes=comp_nodes)

    # @morphtree.originalTreetypeDecorator
    # def _calcFdMatrix(self, dx=10.):
    #     matdict = {}
    #     locs = [{'node': 1, 'x': 0.}]
    #     # set the first element
    #     soma = self.tree.root
    #     matdict[(0,0)] = 4.0*np.pi*soma.R**2 * soma.G
    #     # recursion
    #     cnodes = root.getChildNodes()[2:]
    #     numel_l = [1]
    #     for cnode in cnodes:
    #         if not is_changenode(cnode):
    #             cnode = find_previous_changenode(cnode)[0]
    #         self._fdMatrixFromRoot(cnode, root, 0, numel_l, locs, matdict, dx=dx)
    #     # create the matrix
    #     FDmat = np.zeros((len(locs), len(locs)))
    #     for ind in matdict:
    #         FDmat[ind] = matdict[ind]

    #     return FDmat, locs # caution, not the reduced locs yet

    # def _fdMatrixFromRoot(self, node, pnode, ibranch, numel_l, locs, matdict, dx=10.*1e-4):
    #     numel = numel_l[0]
    #     # distance between the two nodes and radius of the cylinder
    #     radius *= node.R*1e-4; length *= node.L*1e-4
    #     num = np.around(length/dx)
    #     xvals = np.linspace(0.,1.,max(num+1,2))
    #     dx_ = xvals[1]*length
    #     # set the first element
    #     matdict[(ibranch,numel)] = - np.pi*radius**2 / (node.r_a*dx_)
    #     matdict[(ibranch,ibranch)] += np.pi*radius**2 / (node.r_a*dx_)
    #     matdict[(numel,numel)] = 2.*np.pi*radius**2 / (node.r_a*dx_)
    #     matdict[(numel,ibranch)] = - np.pi*radius**2 / (node.r_a*dx_)
    #     locs.append({'node': node._index, 'x': xvals[1]})
    #     # set the other elements
    #     if len(xvals) > 2:
    #         i = 0; j = 0
    #         if len(xvals) > 3:
    #             for x in xvals[2:-1]:
    #                 j = i+1
    #                 matdict[(numel+i,numel+j)] = - np.pi*radius**2 / (node.r_a*dx_)
    #                 matdict[(numel+j,numel+j)] = 2. * np.pi*radius**2 / (node.r_a*dx_)
    #                                            # + 2.*np.pi*radius*dx_*node.G
    #                 matdict[(numel+j,numel+i)] = - np.pi*radius**2 / (node.r_a*dx_)
    #                 locs.append({'node': node._index, 'x': x})
    #                 i += 1
    #         # set the last element
    #         j = i+1
    #         matdict[(numel+i,numel+j)] = - np.pi*radius**2 / (node.r_a*dx_)
    #         matdict[(numel+j,numel+j)] = np.pi*radius**2 / (node.r_a*dx_) 
    #         matdict[(numel+j,numel+i)] = - np.pi*radius**2 / (node.r_a*dx_)
    #         locs.append({'node': node._index, 'x': 1.})
    #     numel_l[0] = numel+len(xvals)-1
    #     # assert numel_l[0] == len(locs)
    #     # if node is leaf, then implement other bc
    #     if len(xvals) > 2:
    #         ibranch = numel+j
    #     else:
    #         ibranch = numel
    #     # move on the further elements
    #     for cnode in node.child_nodes:
    #         self._fdMatrixFromRoot(cnode, node, ibranch, numel_l, locs, matdict, dx=dx)






