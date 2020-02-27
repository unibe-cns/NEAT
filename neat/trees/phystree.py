"""
File contains:

    - :class:`PhysNode`
    - :class:`PhysTree`

Author: W. Wybo
"""

import numpy as np

import warnings

from . import morphtree
from .morphtree import MorphNode, MorphTree
from ..channels import concmechs, ionchannels


class PhysNode(MorphNode):
    def __init__(self, index, p3d=None,
                       c_m=1., r_a=100*1e-6, g_shunt=0., e_eq=-75.):
        super(PhysNode, self).__init__(index, p3d)
        self.currents = {} #{name: (g_max (uS/cm^2), e_rev (mV))}
        self.concmechs = {}
        # biophysical parameters
        self.c_m = c_m # uF/cm^2
        self.r_a = r_a # MOhm*cm
        self.g_shunt = g_shunt # uS
        self.e_eq = e_eq

    def setPhysiology(self, c_m, r_a, g_shunt=0.):
        """
        Set the physiological parameters of the current

        Parameters
        ----------
        c_m: float
            the membrance capacitance (uF/cm^2)
        r_a: float
            the axial current (MOhm*cm)
        g_shunt: float
            A point-like shunt, located at x=1 on the node. Defaults to 0.
        """
        self.c_m = c_m # uF/cm^2
        self.r_a = r_a # MOhm*cm
        self.g_shunt = g_shunt

    def _addCurrent(self, channel_name, g_max, e_rev):
        """
        Add an ion channel current at this node. ('L' as `channel_name`
        signifies the leak current)

        Parameters
        ----------
        channel_name: string
            the name of the current
        g_max: float
            the conductance of the current (uS/cm^2)
        e_rev: float
            the reversal potential of the current (mV)
        """
        self.currents[channel_name] = [g_max, e_rev]

    def addConcMech(self, ion, params={}):
        """
        Add a concentration mechanism at this node.

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism (only used for NEURON model)
        """
        if set(params.keys()) == {'gamma', 'tau'}:
            self.concmechs[ion] = concmechs.ExpConcMech(ion,
                                        params['tau'], params['gamma'])
        else:
            warnings.warn('These parameters do not match any NEAT concentration ' + \
                          'mechanism, no concentration mechanism has been added', UserWarning)

    def setEEq(self, e_eq):
        """
        Set the equilibrium potential at the node.

        Parameters
        ----------
        e_eq: float
            the equilibrium potential (mV)
        """
        self.e_eq = e_eq

    def fitLeakCurrent(self, channel_storage, e_eq_target=-75., tau_m_target=10.):
        """
        """
        gsum = 0.
        i_eq = 0.

        for channel_name in set(self.currents.keys()) - set('L'):
            g, e = self.currents[channel_name]
            # get the ionchannel object
            channel = channel_storage[channel_name]
            # compute channel conductance and current
            p_open = channel.computePOpen(e_eq_target)
            g_chan = g * p_open
            i_chan = g_chan * (e - e_eq_target)
            gsum += g_chan
            i_eq += i_chan

        if self.c_m / (tau_m_target*1e-3) < gsum:
            warnings.warn('Membrane time scale is chosen larger than ' + \
                          'possible, adding small leak conductance')
            tau_m_target = self.c_m / (gsum + 20.)
        else:
            tau_m_target *= 1e-3
        g_l = self.c_m / tau_m_target - gsum
        e_l = e_eq_target - i_eq / g_l
        self.currents['L'] = (g_l, e_l)
        self.e_eq = e_eq_target

    def getGTot(self, channel_storage, v=None):
        """
        Get the total conductance of the membrane at a steady state given voltage,
        if nothing is given, the equilibrium potential is used to compute membrane
        conductance.

        Parameters
        ----------
            v: float (optional, defaults to `self.e_eq`)
                the potential (in mV) at which to compute the membrane conductance

        Returns
        -------
            float
                the total conductance of the membrane (uS / cm^2)
        """
        v = self.e_eq if v is None else v
        g_tot = self.currents['L'][0]
        for channel_name in set(self.currents.keys()) - set('L'):
            g, e = self.currents[channel_name]
            # create the ionchannel object
            channel = channel_storage[channel_name]
            g_tot += g * channel.computePOpen(v)

        return g_tot

    def asPassiveMembrane(self, channel_storage, v=None):
        v = self.e_eq if v is None else v
        g_l = self.getGTot(channel_storage, v=v)
        t_m = self.c_m / g_l * 1e3 # time scale in ms
        self.currents = {'L': (0., 0.)} # dummy values
        self.fitLeakCurrent(channel_storage, e_eq_target=v, tau_m_target=t_m)

    def __str__(self, with_parent=False, with_children=False):
        node_string = super(PhysNode, self).__str__()
        if self.parent_node is not None:
            node_string += ', Parent: ' + super(PhysNode, self.parent_node).__str__()
        node_string += ' --- (r_a = ' + str(self.r_a) + ' MOhm*cm, ' + \
                       ', '.join(['g_' + cname + ' = ' + str(cpar[0]) + ' uS/cm^2' \
                            for cname, cpar in self.currents.items()]) + \
                       ', c_m = ' + str(self.c_m) + ' uF/cm^2)'
        return node_string


class PhysTree(MorphTree):
    def __init__(self, file_n=None, types=[1,3,4]):
        super(PhysTree, self).__init__(file_n=file_n, types=types)
        # set basic physiology parameters (c_m = 1.0 uF/cm^2 and
        # r_a = 0.0001 MOhm*cm)
        for node in self:
            node.setPhysiology(1.0, 100./1e6)
        self.channel_storage = {}

    def createCorrespondingNode(self, node_index, p3d=None,
                                      c_m=1., r_a=100*1e-6, g_shunt=0., e_eq=-75.):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return PhysNode(node_index, p3d=p3d)

    @morphtree.originalTreetypeDecorator
    def asPassiveMembrane(self, node_arg=None):
        """
        Makes the membrane act as a passive membrane (for the nodes in
        ``node_arg``), channels are assumed to add a conductance of
        g_max * p_open to the membrane conductance, where p_open for each node
        is evaluated at the equilibrium potential stored in that node

        Parameters
        ----------
        node_arg: optional
                see documentation of :func:`MorphTree._convertNodeArgToNodes`.
                Defaults to None. The nodes for which the membrane is set to
                passive
        """
        for node in self._convertNodeArgToNodes(node_arg):
            node.asPassiveMembrane(self.channel_storage)

    def _distr2Float(self, distr, node, argname=''):
        if isinstance(distr, float):
            val = distr
        elif isinstance(distr, dict):
            val = distr[node.index]
        elif hasattr(distr, '__call__'):
            d2s = self.pathLength({'node': node.index, 'x': .5}, (1., 0.5))
            val = distr(d2s)
        else:
            raise TypeError(argname + ' argument should be a float, dict ' + \
                            'or a callable')
        return val

    @morphtree.originalTreetypeDecorator
    def setEEq(self, e_eq_distr, node_arg=None):
        """
        Set the equilibrium potentials throughout the tree

        Parameters
        ----------
        e_eq_distr: float, dict or :func:`float -> float`
            The equilibrium potentials [mV]
        """
        for node in self._convertNodeArgToNodes(node_arg):
            e = self._distr2Float(e_eq_distr, node, argname='`e_eq_distr`')
            node.setEEq(e)

    @morphtree.originalTreetypeDecorator
    def setPhysiology(self, c_m_distr, r_a_distr, g_s_distr=None, node_arg=None):
        """
        Set specifice membrane capacitance, axial resistance and (optionally)
        static point-like shunt conductances in the tree. Capacitance is stored
        at each node as the attribute 'c_m' (uF/cm2) and axial resistance as the
        attribute 'r_a' (MOhm*cm)

        Parameters
        ----------
        c_m_distr: float, dict or :func:`float -> float`
            specific membrance capacitance
        r_a_distr: float, dict or :func:`float -> float`
            axial resistance
        g_s_distr: float, dict, :func:`float -> float` or None (optional, default
            is `None`)
            point like shunt conductances (placed at `(node.index, 1.)` for the
            nodes in ``node_arg``). By default no shunt conductances are added
        node_arg: optional
            see documentation of :func:`MorphTree._convertNodeArgToNodes`.
            Defaults to None
        """
        for node in self._convertNodeArgToNodes(node_arg):
            c_m = self._distr2Float(c_m_distr, node, argname='`c_m_distr`')
            r_a = self._distr2Float(r_a_distr, node, argname='`r_a_distr`')
            g_s = self._distr2Float(g_s_distr, node, argname='`g_s_distr`') if \
                  g_s_distr is not None else 0.
            node.setPhysiology(c_m, r_a, g_s)

    @morphtree.originalTreetypeDecorator
    def setLeakCurrent(self, g_l_distr, e_l_distr, node_arg=None):
        """
        Set the parameters of the leak current. At each node, leak is stored
        under the attribute `node.currents['L']` at a tuple `(g_l, e_l)` with
        `g_l` the conductance [uS/cm^2] and `e_l` the reversal [mV]

        parameters:
        ----------
        g_l_distr: float, dict or :func:`float -> float`
            If float, the leak conductance is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma (micron) and the output
            the leak conductance [uS/cm^2] at that distance. If it is a
            dict, keys are the node indices and values the ion leak
            conductances [uS/cm^2].
        e_l_distr: float, dict or :func:`float -> float`
            If float, the reversal [mV] is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma [um] and the output
            the reversal at that distance. If it is a
            dict, keys are the node indices and values the ion reversals.
        node_arg: optional
            see documentation of :func:`MorphTree._convertNodeArgToNodes`.
            Defaults to None
        """
        for node in self._convertNodeArgToNodes(node_arg):
            g_l = self._distr2Float(g_l_distr, node, argname='`g_l_distr`')
            e_l = self._distr2Float(e_l_distr, node, argname='`e_l_distr`')
            node._addCurrent('L', g_l, e_l)

    @morphtree.originalTreetypeDecorator
    def addCurrent(self, channel, g_max_distr, e_rev_distr, node_arg=None):
        """
        Adds a channel to the morphology. At each node, the channel is stored
        under the attribute `node.currents[channel.__class__.__name__]` as a
        tuple `(g_max, e_rev)` with `g_max` the maximal conductance [uS/cm^2]
        and `e_rev` the reversal [mV]

        Parameters
        ----------
        channel_name: :class:`IonChannel`
            The ion channel
        g_max_distr: float, dict or :func:`float -> float`
            If float, the maximal conductance is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma (micron) and the output
            the ion channel density (uS/cm^2) at that distance. If it is a
            dict, keys are the node indices and values the ion channel
            densities (uS/cm^2).
        e_rev_distr: float, dict or :func:`float -> float`
            If float, the reversal (mV) is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma (micron) and the output
            the reversal at that distance. If it is a
            dict, keys are the node indices and values the ion reversals.
        node_arg: optional
            see documentation of :func:`MorphTree._convertNodeArgToNodes`.
            Defaults to None
        """
        if not isinstance(channel, ionchannels.IonChannel):
            raise IOError('`channel` argmument needs to be of class `neat.IonChannel`')

        channel_name = channel.__class__.__name__
        self.channel_storage[channel_name] = channel
        # add the ion channel to the nodes
        for node in self._convertNodeArgToNodes(node_arg):
            g_max = self._distr2Float(g_max_distr, node, argname='`g_max_distr`')
            e_rev = self._distr2Float(e_rev_distr, node, argname='`e_rev_distr`')
            node._addCurrent(channel_name, g_max, e_rev)

    @morphtree.originalTreetypeDecorator

    def getChannelsInTree(self):
        """
        Returns list of strings of all channel names in the tree

        Returns
        -------
        list of string
            the channel names
        """
        return self.channel_storage.keys()

    @morphtree.originalTreetypeDecorator
    def addConcMech(self, ion, params={}, node_arg=None):
        """
        Add a concentration mechanism to the tree

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism (only used for NEURON model)
        node_arg:
            see documentation of :func:`MorphTree._convertNodeArgToNodes`.
            Defaults to None
        """
        for node in self._convertNodeArgToNodes(node_arg):
            node.addConcMech(ion, params=params)

    @morphtree.originalTreetypeDecorator
    def fitLeakCurrent(self, e_eq_target_distr, tau_m_target_distr, node_arg=None):
        """
        Fits the leak current to fix equilibrium potential and membrane time-
        scale.

        !!! Should only be called after all ion channels have been added !!!

        Parameters
        ----------
        e_eq_target_distr: float, dict or :func:`float -> float`
            The target reversal potential (mV). If float, the target reversal is
            set to this value for all the nodes specified in `node_arg`. If it
            is a function, the input must specify the distance from the soma (um)
            and the output the target reversal at that distance. If it is a
            dict, keys are the node indices and values the target reversals
        tau_m_target_distr: float, dict or :func:`float -> float`
            The target membrane time-scale (ms). If float, the target time-scale is
            set to this value for all the nodes specified in `node_arg`. If it
            is a function, the input must specify the distance from the soma (um)
            and the output the target time-scale at that distance. If it is a
            dict, keys are the node indices and values the target time-scales
        node_arg:
            see documentation of :func:`MorphTree._convertNodeArgToNodes`.
            Defaults to None
        """
        for node in self._convertNodeArgToNodes(node_arg):
            e_eq_target = self._distr2Float(e_eq_target_distr, node, argname='`g_max_distr`')
            tau_m_target = self._distr2Float(tau_m_target_distr, node, argname='`e_rev_distr`')
            assert tau_m_target > 0.
            node.fitLeakCurrent(e_eq_target=e_eq_target, tau_m_target=tau_m_target,
                                channel_storage=self.channel_storage)

    def _evaluateCompCriteria(self, node, eps=1e-8, rbool=False):
        """
        Return ``True`` if relative difference in any physiological parameters
        between node and child node is larger than margin ``eps``

        Parameters
        ----------
        node: ::class::`MorphNode`
            node that is compared to parent node
        eps: float (optional, default ``1e-8``)
            the margin

        return
        ------
        bool
        """
        rbool = super(PhysTree, self)._evaluateCompCriteria(node, eps=eps, rbool=rbool)

        if not rbool:
            cnode = node.child_nodes[0]
            rbool = np.abs(node.r_a - cnode.r_a) > eps * np.max([node.r_a, cnode.r_a])
        if not rbool:
            rbool = np.abs(node.c_m - cnode.c_m) > eps * np.max([node.c_m, cnode.c_m])
        if not rbool:
            rbool = set(node.currents.keys()) != set(cnode.currents.keys())
        if not rbool:
            for chan_name, channel in node.currents.items():
                if not rbool:
                    rbool = np.abs(channel[0] - cnode.currents[chan_name][0]) > eps * \
                             np.max([np.abs(channel[0]), np.abs(cnode.currents[chan_name][0])])
                if not rbool:
                    rbool = np.abs(channel[1] - cnode.currents[chan_name][1]) > eps * \
                             np.max([np.abs(channel[1]), np.abs(cnode.currents[chan_name][1])])
        if not rbool:
            rbool = node.g_shunt > 0.001*eps

        return rbool

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






