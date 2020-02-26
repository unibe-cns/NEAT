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
from ..channels import channelcollection, concmechs


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

    def addCurrent(self, channel_name, g_max, e_rev=None, channel_storage=None):
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
        if e_rev is None:
            e_rev = channelcollection.E_REV_DICT[channel_name]
        self.currents[channel_name] = (g_max, e_rev)
        if channel_name != 'L' and \
           channel_storage is not None and \
           channel_name not in channel_storage:
            channel_storage[channel_name] = \
                eval('channelcollection.' + channel_name + '()')

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

    def getCurrent(self, channel_name, channel_storage=None):
        """
        Returns an ``::class::neat.channels.ionchannels.IonChannel`` object. If
        `channel_storage` is given,

        Parameters
        ----------
        channel_name: string
            the name of the ion channel
        channel_storage: dict of ionchannels (optional)
            keys are the names of the ion channels, and values the channel
            instances
        """
        if channel_storage is not None:
            try:
                return channel_storage[channel_name]
            except (KeyError):
                chan = eval('channelcollection.' + channel_name + '()')
                channel_storage[channel_name] = chan
                return chan
        else:
            return eval('channelcollection.' + channel_name + '()')

    def setEEq(self, e_eq):
        """
        Set the equilibrium potential at the node.

        Parameters
        ----------
        e_eq: float
            the equilibrium potential (mV)
        """
        self.e_eq = e_eq

    def fitLeakCurrent(self, e_eq_target=-75., tau_m_target=10., channel_storage=None):
        gsum = 0.
        i_eq = 0.
        for channel_name in set(self.currents.keys()) - set('L'):
            g, e = self.currents[channel_name]
            # create the ionchannel object
            channel = self.getCurrent(channel_name, channel_storage=channel_storage)
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

    def getGTot(self, v=None, channel_storage=None):
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
            channel = self.getCurrent(channel_name, channel_storage=channel_storage)
            g_tot += g * channel.computePOpen(v)

        return g_tot

    def setGTot(self, illegal):
        raise AttributeError("`g_tot` is a read-only attribute, set the leak " + \
                             "conductance by calling ``func:addCurrent`` with " + \
                             " \'L\' as `channel_name`")

    g_tot = property(getGTot, setGTot)

    def asPassiveMembrane(self, v=None, channel_storage=None):
        v = self.e_eq if v is None else v
        g_l = self.getGTot(v=v, channel_storage=channel_storage)
        t_m = self.c_m / g_l * 1e3 # time scale in ms
        self.currents = {'L': (0., 0.)} # dummy values
        self.fitLeakCurrent(e_eq_target=v, tau_m_target=t_m,
                                channel_storage=channel_storage)

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
            node.asPassiveMembrane(channel_storage=self.channel_storage)

    @morphtree.originalTreetypeDecorator
    def setEEq(self, e_eq):
        """
        Set the equilibrium potentials throughout the tree. Can be iterable,
        than the `len(e_eq)` is assumed to be the same as `len(self)`

        Parameters
        ----------
        e_eq: float or iterable of floats
            The equilibrium potentials
        """
        if not hasattr(e_eq, '__iter__'):
            e_eq = e_eq * np.ones(len(self))
        for e, node in zip(e_eq, self): node.setEEq(e)

    @morphtree.originalTreetypeDecorator
    def setPhysiology(self, c_m_distr, r_a_distr, g_s_distr=None, node_arg=None):
        """
        Set specifice membrane capacitance, axial resistance and (optionally)
        static point-like shunt conductances in the tree

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
            # get the membrane capacitance
            if isinstance(c_m_distr, float):
                c_m = c_m_distr
            elif isinstance(c_m_distr, dict):
                c_m = c_m_distr[node.index]
            elif hasattr(c_m_distr, '__call__'):
                d2s = self.pathLength({'node': node.index, 'x': .5}, (1., 0.5))
                c_m = c_m_distr(d2s)
            else:
                raise TypeError('`c_m_distr` argument should be a float, dict ' + \
                                'or a callable')
            # get the axial resistance
            if isinstance(r_a_distr, float):
                r_a = r_a_distr
            elif isinstance(r_a_distr, dict):
                r_a = r_a_distr[node.index]
            elif hasattr(r_a_distr, '__call__'):
                d2s = self.pathLength({'node': node.index, 'x': .5}, (1., 0.5))
                r_a = r_a_distr(d2s)
            else:
                raise TypeError('`r_a_distr` argument should be a float, dict ' + \
                                'or a callable')
            if g_s_distr is not None:
                # get the static shunt conductances
                if isinstance(g_s_distr, float):
                    g_s = g_s_distr
                elif isinstance(g_s_distr, dict):
                    g_s = g_s_distr[node.index]
                elif hasattr(g_s_distr, '__call__'):
                    d2s = self.pathLength({'node': node.index, 'x': .5}, (1., 0.5))
                    g_s = g_s_distr(d2s)
                else:
                    raise TypeError('`g_s_distr` argument should be a float, dict ' + \
                                    'or a callable')
            else:
                g_s = 0.
            node.setPhysiology(c_m, r_a, g_s)

    @morphtree.originalTreetypeDecorator
    def addCurrent(self, channel_name, g_max_distr, e_rev=None, node_arg=None):
        """
        Adds a channel to the morphology.

        Parameters
        ----------
            channel_name: string
                The name of the channel type
            g_max_distr: float, dict or :func:`float -> float`
                If float, the maximal conductance is set to this value for all
                the nodes specified in `node_arg`. If it is a function, the input
                must specify the distance from the soma (micron) and the output
                the ion channel density (uS/cm^2) at that distance. If it is a
                dict, keys are the node indices and values the ion channel
                densities (uS/cm^2).
            node_arg: optional
                see documentation of :func:`MorphTree._convertNodeArgToNodes`.
                Defaults to None
        """
        # add the ion channel to the nodes
        for node in self._convertNodeArgToNodes(node_arg):
            # get the ion channel conductance
            if isinstance(g_max_distr, float):
                g_max = g_max_distr
            elif isinstance(g_max_distr, dict):
                g_max = g_max_distr[node.index]
            elif hasattr(g_max_distr, '__call__'):
                d2s = self.pathLength({'node': node.index, 'x': .5}, (1., 0.5))
                g_max = g_max_distr(d2s)
            else:
                raise TypeError('`g_max_distr` argument should be a float, dict \
                                or a callable')
            node.addCurrent(channel_name, g_max, e_rev=e_rev,
                            channel_storage=self.channel_storage)

    @morphtree.originalTreetypeDecorator
    def getChannelsInTree(self, store=False):
        """
        Returns list of strings of all channel names in the tree

        Parameters
        ----------
        store: bool, optional (default `False`)
            if `True`, stores all channels in the tree in `self.channel_storage`
            if they are not already there

        Returns
        -------
        list of string
            the channel names
        """
        channel_names = list(set([c_name for node in self for c_name in node.currents if c_name != 'L']))
        if store:
            for c_name in channel_names:
                self.channel_storage[c_name] = self.root.getCurrent(c_name)
        return channel_names

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

    def fitLeakCurrent(self, e_eq_target=-75., tau_m_target=10., node_arg=None):
        """
        Fits the leak current to fix equilibrium potential and membrane time-
        scale.

        Parameters
        ----------
            e_eq_target: float
                The target reversal potential (mV). Defaults to -75 mV.
            tau_m_target: float
                The target membrane time-scale (ms). Defaults to 10 ms.
            node_arg:
                see documentation of :func:`MorphTree._convertNodeArgToNodes`.
                Defaults to None
        """
        assert tau_m_target > 0.
        for node in self._convertNodeArgToNodes(node_arg):
            node.fitLeakCurrent(e_eq_target=e_eq_target, tau_m_target=tau_m_target,
                                channel_storage=self.channel_storage)

    def computeEquilibirumPotential(self):
        pass

    def setCompTree(self, eps=1e-8):
        comp_nodes = [n for n in self if n.g_shunt > eps]
        for node in self.nodes[1:]:
            pnode = node.parent_node
            # check if parameters are the same
            # if not( np.abs(node.r_a - pnode.r_a) < eps and \
            #     np.abs(node.c_m - pnode.c_m) < eps and \
            #     np.abs(node.R - pnode.R) < eps and \
            #     set(node.currents.keys()) == set(pnode.currents.keys()) and
            #     not sum([sum([np.abs(curr[0] - pnode.currents[key][0]),
            #                   np.abs(curr[1] - pnode.currents[key][1])])
            #              for key, curr in node.currents.iteritems()])):
            if not( np.abs(node.r_a - pnode.r_a) < eps * np.max([node.r_a, pnode.r_a]) and \
                np.abs(node.c_m - pnode.c_m) < eps * np.max([node.c_m, pnode.c_m]) and \
                np.abs(node.R - pnode.R) < eps * np.max([node.R, pnode.R]) and \
                set(node.currents.keys()) == set(pnode.currents.keys()) and
                not sum([sum([np.abs(curr[0] - pnode.currents[key][0]) > eps * np.max([np.abs(curr[0]), np.abs(pnode.currents[key][0])]),
                              np.abs(curr[1] - pnode.currents[key][1]) > eps * np.max([np.abs(curr[1]), np.abs(pnode.currents[key][1])])])
                         for key, curr in node.currents.items()])):
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






