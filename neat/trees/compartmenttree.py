"""
File contains:

    - `CompartmentNode`
    - `CompartmentTree`

Author: W. Wybo
"""


import numpy as np
import scipy.linalg as la
import scipy.optimize as so
import sympy as sp

from .stree import SNode, STree
from ..channels import channelcollection
from ..tools import kernelextraction as ke

import copy
import warnings
import itertools
from operator import mul
from functools import reduce


class CompartmentNode(SNode):
    """
    Implements a node for `CompartmentTree`

    Attributes
    ----------
    ca: float
        capacitance of the compartment (uF)
    g_l: float
        leak conductance at the compartment (uS)
    g_c: float
        Coupling conductance of compartment with parent compartment (uS).
        Ignore if node is the root
    e_eq: float
        equilibrium potential at the compartment
    currents: dict {str: [g_bar, e_rev]}
        dictionary with as keys the channel names and as elements lists of length
        two with contain the maximal conductance (uS) and the channels'
        reversal potential in (mV)
    concmechs: dict {str: `neat.channels.concmechs.ConcMech`}
        dictionary with as keys the ion names and as values the concentration
        mechanisms governing the concentration of each ion channel
    expansion_points: dict {str: np.ndarray}
        dictionary with as keys the channel names and as elements the state
        variables of the ion channel around which to compute the linearizations
    """
    def __init__(self, index, loc_ind=None, ca=1., g_c=0., g_l=1e-2, e_eq=-75.):
        super().__init__(index)
        # location index this node corresponds to
        self._loc_ind = loc_ind
        # compartment params
        self.ca = ca   # capacitance (uF)
        self.g_c = g_c # coupling conductance (uS)
        # self.g_l = g_l # leak conductance (uS)
        self.e_eq = e_eq # equilibrium potential (mV)
        self.currents = {'L': [g_l, e_eq]} # ion channel conductance (uS) and reversals (mV)
        self.concmechs = {}
        self.expansion_points = {}

    def setLocInd(self, loc_ind):
        self._loc_ind = loc_ind

    def getLocInd(self):
        if self._loc_ind is None:
            raise AttributeError("`self.loc_ind` is undefined, this node has " + \
                                 "not been associated with a location")
        else:
            return self._loc_ind

    loc_ind = property(getLocInd, setLocInd)

    def __str__(self, with_parent=False, with_children=False):
        node_string = super().__str__()
        if self.parent_node is not None:
            node_string += ', Parent: ' + super().__str__()
        node_string += ' --- (g_c = %.12f uS, '%self.g_c + \
                       ', '.join(['g_' + cname + ' = %.12f uS'%cpar[0] \
                            for cname, cpar in self.currents.items()]) + \
                       ', c = %.12f uF)'%self.ca
        return node_string

    def _addCurrent(self, channel_name, e_rev):
        """
        Add an ion channel current at this node. ('L' as `channel_name`
        signifies the leak current)

        Parameters
        ----------
        channel_name: string
            the name of the current
        e_rev: float
            the reversal potential of the current (mV)
        """
        self.currents[channel_name] = [0., e_rev]

    def addConcMech(self, ion, params=None):
        """
        Add a concentration mechanism at this node.

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism (only used for NEURON model)
        """
        if params is None:
            params = {}
        if set(params.keys()) == {'gamma', 'tau'}:
            self.concmechs[ion] = concmechs.ExpConcMech(ion,
                                        params['tau'], params['gamma'])
        else:
            warnings.warn('These parameters do not match any NEAT concentration ' + \
                          'mechanism, no concentration mechanism has been added', UserWarning)

    def setExpansionPoint(self, channel_name, statevar):
        """
        Set the choice for the state variables of the ion channel around which
        to linearize.

        Note that when adding an ion channel to the node, the default expansion
        point setting is to linearize around the asymptotic values for the state
        variables at the equilibrium potential store in `self.e_eq`.
        Hence, this function only needs to be called to change that setting.

        Parameters
        ----------
        channel_name: string
            the name of the ion channel
        statevar: dict of float
            values of the state variable expansion point
        """
        if statevar is None:
            statevar = {}
        self.expansion_points[channel_name] = statevar

    def getExpansionPoint(self, channel_name):
        try:
            return self.expansion_points[channel_name]
        except KeyError:
            self.expansion_points[channel_name] = {}
            return self.expansion_points[channel_name]

    def calcMembraneConductanceTerms(self, channel_storage,
                freqs=0., v=None, channel_names=None):
        """
        Contribution of linearized ion channel to conductance matrix

        Parameters
        ----------
        channel_storage: dict of ion channels
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        freqs: np.ndarray (ndim = 1, dtype = complex or float) or float or complex
            The frequencies at which the impedance terms are to be evaluated
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the ion channels that have to be included in the
            conductance term

        Returns
        -------
        dict of np.ndarray or float or complex
            Each entry in the dict is of the same type as ``freqs`` and is the
            conductance term of a channel
        """
        if channel_names is None: channel_names = list(self.currents.keys())
        if v is None: v = self.e_eq

        cond_terms = {}
        if 'L' in channel_names:
            cond_terms['L'] = 1. # leak conductance has 1 as prefactor
        for channel_name in set(channel_names) - set('L'):
            e = self.currents[channel_name][1]
            # get the ionchannel object
            channel = channel_storage[channel_name]
            # check if needs to be computed around expansion point
            sv = self.getExpansionPoint(channel_name)
            # add linearized channel contribution to membrane conductance
            cond_terms[channel_name] = - channel.computeLinSum(v, freqs, e, **sv)

        return cond_terms

    def calcMembraneConcentrationTerms(self, ion, channel_storage,
                freqs=0., v=None, channel_names=None):
        """
        Contribution of linearized concentration dependence to conductance matrix

        Parameters
        ----------
        ion: str
            The ion for which the concentration terms are to be calculated
        channel_storage: dict of ion channels
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        freqs: np.ndarray (ndim = 1, dtype = complex or float) or float or complex
            The frequencies at which the impedance terms are to be evaluated
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the ion channels that have to be included in the
            conductance term

        Returns
        -------
        dict of np.ndarray or float or complex
            Each entry in the dict is of the same type as ``freqs`` and is the
            conductance term of a channel
        """
        if channel_names is None: channel_names = list(self.currents.keys())
        if v is None: v = self.e_eq

        conc_write_channels = np.zeros_like(freqs)
        conc_read_channels  = np.zeros_like(freqs)
        for channel_name, (g, e) in self.currents.items():
            if channel_name in channel_names and channel_name != 'L':
                channel = channel_storage[channel_name]
                # check if needs to be computed around expansion point
                sv = self.getExpansionPoint(channel_name)
                # if the channel adds to ion channel current, add it here
                if channel.ion == ion:
                    conc_write_channels += \
                    g * channel.computeLinSum(v, freqs, e, **sv)
                # if channel reads the ion channel current, add it here
                if ion in channel.concentrations:
                    conc_read_channels -= \
                    g * channel.computeLinConc(v, freqs, e, ion, **sv)

        return conc_write_channels * \
               conc_read_channels * \
               self.concmechs[ion].computeLin(freqs)

    def getGTot(self, channel_storage,
                      v=None, channel_names=None, p_open_channels=None):
        """
        Compute the total conductance of a set of channels evaluated at a given
        voltage

        Parameters
        ----------
        channel_storage: dict {str: `neat.IonChannel`}
            Dictionary of all ion channels on the `neat.CompartmentTree`
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the channel that have to be included in the calculation
        p_open_channels: dict {str: float}, optional
            The open probalities of the channels. Custom set of open
            probabilities. Overwrites both `self.expansion_point` and `v`.
            Defaults to `None`.

        Returns
        -------
        float: the total conductance
        """
        if channel_names is None: channel_names = list(self.currents.keys())
        if v is None: v = self.e_eq

        # compute total conductance around `self.e_eq`
        g_tot = self.currents['L'][0] if 'L' in channel_names else 0.
        for channel_name in channel_names:
            if channel_name != 'L':
                g, e = self.currents[channel_name]
                # create the ionchannel object
                channel = channel_storage[channel_name]
                # check if needs to be computed around expansion point
                sv = self.getExpansionPoint(channel_name)
                # open probability
                if p_open_channels is None:
                    p_o = channel.computePOpen(v, **sv)
                else:
                    p_o = p_open_channels[channel_name]
                # add to total conductance
                g_tot += g * p_o

        return g_tot

    def getITot(self, channel_storage,
                      v=None, channel_names=None, p_open_channels={}):
        """
        Compute the total current of a set of channels evaluated at a given
        voltage

        Parameters
        ----------
        channel_storage: dict {str: `neat.IonChannel`}
            Dictionary of all ion channels on the `neat.CompartmentTree`
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the channel that have to be included in the calculation
        p_open_channels: dict {str: float}, optional
            The open probalities of the channels. Custom set of open
            probabilities. Overwrites probabilities given by both
            `self.expansion_point` and `v`. Defaults to `None`.

        Returns
        -------
        float: the total conductance
        """

        if channel_names is None: channel_names = list(self.currents.keys())
        if v is None: v = self.e_eq

        # compute total conductance around `self.e_eq`
        i_tot = self.currents['L'][0] * (v - self.currents['L'][1]) if 'L' in channel_names else 0.
        for channel_name in channel_names:
            if channel_name != 'L':
                g, e = self.currents[channel_name]
                if channel_name not in p_open_channels:
                    # create the ionchannel object
                    channel = channel_storage[channel_name]
                    # check if needs to be computed around expansion point
                    sv = self.getExpansionPoint(channel_name)
                    i_tot += g * channel.computePOpen(v, **sv) * (v - e)
                else:
                    i_tot += g * p_open_channels[channel_name] * (v - e)

        return i_tot

    def getDrive(self, channel_name, v=None, channel_storage=None):
        v = self.e_eq if v is None else v
        _, e = self.currents[channel_name]
        # create the ionchannel object
        channel = self.getCurrent(channel_name, channel_storage=channel_storage)
        sv = self.expansion_points[channel_name]
        return channel.computePOpen(v, statevars=sv) * (v - e)

    def getDynamicDrive(self, channel_name, p_open, v):
        assert p_open.shape == v.shape
        _, e = self.currents[channel_name]
        return p_open * (v - e)

    def getDynamicDrive_(self, channel_name, v, dt, channel_storage=None):
        # assert p_open.shape == v.shape
        _, e = self.currents[channel_name]
        if channel_storage is not None:
            channel = channel_storage[channel_name]
        else:
            channel = eval('channelcollection.' + channel_name + '()')
        # storage
        p_open = np.zeros_like(v)
        # initialize
        sv_inf_prev = channel.computeVarInf(v[0])
        tau_prev = channel.computeTauInf(v[0])
        sv = sv_inf_prev
        p_open[0] = channel.computePOpen(v[0], statevars=sv)
        for tt in range(1,len(v)):
            sv_inf = channel.computeVarInf(v[tt])
            tau = channel.computeTauInf(v[tt])
            # sv_inf_aux = (sv_inf + sv_inf_prev) / 2.
            f_aux  = -2. / (tau + tau_prev)
            h_prev = sv_inf_prev / tau_prev
            h_now  = sv_inf / tau
            # sv[:,:,tt] = (sv[:,:,tt-1] + dt * sv_inf_aux / tau_aux) / (1. + dt / tau_aux)
            p0_aux = np.exp(f_aux * dt)
            p1_aux = (1. - p0_aux) / (f_aux**2 * dt)
            p2_aux = p0_aux / f_aux + p1_aux
            p3_aux = -1. / f_aux - p1_aux
            # next step sv
            sv = p0_aux * sv + p2_aux * h_prev + p3_aux * h_now
            # store for next step
            sv_inf_prev = sv_inf
            tau_prev = tau
            # store open probability
            p_open[tt] = channel.computePOpen(v[tt], statevars=sv)

        return p_open * (v - e)

    def getDynamicI(self, channel_name, p_open, v):
        assert p_open.shape == v.shape
        g, e = self.currents[channel_name]
        return g * p_open * (v - e)


class CompartmentTree(STree):
    """
    Abstract tree that implements physiological parameters for reduced
    compartmental models. Also implements the matrix algebra to fit physiological
    parameters to impedance matrices
    """
    def __init__(self, root=None):
        super().__init__(root=root)
        self.channel_storage = {}
        # for fitting the model
        self.resetFitData()

    def _createCorrespondingNode(self, index, ca=1., g_c=0., g_l=1e-2):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
        node_index: int
            index of the new node
        """
        return CompartmentNode(index, ca=ca, g_c=g_c, g_l=g_l)

    def setEEq(self, e_eq, indexing='locs'):
        """
        Set the equilibrium potential at all nodes on the compartment tree

        Parameters
        ----------
        e_eq: float or np.array of floats
            The equilibrium potential(s). If a float, the same potential is set
            at every node. If a numpy array, must have the same length as `self`
        indexing: 'locs' or 'tree'
            The ordering of the equilibrium potentials. If 'locs', assumes the
            equilibrium potentials are in the order of the list of locations
            to which the tree is fitted. If 'tree', assumes they are in the order
            of which nodes appear during iteration
        """
        if isinstance(e_eq, float) or isinstance(e_eq, int):
            e_eq = e_eq * np.ones(len(self), dtype=float)
        elif indexing == 'locs':
            e_eq = self._permuteToTree(np.array(e_eq))
        for ii, node in enumerate(self): node.e_eq = e_eq[ii]

    def getEEq(self, indexing='locs'):
        """
        Get the equilibrium potentials at each node.

        Parameters
        ----------
        indexing: 'locs' or 'tree'
            The ordering of the returned array. If 'locs', returns the array
            in the order of the list of locations to which the tree is fitted.
            If 'tree', returns the array in the order in which nodes appear
            during iteration
        """
        e_eq = np.array([node.e_eq for node in self])
        if indexing == 'locs':
            e_eq = self._permuteToLocs(e_eq)
        return e_eq

    def setExpansionPoints(self, expansion_points):
        """
        Set the choice for the state variables of the ion channel around which
        to linearize.

        Note that when adding an ion channel to the tree, the default expansion
        point setting is to linearize around the asymptotic values for the state
        variables at the equilibrium potential store in `self.e_eq`.
        Hence, this function only needs to be called to change that setting.

        Parameters
        ----------
        expansion_points: dict {`channel_name`: ``None`` or dict}
            dictionary with as keys `channel_name` the name of the ion channel
            and as value its expansion point
        """
        to_tree_inds = self._permuteToTreeInds()
        for channel_name, expansion_point in expansion_points.items():
            # if one set of state variables, set throughout neuron
            if expansion_point is None:
                eps = [None for _ in self]
            else:
                eps = [{} for _ in self]
                for svar, exp_p in expansion_point.items():
                    if isinstance(exp_p, float):
                        for ep in eps:
                            ep[svar] = exp_p
                    else:
                        assert len(exp_p) == len(self)
                        for ep, ep_ in zip(eps, exp_p[to_tree_inds]):
                            ep[svar] = ep_

            for node, ep in zip(self, eps):
                node.setExpansionPoint(channel_name, ep)


            # elif isinstance(expansion_point, np.ndarray):
            #     if expansion_point.ndim == 3:
            #         svs = np.array(expansion_point)
            #     elif expansion_point.ndim == 2:
            #         svs = np.array([expansion_point for _ in self])
            # for node, sv in zip(self, svs[to_tree_inds]):
            #     node.setExpansionPoint(channel_name, statevar=sv)

    def removeExpansionPoints(self):
        for node in self:
            node.expansion_points = {}

    def fitEL(self):
        """
        Fit the leak reversal potential to obtain the stored equilibirum potentials
        as resting membrane potential
        """
        e_l_0 = self.getEEq(indexing='tree')
        # compute the solutions
        fun = self._fun(e_l_0)
        jac = self._jac(e_l_0)
        e_l = np.linalg.solve(jac, -fun + np.dot(jac, e_l_0))
        # set the leak reversals
        for ii, node in enumerate(self):
            node.currents['L'][1] = e_l[ii]

    def _fun(self, e_l):
        # set the leak reversal potentials
        for ii, node in enumerate(self):
            node.currents['L'][1] = e_l[ii]
        # compute the function values (currents)
        fun_vals = np.zeros(len(self))
        for ii, node in enumerate(self):
            fun_vals[ii] += node.getITot(self.channel_storage)
            # add the parent node coupling term
            if node.parent_node is not None:
                fun_vals[ii] += node.g_c * (node.e_eq - node.parent_node.e_eq)
            # add the child node coupling terms
            for cnode in node.child_nodes:
                fun_vals[ii] += cnode.g_c * (node.e_eq - cnode.e_eq)
        return fun_vals

    def _jac(self, e_l):
        for ii, node in enumerate(self):
            node.currents['L'][1] = e_l[ii]
        jac_vals = np.array([-node.currents['L'][0] for node in self])
        return np.diag(jac_vals)

    def addCurrent(self, channel, e_rev):
        """
        Add an ion channel current to the tree

        Parameters
        ----------
        channel_name: string
            The name of the channel type
        e_rev: float
            The reversal potential of the ion channel [mV]
        """

        channel_name = channel.__class__.__name__
        self.channel_storage[channel_name] = channel
        for ii, node in enumerate(self):
            node._addCurrent(channel_name, e_rev)

    def addConcMech(self, ion, params={}):
        """
        Add a concentration mechanism to the tree

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism
        """
        for node in self: node.addConcMech(ion, params=params)

    def _permuteToTreeInds(self):
        return np.array([node.loc_ind for node in self])

    def _permuteToTree(self, mat):
        index_arr = self._permuteToTreeInds()
        if mat.ndim == 1:
            return mat[index_arr]
        elif mat.ndim == 2:
            return mat[index_arr,:][:,index_arr]
        elif mat.ndim == 3:
            return mat[:,index_arr,:][:,:,index_arr]

    def _permuteToLocsInds(self):
        """
        give index list that can be used to permutate the axes of the impedance
        and system matrix to correspond to the associated set of locations
        """
        loc_inds = np.array([node.loc_ind for node in self])
        return np.argsort(loc_inds)

    def _permuteToLocs(self, mat):
        index_arr = self._permuteToLocsInds()
        if mat.ndim == 1:
            return mat[index_arr]
        elif mat.ndim == 2:
            return mat[index_arr,:][:,index_arr]
        elif mat.ndim == 3:
            return mat[:,index_arr,:][:,:,index_arr]

    def getEquivalentLocs(self):
        """
        Get list of fake locations in the same order as original list of locations
        to which the compartment tree was fitted.

        Returns
        -------
        list of tuple
            Tuple has the form `(node.index, .5)`
        """
        loc_inds = [node.loc_ind for node in self]
        index_arr = np.argsort(loc_inds)
        locs_unordered = [(node.index, .5) for node in self]
        return [locs_unordered[ind] for ind in index_arr]

    def calcImpedanceMatrix(self, freqs=0., channel_names=None, indexing='locs',
                                use_conc=False):
        """
        Constructs the impedance matrix of the model for each frequency provided
        in `freqs`. This matrix is evaluated at the equilibrium potentials
        stored in each node

        Parameters
        ----------
            freqs: np.array (dtype = complex) or float
                Frequencies at which the matrix is evaluated [Hz]
            channel_names: ``None`` (default) or `list` of `str`
                The channels to be included in the matrix. If ``None``, all
                channels present on the tree are included in the calculation
            use_conc: bool
                wheter or not to use the concentration dynamics
            indexing: 'tree' or 'locs'
                Whether the indexing order of the matrix corresponds to the tree
                nodes (order in which they occur in the iteration) or to the
                locations on which the reduced model is based

        Returns
        -------
            `np.ndarray` (ndim = 3, dtype = complex)
                The first dimension corresponds to the
                frequency, the second and third dimension contain the impedance
                matrix for that frequency
        """
        return np.linalg.inv(self.calcSystemMatrix(freqs=freqs,
                             channel_names=channel_names, indexing=indexing,
                             use_conc=use_conc))

    def calcConductanceMatrix(self, indexing='locs'):
        """
        Constructs the conductance matrix of the model

        Returns
        -------
        `np.ndarray` (``dtype = float``, ``ndim = 2``)
            the conductance matrix
        """
        g_mat = np.zeros((len(self), len(self)))
        for node in self:
            ii = node.index
            g_mat[ii, ii] += node.getGTot(self.channel_storage) + node.g_c
            if node.parent_node is not None:
                jj = node.parent_node.index
                g_mat[jj,jj] += node.g_c
                g_mat[ii,jj] -= node.g_c
                g_mat[jj,ii] -= node.g_c
        if indexing == 'locs':
            return self._permuteToLocs(g_mat)
        elif indexing == 'tree':
            return g_mat
        else:
            raise ValueError('invalid argument for `indexing`, ' + \
                             'has to be \'tree\' or \'locs\'')

    def calcSystemMatrix(self, freqs=0., channel_names=None,
                               with_ca=True, use_conc=False,
                               indexing='locs'):
        """
        Constructs the matrix of conductance and capacitance terms of the model
        for each frequency provided in ``freqs``. this matrix is evaluated at
        the equilibrium potentials stored in each node

        Parameters
        ----------
            freqs: np.array (dtype = complex) or float (default ``0.``)
                Frequencies at which the matrix is evaluated [Hz]
            channel_names: `None` (default) or `list` of `str`
                The channels to be included in the matrix. If `None`, all
                channels present on the tree are included in the calculation
            with_ca: `bool`
                Whether or not to include the capacitive currents
            use_conc: `bool`
                wheter or not to use the concentration dynamics
            indexing: 'tree' or 'locs'
                Whether the indexing order of the matrix corresponds to the tree
                nodes (order in which they occur in the iteration) or to the
                locations on which the reduced model is based

        Returns
        -------
            `np.ndarray` (``ndim = 3, dtype = complex``)
                The first dimension corresponds to the
                frequency, the second and third dimension contain the impedance
                matrix for that frequency
        """
        # process inputs
        no_freq_dim = False
        if isinstance(freqs, float) or isinstance(freqs, complex):
            freqs = np.array([freqs])
            no_freq_dim = True
        if channel_names is None:
            channel_names = ['L'] + list(self.channel_storage.keys())

        s_mat = np.zeros((len(freqs), len(self), len(self)), dtype=freqs.dtype)
        for node in self:
            ii = node.index
            # set the capacitance contribution
            if with_ca: s_mat[:,ii,ii] += freqs * node.ca
            # set the coupling conductances
            s_mat[:,ii,ii] += node.g_c
            if node.parent_node is not None:
                jj = node.parent_node.index
                s_mat[:,jj,jj] += node.g_c
                s_mat[:,ii,jj] -= node.g_c
                s_mat[:,jj,ii] -= node.g_c
            # set the ion channel contributions
            g_terms = node.calcMembraneConductanceTerms(self.channel_storage,
                            freqs=freqs, channel_names=channel_names)
            s_mat[:,ii,ii] += sum([node.currents[c_name][0] * g_term \
                                   for c_name, g_term in g_terms.items()])
            if use_conc:
                for ion, concmech in node.concmechs.items():
                    c_term = node.calcMembraneConcentrationTerms(
                                        ion, self.channel_storage,
                                        freqs=freqs, channel_names=channel_names)
                    s_mat[:,ii,ii] += concmech.gamma * c_term

        if indexing == 'locs':
            s_mat = self._permuteToLocs(s_mat)
        elif not indexing == 'tree':
            raise ValueError('invalid argument for `indexing`, ' + \
                             'has to be \'tree\' or \'locs\'')

        return s_mat[0,:,:] if no_freq_dim else s_mat

    def calcEigenvalues(self, indexing='tree'):
        """
        Calculates the eigenvalues and eigenvectors of the passive system

        Returns
        -------
        np.ndarray (ndim = 1, dtype = complex)
            the eigenvalues
        np.ndarray (ndim = 2, dtype = complex)
            the right eigenvector matrix
        indexing: 'tree' or 'locs'
            Whether the indexing order of the matrix corresponds to the tree
            nodes (order in which they occur in the iteration) or to the
            locations on which the reduced model is based
        """
        # get the system matrix
        mat = self.calcSystemMatrix(freqs=0., channel_names=['L'],
                                    with_ca=False, indexing=indexing)
        ca_vec = np.array([node.ca for node in self])
        if indexing == 'locs':
            ca_vec = self._permuteToLocs(ca_vec)
        mat /= ca_vec[:,None]
        # compute the eigenvalues
        alphas, phimat = la.eig(mat)
        if max(np.max(np.abs(alphas.imag)), np.max(np.abs(phimat.imag))) < 1e-5:
            alphas = alphas.real
            phimat = phimat.real
        phimat_inv = la.inv(phimat)

        alphas /= -1e3
        phimat_inv /= ca_vec[None,:] * 1e3
        return alphas, phimat, phimat_inv

    def _calcConvolution(self, dt, inputs):
        """
        Compute the convolution of the `inputs` with the impedance matrix of the
        passive system

        Parameters
        ----------
        inputs: np.ndarray (ndim = 3)
            The inputs. First dimension is the input site (tree indices) and
            last dimension is time. Middle dimension can be arbitrary. Convolution
            is computed for all elements on the first 2 axes

        Return
        ------
        np.ndarray (ndim = 4)
            The convolutions
        """
        # compute the system eigenvalues for convolution
        alphas, phimat, phimat_inv = self.calcEigenvalues()
        # propagator s to compute convolution
        p0 = np.exp(alphas*dt)
        p1 = - 1. / alphas + (p0 - 1.) / (alphas**2 * dt)
        p2 =   p0 / alphas - (p0 - 1.) / (alphas**2 * dt)
        p_ = - 1. / alphas
        # multiply input matrix with phimat (original indices kct)
        inputs = np.einsum('nk,kct->nkct', phimat_inv, inputs)
        inputs = np.einsum('ln,nkct->lnkct', phimat, inputs)
        inputs = np.moveaxis(inputs, -1, 0) #tlnkc
        # do the convolution
        convres = np.zeros_like(inputs)
        convvar = np.einsum('n,lnkc->lnkc', p_, inputs[0])
        convres[0] = convvar
        for kk, inp in enumerate(inputs[1:]):
            inp_prev = inputs[kk]
            convvar = np.einsum('n,lnkc->lnkc', p0, convvar) + \
                      np.einsum('n,lnkc->lnkc', p1, inp) + \
                      np.einsum('n,lnkc->lnkc', p2, inp_prev)
            convres[kk+1] = convvar
        # recast result
        convres = np.einsum('tlnkc->tlkc', convres)
        convres = np.moveaxis(convres, 0, -1) # lkct

        return convres.real

    def _preprocessZMatArg(self, z_mat_arg):
        if isinstance(z_mat_arg, np.ndarray):
            return [self._permuteToTree(z_mat_arg)]
        elif isinstance(z_mat_arg, list):
            return [self._permuteToTree(z_mat) for z_mat in z_mat_arg]
        else:
            raise ValueError('`z_mat_arg` has to be ``np.ndarray`` or list of ' + \
                             '`np.ndarray`')

    def _preprocessEEqs(self, e_eqs, w_e_eqs=None):
        # preprocess e_eqs argument
        if e_eqs is None:
            e_eqs = np.array([self.getEEq(indexing='tree')])
        if isinstance(e_eqs, float):
            e_eqs = np.array([e_eqs])
        elif isinstance(e_eqs, list) or isinstance(e_eqs, tuple):
            e_eqs = np.array(e_eqs)
        elif isinstance(e_eqs, np.ndarray):
            pass
        else:
            raise TypeError('`e_eqs` has to be ``float`` or list or ' + \
                             '``np.ndarray`` of ``floats`` or ``np.ndarray``')
        # preprocess the w_e_eqs argument
        if w_e_eqs is None:
            w_e_eqs = np.ones_like(e_eqs)
        elif isinstance(w_e_eqs, float):
            w_e_eqs = np.array([e_eqs])
        elif isinstance(w_e_eqs, list) or isinstance(w_e_eqs, tuple):
            w_e_eqs = np.array(w_e_eqs)
        # check if arrays have the same shape
        assert w_e_eqs.shape[0] == e_eqs.shape[0]

        return e_eqs, w_e_eqs

    def _preprocessFreqs(self, freqs, w_freqs=None, z_mat_arg=None):
        if isinstance(freqs, float) or isinstance(freqs, complex):
            freqs = np.array([freqs])
        if w_freqs is None:
            w_freqs = np.ones_like(freqs)
        else:
            assert w_freqs.shape[0] == freqs.shape[0]
        # convert to 3d matrices if they are two dimensional
        z_mat_arg_ = []
        for z_mat in z_mat_arg:
            if z_mat.ndim == 2:
                z_mat_arg_.append(z_mat[np.newaxis,:,:])
            else:
                z_mat_arg_.append(z_mat)
            assert z_mat_arg_[-1].shape[0] == freqs.shape[0]
        z_mat_arg = z_mat_arg_
        return freqs, w_freqs, z_mat_arg

    def _toStructureTensorGMC(self, channel_names):
        g_vec = self._toVecGMC(channel_names)
        g_struct = np.zeros((len(self), len(self), len(g_vec)))
        kk = 0 # counter
        for node in self:
            ii = node.index
            g_terms = node.calcMembraneConductanceTerms(self.channel_storage,
                                freqs=0., channel_names=['L']+channel_names)
            if node.parent_node == None:
                # membrance conductance elements
                for channel_name in channel_names:
                    g_struct[0, 0, kk] += g_terms[channel_name]
                    kk += 1
            else:
                jj = node.parent_node.index
                # coupling conductance element
                g_struct[ii, jj, kk] -= 1.
                g_struct[jj, ii, kk] -= 1.
                g_struct[jj, jj, kk] += 1.
                g_struct[ii, ii, kk] += 1.
                kk += 1
                # membrance conductance elements
                for channel_name in channel_names:
                    g_struct[ii, ii, kk] += g_terms[channel_name]
                    kk += 1
        return g_struct

    def _toVecGMC(self, channel_names):
        """
        Place all conductances to be fitted in a single vector
        """
        g_list = []
        for node in self:
            if node.parent_node is None:
                g_list.extend([node.currents[c_name][0] for c_name in channel_names])
            else:
                g_list.extend([node.g_c] + \
                              [node.currents[c_name][0] for c_name in channel_names])
        return np.array(g_list)

    def _toTreeGMC(self, g_vec, channel_names):
        kk = 0 # counter
        for ii, node in enumerate(self):
            if node.parent_node is None:
                for channel_name in channel_names:
                    node.currents[channel_name][0] = g_vec[kk]
                    kk += 1
            else:
                node.g_c = g_vec[kk]
                kk += 1
                for channel_name in channel_names:
                    node.currents[channel_name][0] = g_vec[kk]
                    kk += 1

    def _toStructureTensorGM(self, freqs, channel_names, all_channel_names=None):
        # to construct appropriate channel vector
        if all_channel_names is None:
            all_channel_names = channel_names
        else:
            assert set(channel_names).issubset(all_channel_names)
        g_vec = self._toVecGM(all_channel_names)
        g_struct = np.zeros((len(freqs), len(self), len(self), len(g_vec)), dtype=freqs.dtype)
        # fill the fit structure
        kk = 0 # counter
        for node in self:
            ii = node.index
            g_terms = node.calcMembraneConductanceTerms(self.channel_storage,
                                    freqs=freqs, channel_names=channel_names)
            # membrance conductance elements
            for channel_name in all_channel_names:
                if channel_name in channel_names:
                    g_struct[:,ii,ii,kk] += g_terms[channel_name]
                kk += 1
        return g_struct

    def _toVecGM(self, channel_names):
        """
        Place all conductances to be fitted in a single vector
        """
        g_list = []
        for node in self:
            g_list.extend([node.currents[c_name][0] for c_name in channel_names])
        return np.array(g_list)

    def _toTreeGM(self, g_vec, channel_names):
        kk = 0 # counter
        for ii, node in enumerate(self):
            for channel_name in channel_names:
                node.currents[channel_name][0] = g_vec[kk]
                kk += 1

    def _toStructureTensorConc(self, ion, freqs, channel_names):
        # to construct appropriate channel vector
        c_struct = np.zeros((len(freqs), len(self), len(self), len(self)), dtype=freqs.dtype)
        # fill the fit structure
        for node in self:
            ii = node.index
            c_term = node.calcMembraneConcentrationTerms(ion, self.channel_storage,
                                    freqs=freqs, channel_names=channel_names)
            c_struct[:,ii,ii,ii] += c_term
        return c_struct

    def _toVecConc(self, ion):
        """
        Place concentration mechanisms to be fitted in a single vector
        """
        return np.array([node.concmechs[ion].gamma for node in self])

    def _toTreeConc(self, c_vec, ion):
        for ii, node in enumerate(self):
            node.concmechs[ion].gamma = c_vec[ii]

    def _toStructureTensorC(self, freqs):
        c_vec = self._toVecC()
        c_struct = np.zeros((len(freqs), len(self), len(self), len(c_vec)), dtype=complex)
        for node in self:
            ii = node.index
            # capacitance elements
            c_struct[:, ii, ii, ii] += freqs
        return c_struct

    def _toVecC(self):
        return np.array([node.ca for node in self])

    def _toTreeC(self, c_vec):
        for ii, node in enumerate(self):
            node.ca = c_vec[ii]

    def computeGMC(self, z_mat_arg, e_eqs=None, channel_names=['L']):
        """
        Fit the models' membrane and coupling conductances to a given steady
        state impedance matrix.

        Parameters
        ----------
        z_mat_arg: np.ndarray (ndim = 2, dtype = float or complex) or
                   list of np.ndarray (ndim = 2, dtype = float or complex)
            If a single array, represents the steady state impedance matrix,
            If a list of arrays, represents the steady state impedance
            matrices for each equilibrium potential in ``e_eqs``
        e_eqs: np.ndarray (ndim = 1, dtype = float) or float
            The equilibirum potentials in each compartment for each
            evaluation of ``z_mat``
        channel_names: list of string (defaults to ['L'])
            Names of the ion channels that have been included in the impedance
            matrix calculation and for whom the conductances are fit. Default is
            only leak conductance
        """
        z_mat_arg = self._preprocessZMatArg(z_mat_arg)
        e_eqs, _ = self._preprocessEEqs(e_eqs)
        assert len(z_mat_arg) == len(e_eqs)
        # do the fit
        mats_feature = []
        vecs_target = []
        for z_mat, e_eq in zip(z_mat_arg, e_eqs):
            # set equilibrium conductances
            self.setEEq(e_eq)
            # create the matrices for linear fit
            g_struct = self._toStructureTensorGMC(channel_names)
            tensor_feature = np.einsum('ij,jkl->ikl', z_mat, g_struct)
            tshape = tensor_feature.shape
            mat_feature_aux = np.reshape(tensor_feature,
                                         (tshape[0]*tshape[1], tshape[2]))
            vec_target_aux = np.reshape(np.eye(len(self)), (len(self)*len(self),))
            mats_feature.append(mat_feature_aux)
            vecs_target.append(vec_target_aux)
        mat_feature = np.concatenate(mats_feature, 0)
        vec_target = np.concatenate(vecs_target)
        # linear regression fit
        # res = la.lstsq(mat_feature, vec_target)
        res = so.nnls(mat_feature, vec_target)
        g_vec = res[0].real
        # set the conductances
        self._toTreeGMC(g_vec, channel_names)

    def computeGChanFromImpedance(self, channel_names, z_mat, e_eq, freqs,
                                sv=None, weight=1.,
                                all_channel_names=None, other_channel_names=None,
                                action='store'):
        """
        Fit the conductances of multiple channels from the given impedance
        matrices, or store the feature matrix and target vector for later use
        (see `action`).

        Parameters
        ----------
        channel_names: list of str
            The names of the ion channels whose conductances are to be fitted
        z_mat: np.ndarray (ndim=3)
            The impedance matrix to which the ion channel is fitted. Shape is
            ``(F, N, N)`` with ``N`` the number of compartments and ``F`` the
            number of frequencies at which the matrix is evaluated
        e_eq: float
            The equilibirum potential at which the impedance matrix was computed
        freqs: np.array
            The frequencies at which `z_mat` is computed (shape is ``(F,)``)
        sv: dict {channel_name: np.ndarray} (optional)
            The state variable expansion point. If ``np.ndarray``, assumes it is
            the expansion point of the channel that is fitted. If dict, the
            expansion points of multiple channels can be specified. An empty dict
            implies the asymptotic points derived from the equilibrium potential
        weight: float
            The relative weight of the feature matrices in this part of the fit
        all_channel_names: list of str or ``None``
            The names of all channels whose conductances will be fitted in a
            single linear least squares fit
        other_channel_names: list of str or ``None`` (default)
            List of channels present in `z_mat`, but whose conductances are
            already fitted. If ``None`` and 'L' is not in `all_channel_names`,
            sets `other_channel_names` to 'L'
        action: 'fit', 'store' or 'return'
            If 'fit', fits the conductances for this feature matrix and target
            vector for directly; only based on `z_mat`; nothing is stored.
            If 'store', stores the feature matrix and target vector to fit later
            on. Relative weight in fit will be determined by `weight`.
            If 'return', returns the feature matrix and target vector. Nothing
            is stored
        """
        # to construct appropriate channel vector
        if all_channel_names is None:
            all_channel_names = channel_names
        else:
            assert set(channel_names).issubset(all_channel_names)
        if other_channel_names is None and 'L' not in all_channel_names:
            other_channel_names = ['L']
        if sv is None:
            sv = {}

        z_mat = self._permuteToTree(z_mat)
        if isinstance(freqs, float):
            freqs = np.array([freqs])
        # set equilibrium conductances
        self.setEEq(e_eq)
        # set channel expansion point
        self.setExpansionPoints(sv)
        # feature matrix
        g_struct = self._toStructureTensorGM(freqs=freqs, channel_names=channel_names,
                                             all_channel_names=all_channel_names)
        tensor_feature = np.einsum('oij,ojkl->oikl', z_mat, g_struct)
        tshape = tensor_feature.shape
        mat_feature = np.reshape(tensor_feature,
                                     (tshape[0]*tshape[1]*tshape[2], tshape[3]))
        # target vector
        g_mat = self.calcSystemMatrix(freqs,
                            channel_names=other_channel_names, indexing='tree')
        zg_prod = np.einsum('oij,ojk->oik', z_mat, g_mat)
        mat_target = np.eye(len(self))[np.newaxis,:,:] - zg_prod
        vec_target = np.reshape(mat_target, (tshape[0]*tshape[1]*tshape[2],))

        return self._fitResAction(action, mat_feature, vec_target, weight,
                                  channel_names=all_channel_names)

    def computeGSingleChanFromImpedance(self, channel_name, z_mat, e_eq, freqs,
                                sv=None, weight=1.,
                                all_channel_names=None, other_channel_names=None,
                                action='store'):
        """
        Fit the conductances of a single channel from the given impedance
        matrices, or store the feature matrix and target vector for later use
        (see `action`).

        Parameters
        ----------
        channel_name: str
            The name of the ion channel whose conductances are to be fitted
        z_mat: np.ndarray (ndim=3)
            The impedance matrix to which the ion channel is fitted. Shape is
            ``(F, N, N)`` with ``N`` the number of compartments and ``F`` the
            number of frequencies at which the matrix is evaluated
        e_eq: float
            The equilibirum potential at which the impedance matrix was computed
        freqs: np.array
            The frequencies at which `z_mat` is computed (shape is ``(F,)``)
        sv: dict or nested dict of float or np.array, or None (default)
            The state variable expansion point. If simple dict, assumes it is
            the expansion point of the channel that is fitted. If nested dict, the
            expansion points of multiple channels can be specified. ``None``
            implies the asymptotic point derived from the equilibrium potential
        weight: float
            The relative weight of the feature matrices in this part of the fit
        all_channel_names: list of str or ``None``
            The names of all channels whose conductances will be fitted in a
            single linear least squares fit
        other_channel_names: list of str or ``None`` (default)
            List of channels present in `z_mat`, but whose conductances are
            already fitted. If ``None`` and 'L' is not in `all_channel_names`,
            sets `other_channel_names` to 'L'
        action: 'fit', 'store' or 'return'
            If 'fit', fits the conductances for this feature matrix and target
            vector for directly; only based on `z_mat`; nothing is stored.
            If 'store', stores the feature matrix and target vector to fit later
            on. Relative weight in fit will be determined by `weight`.
            If 'return', returns the feature matrix and target vector. Nothing
            is stored
        """
        # to construct appropriate channel vector
        if all_channel_names is None:
            all_channel_names = [channel_name]
        else:
            assert channel_name in all_channel_names
        if other_channel_names is None and 'L' not in all_channel_names:
            other_channel_names = ['L']

        z_mat = self._permuteToTree(z_mat)
        if isinstance(freqs, float):
            freqs = np.array([freqs])
        if sv is None or not isinstance(list(sv.items())[0], dict):
            # if it is not a nested dict, make nested dict
            sv = {channel_name: sv}
        # set equilibrium conductances
        self.setEEq(e_eq)
        # set channel expansion point
        self.setExpansionPoints(sv)
        # feature matrix
        g_struct = self._toStructureTensorGM(freqs=freqs, channel_names=[channel_name],
                                             all_channel_names=all_channel_names)
        tensor_feature = np.einsum('oij,ojkl->oikl', z_mat, g_struct)
        tshape = tensor_feature.shape
        mat_feature = np.reshape(tensor_feature,
                                     (tshape[0]*tshape[1]*tshape[2], tshape[3]))
        # target vector
        g_mat = self.calcSystemMatrix(freqs,
                            channel_names=other_channel_names, indexing='tree')
        zg_prod = np.einsum('oij,ojk->oik', z_mat, g_mat)
        mat_target = np.eye(len(self))[np.newaxis,:,:] - zg_prod
        vec_target = np.reshape(mat_target, (tshape[0]*tshape[1]*tshape[2],))

        self.removeExpansionPoints()

        return self._fitResAction(action, mat_feature, vec_target, weight,
                                  channel_names=all_channel_names)

    def computeConcMech(self, z_mat, e_eq, freqs, ion, sv_s=None,
                        weight=1., channel_names=None, action='store'):
        """
        Experimental function to fit the parameters of concentration mechanisms
        """
        np.set_printoptions(precision=5, linewidth=200)
        if sv_s is None:
            sv_s = [None for _ in channel_names]
        exp_points = {c_name: sv for c_name, sv in zip(channel_names, sv_s)}
        self.setExpansionPoints(exp_points)

        z_mat = self._permuteToTree(z_mat)
        if isinstance(freqs, float):
            freqs = np.array([freqs])
        # set equilibrium conductances
        self.setEEq(e_eq)
        # feature matrix
        g_struct = self._toStructureTensorConc(ion, freqs, channel_names)

        tensor_feature = np.einsum('oij,ojkl->oikl', z_mat, g_struct)
        tshape = tensor_feature.shape
        mat_feature = np.reshape(tensor_feature,
                                     (tshape[0]*tshape[1]*tshape[2], tshape[3]))
        # target vector
        g_mat = self.calcSystemMatrix(freqs, channel_names=channel_names+['L'],
                                             indexing='tree')

        zg_prod = np.einsum('oij,ojk->oik', z_mat, g_mat)
        mat_target = np.eye(len(self))[np.newaxis,:,:] - zg_prod
        vec_target = np.reshape(mat_target, (tshape[0]*tshape[1]*tshape[2],))

        # print(np.set_printoptions(precision=2))
        # print('z_mat fitted   =\n', self.calcImpedanceMatrix(use_conc=True, freqs=freqs, channel_names=channel_names)[0].real)
        # print('z_mat standard =\n', self.calcImpedanceMatrix(use_conc=True, freqs=freqs, channel_names=channel_names)[0].real)
        # print('z_mat no conc  =\n', self.calcImpedanceMatrix(use_conc=False, freqs=freqs, channel_names=channel_names)[0].real)

        self.removeExpansionPoints()

        return self._fitResAction(action, mat_feature, vec_target, weight, ion=ion)


    def computeC(self, alphas, phimat, weights=None, tau_eps=5.):
        """
        Fit the capacitances to the eigenmode expansion

        Parameters
        ----------
        alphas: np.ndarray (shape=(K,))
            The eigenmode inverse timescales (1/s)
        phimat: np.ndarray (shape=(K,C))
            The eigenmode vectors (C the number of compartments)
        weights: np.ndarray (shape=(K,)) or None
            The weights given to each eigenmode in the fit
        """
        # np.set_printoptions(precision=2)
        n_c, n_a = len(self), len(alphas)
        assert phimat.shape == (n_a, n_c)
        if weights is None: weights = np.ones_like(alphas)
        # construct the passive conductance matrix
        g_mat = - self.calcSystemMatrix(freqs=0., channel_names=['L'],
                                        with_ca=False, indexing='tree')

        # set lower limit for capacitance, fit not always well conditioned
        g_tot = np.array([node.getGTot(self.channel_storage, channel_names=['L']) \
                          for node in self])
        c_lim =  g_tot / (-alphas[0] * tau_eps)
        gamma_mat = alphas[:,None] * phimat * c_lim[None,:]

        # construct feature matrix and target vector
        mat_feature = np.zeros((n_a*n_c, n_c))
        vec_target = np.zeros(n_a*n_c)
        for ii, node in enumerate(self):
            mat_feature[ii*n_a:(ii+1)*n_a,ii] = alphas * phimat[:,ii] * weights
            vec_target[ii*n_a:(ii+1)*n_a] = np.reshape(np.dot(phimat, g_mat[ii:ii+1,:].T) - gamma_mat[:,ii:ii+1], n_a) * weights

        # least squares fit
        res = so.nnls(mat_feature, vec_target)[0]
        c_vec = res + c_lim
        self._toTreeC(c_vec)

    def computeCVF(self, freqs, zf_mat, eps=.05, max_iter=20):
        """
        Experimental function to compute capacitances from sparse Green's
        function matrices (Wybo et al., 2015)
        """
        assert freqs.shape[0] == zf_mat.shape[0]
        # compute inv
        zf_mat = self._permuteToTree(zf_mat)
        zf_inv = np.linalg.inv(zf_mat)
        hf_all = []
        for ii, node in enumerate(self):
            hf_ii = [1./zf_inv[:, node.index, node.index]]
            if node.parent_node is not None:
                hf_ii += [-zf_inv[:, node.index, node.parent_node.index] / zf_inv[:, node.index, node.index]]
            hf_ii += [-zf_inv[:, node.index, cn.index] / zf_inv[:, node.index, node.index] for cn in node.child_nodes]
            hf_all.append(np.array(hf_ii))
        # compute row kernels
        fef = ke.fExpFitter()
        # perform vector fits
        ca_vec = []
        for ii, (node ,hf_vec) in enumerate(zip(self,hf_all)):
            # run vector fit
            alphas_, gammas_, pairs, rms = fef.fitFExp_vector(freqs, hf_vec, deg=1)
            alpha = alphas_[0].real
            gammas = gammas_[:,0].real
            # coupling conductance vector
            g_cs = [node.g_c] if node.parent_node is not None else []
            g_cs += [cn.g_c for cn in node.child_nodes]
            g_cs = np.array(g_cs)

            # different c values
            ca_vec.append((node.currents['L'][0]+np.sum(g_cs)) / alpha)

        self._toTreeC(ca_vec)

    def computeGChanFromTrace(self, dv_mat, v_mat, i_mat,
                         p_open_channels=None, p_open_other_channels={}, test={},
                         weight=1.,
                         channel_names=None, all_channel_names=None, other_channel_names=None,
                         action='store'):
        """
        Experimental fit from trace, untested. Assumes leak conductance, coupling
        conductance and capacitance have already been fitted

        Parameters
        ----------
        dv_mat: np.ndarray (n,k)
        v_mat: np.ndarray (n,k)
        i_mat: np.ndarray (n,k)
            n = nr. of locations, k = nr. of fit points
        """
        # print '\nxxxxxxxx'
        # check size
        assert v_mat.shape == i_mat.shape
        assert dv_mat.shape == i_mat.shape
        assert dv_mat.shape == i_mat.shape
        for channel_name, p_open in p_open_channels.items():
            assert p_open.shape == i_mat.shape
        for channel_name, p_open in p_open_other_channels.items():
            assert p_open.shape == i_mat.shape

        # define channel name lists
        if channel_names is None:
            channel_names = list(p_open_channels.keys())
        else:
            assert set(channel_names) == set(p_open_channels.keys())
        if other_channel_names is None:
            other_channel_names = list(p_open_other_channels.keys())
        else:
            assert set(other_channel_names) == set(p_open_other_channels.keys())
        if all_channel_names == None:
            all_channel_names = channel_names
        else:
            assert set(channel_names).issubset(all_channel_names)

        # numbers for fit
        n_loc, n_fp, n_chan = len(self), i_mat.shape[1], len(all_channel_names)

        mat_feature = np.zeros((n_fp * n_loc, n_loc * n_chan))
        vec_target = np.zeros(n_fp * n_loc)
        for ii, node in enumerate(self):
            # define fit vectors
            i_vec = np.zeros(n_fp)
            d_vec = np.zeros((n_fp, n_chan))
            # add input current
            i_vec += i_mat[node.loc_ind]
            # add capacitive current
            i_vec -= node.ca * dv_mat[node.loc_ind] * 1e3 # convert to nA
            # add the coupling terms
            pnode = node.parent_node
            if pnode is not None:
                i_vec += node.g_c * (v_mat[pnode.loc_ind] - v_mat[node.loc_ind])
            for cnode in node.child_nodes:
                i_vec += cnode.g_c * (v_mat[cnode.loc_ind] - v_mat[node.loc_ind])
            # add the leak terms
            g_l, e_l = node.currents['L']
            i_vec += g_l * (e_l - v_mat[node.loc_ind])
            # add the ion channel current
            for channel_name, p_open in p_open_other_channels.items():
                i_vec -= node.getDynamicI(channel_name,
                                        p_open[node.loc_ind], v_mat[node.loc_ind])
            # drive terms
            for kk, channel_name in enumerate(all_channel_names):
                if channel_name in channel_names:
                    p_open = p_open_channels[channel_name]
                    d_vec[:, kk] = node.getDynamicDrive(channel_name,
                                            p_open[node.loc_ind], v_mat[node.loc_ind])

        return self._fitResAction(action, mat_feature, vec_target, weight,
                                  channel_names=all_channel_names)

    def computeGChanFromTraceConv(self, dt, v_mat, i_mat,
                         p_open_channels=None, p_open_other_channels={}, test={},
                         weight=1.,
                         channel_names=None, all_channel_names=None, other_channel_names=None,
                         v_pas=None,
                         action='store'):
        """
        Experimental fit from trace, untested. Assumes leak conductance, coupling
        conductance and capacitance have already been fitted

        Parameters
        ----------
        dv_mat: np.ndarray (n,k)
        v_mat: np.ndarray (n,k)
        i_mat: np.ndarray (n,k)
            n = nr. of locations, k = nr. of fit points
        """


        import matplotlib.pyplot as pl
        # check size
        assert v_mat.shape == i_mat.shape
        for channel_name, p_open in p_open_channels.items():
            assert p_open.shape == i_mat.shape
        for channel_name, p_open in p_open_other_channels.items():
            assert p_open.shape == i_mat.shape

        # define channel name lists
        if channel_names is None:
            channel_names = list(p_open_channels.keys())
        else:
            assert set(channel_names) == set(p_open_channels.keys())
        if other_channel_names is None:
            other_channel_names = list(p_open_other_channels.keys())
        else:
            assert set(other_channel_names) == set(p_open_other_channels.keys())
        if all_channel_names == None:
            all_channel_names = channel_names
        else:
            assert set(channel_names).issubset(all_channel_names)

        es_eq = np.array([node.e_eq for node in self])

        # permute inputs to tree
        perm_inds = self._permuteToTreeInds()
        v_mat = v_mat[perm_inds,:]
        i_mat = i_mat[perm_inds,:]
        for p_o in p_open_channels.values():
            p_o = p_o[perm_inds,:]
        for p_o in p_open_other_channels.values():
            p_o = p_o[perm_inds,:]

        # numbers for fit
        n_loc, n_fp, n_chan = len(self), i_mat.shape[1], len(all_channel_names)

        if v_pas is None:
            # compute convolution input current for fit
            for channel_name, p_open in p_open_other_channels.items():
                for ii, node in enumerate(self):
                    i_mat[ii] -= node.getDynamicI(channel_name, p_open[ii], v_mat[ii])
            v_i_in = self._calcConvolution(dt, i_mat[:,np.newaxis,:])
            v_i_in = np.sum(v_i_in[:,:,0,:], axis=1)
            v_fit = v_mat - es_eq[:,None] - v_i_in
        else:
            v_fit = v_mat - es_eq[:,None] - v_pas
        v_fit_aux = v_fit
        v_fit = np.reshape(v_fit, n_loc*n_fp)

        # compute channel drive convolutions
        d_chan = np.zeros((n_loc, n_chan, n_fp))
        for kk, channel_name in enumerate(all_channel_names):
            if channel_name in channel_names:
                p_open = p_open_channels[channel_name]
                for ii, node in enumerate(self):
                    # d_chan[ii,kk,:] -= node.getDynamicDrive(channel_name, p_open[ii], v_mat[ii])
                    d_chan[ii,kk,:] -= node.getDynamicDrive_(channel_name, v_mat[ii], dt, channel_storage=self.channel_storage)
        v_d = self._calcConvolution(dt, d_chan)
        v_d_aux = v_d

        v_d = np.reshape(v_d, (n_loc, n_loc*n_chan, n_fp))
        v_d = np.moveaxis(v_d, -1, 1)
        v_d = np.reshape(v_d, (n_loc * n_fp, n_loc*n_chan))
        # create the matrices for fit
        mat_feature = v_d
        vec_target = v_fit

        g_vec = so.nnls(mat_feature, vec_target)[0]
        print('g single fit =', g_vec)

        n_panel = len(self)+1
        t_arr = np.arange(n_fp) * dt

        pl.figure('fit', figsize=(n_panel*3, n_panel*3))
        gs = pl.GridSpec(n_panel,n_panel)
        gs.update(top=0.98, bottom=0.05, left=0.05, right=0.98, hspace=0.4, wspace=0.4)

        for ii, node in enumerate(self):
            # plot voltage
            ax_v = myAx(pl.subplot(gs[ii+1,0]))
            ax_v.set_title('node %d'%node.index)
            ax_v.plot(t_arr, v_mat[ii] - es_eq[ii], c='r', label=r'$V_{rec}$')
            ax_v.plot(t_arr, v_i_in[ii], c='b', label=r'$V_{inp}$')
            ax_v.plot(t_arr, v_fit_aux[ii], c='y', label=r'$V_{tofit}$')
            # compute fitted voltage
            v_fitted = np.zeros_like(t_arr)
            for jj in range(n_loc):
                for kk in range(n_chan):
                    v_fitted += g_vec[jj*n_chan+kk] * v_d_aux[ii,jj,kk]
            ax_v.plot(t_arr, v_fitted, c='c', ls='--', lw=1.6, label=r'$V_{fitted}$')
            ax_v.set_xlabel(r'$t$ (ms)')
            ax_v.set_ylabel(r'$V$ (mV)')
            myLegend(ax_v, loc='upper left')

            # plot drive
            ax_d = myAx(pl.subplot(gs[0,ii+1]))
            ax_d.set_title('node %d'%node.index)
            for kk, cname in enumerate(channel_names):
                ax_d.plot(t_arr, d_chan[ii,kk], c=colours[kk%len(colours)], label=cname)
            ax_d.set_xlabel(r'$t$ (ms)')
            ax_d.set_ylabel(r'$D$ (mV)')
            myLegend(ax_d, loc='upper left')

            for jj, node in enumerate(self):
                # plot drive convolution
                ax_vd = myAx(pl.subplot(gs[ii+1,jj+1]))
                for kk, cname in enumerate(channel_names):
                    ax_vd.plot(t_arr, g_vec[jj*n_chan+kk] * v_d_aux[ii,jj,kk], c=colours[kk%len(colours)], label=cname)
                ax_vd.set_xlabel(r'$t$ (ms)')
                ax_vd.set_ylabel(r'$C$ (mV)')
                myLegend(ax_vd, loc='upper left')

        return self._fitResAction(action, mat_feature, vec_target, weight,
                                  channel_names=all_channel_names)

    def _fitResAction(self, action, mat_feature, vec_target, weight,
                            ca_lim=[], **kwargs):
        if action == 'fit':
            # linear regression fit
            res = so.nnls(mat_feature, vec_target)
            vec_res = res[0].real
            # set the conductances
            if 'channel_names' in kwargs:
                self._toTreeGM(vec_res, channel_names=kwargs['channel_names'])
            elif 'ion' in kwargs:
                self._toTreeConc(vec_res, kwargs['ion'])
            else:
                raise IOError('Provide \'channel_names\' or \'ion\' as keyword argument')
        elif action == 'return':
            return mat_feature, vec_target
        elif action == 'store':
            if 'channel_names' in kwargs:
                try:
                    assert self.fit_data['ion'] == ''
                except AssertionError:
                    raise IOError('Stored fit matrices are concentration mech fits, ' + \
                                  'do not try to store channel conductance fit matrices')
                if len(self.fit_data['channel_names']) == 0:
                    self.fit_data['channel_names'] = kwargs['channel_names']
                else:
                    try:
                        assert self.fit_data['channel_names'] == kwargs['channel_names']
                    except AssertionError:
                        raise IOError('`channel_names` does not agree with stored ' + \
                                      'channel names for other fits\n' + \
                                      '`channel_names`:      ' + str(kwargs['channel_names']) + \
                                      '\nstored channel names: ' + str(self.fit_data['channel_names']))
            elif 'ion' in kwargs:
                try:
                    assert len(self.fit_data['channel_names']) == 0
                except AssertionError:
                    raise IOError('Stored fit matrices are channel conductance fits, ' + \
                                  'do not try to store concentration fit matrices')
                if self.fit_data['ion'] == '':
                    self.fit_data['ion'] = kwargs['ion']
                else:
                    try:
                        assert self.fit_data['ion'] == kwargs['ion']
                    except AssertionError:
                        raise IOError('`ion` does not agree with stored ion for ' + \
                                      'other fits:\n' + \
                                      '`ion`: ' + kwargs[ion] + \
                                      '\nstored ion: ' + self.fit_data['ion'])

            self.fit_data['mats_feature'].append(mat_feature)
            self.fit_data['vecs_target'].append(vec_target)
            self.fit_data['weights_fit'].append(weight)
        else:
            raise IOError('Undefined action, choose \'fit\', \'return\' or \'store\'.')

    def resetFitData(self):
        """
        Delete all stored feature matrices and and target vectors.
        """
        self.fit_data = dict(mats_feature=[],
                             vecs_target=[],
                             weights_fit=[],
                             channel_names=[],
                             ion='')

    def runFit(self):
        """
        Run a linear least squares fit for the conductances concentration
        mechanisms. The obtained conductances are stored on each node. All
        stored feature matrices and and target vectors are deleted.
        """
        fit_data = self.fit_data
        if len(fit_data['mats_feature']) > 0:
            # apply the weights
            for (m_f, v_t, w_f) in zip(fit_data['mats_feature'], fit_data['vecs_target'], fit_data['weights_fit']):
                nn = len(v_t)
                m_f *= w_f / nn
                v_t *= w_f / nn
            # create the fit matrices
            mat_feature = np.concatenate(fit_data['mats_feature'])
            vec_target = np.concatenate(fit_data['vecs_target'])
            # do the fit
            if len(fit_data['channel_names']) > 0:
                self._fitResAction('fit', mat_feature, vec_target, 1.,
                                   channel_names=fit_data['channel_names'])
            elif fit_data['ion'] != '':
                self._fitResAction('fit', mat_feature, vec_target, 1.,
                                   ion=fit_data['ion'])
            # reset fit data
            self.resetFitData()
        else:
             warnings.warn('No fit matrices are stored, no fit has been performed', UserWarning)

    def computeFakeGeometry(self, fake_c_m=1., fake_r_a=100.*1e-6,
                                  factor_r_a=1e-6, delta=1e-14,
                                  method=2):
        """
        Computes a fake geometry so that the neuron model is a reduced
        compurtmental model

        Parameters
        ----------
        fake_c_m: float [uF / cm^2]
            fake membrane capacitance value used to compute the surfaces of
            the compartments
        fake_r_a: float [MOhm * cm]
            fake axial resistivity value, used to evaluate the lengths of each
            section to yield the correct coupling constants

        Returns
        -------
        radii, lengths: np.array of floats [cm]
            The radii, lengths, resp. surfaces for the section in NEURON. Array
            index corresponds to NEURON index

        Raises
        ------
        AssertionError
            If the node indices are not ordered consecutively when iterating
        """

        assert self.checkOrdered()
        factor_r = 1. / np.sqrt(factor_r_a)
        # compute necessary vectors for calculating
        surfaces = np.array([node.ca / fake_c_m for node in self])
        vec_coupling = np.array([1.] + [1./node.g_c for node in self if \
                                            node.parent_node is not None])
        if method == 1:
            # find the 3d points to construct the segments' geometry
            p0s = -surfaces
            p1s = np.zeros_like(p0s)
            p2s = np.pi * (factor_r**2 - 1.) * np.ones_like(p0s)
            p3s = 2. * np.pi**2 * vec_coupling / fake_r_a * (1. + factor_r)
            # find the polynomial roots
            points = []
            for ii, (p0, p1, p2, p3) in enumerate(zip(p0s, p1s, p2s, p3s)):
                res = np.roots([p3,p2,p1,p0])
                # compute radius and length of first half of section
                radius = res[np.where(res.real > 0.)[0][0]].real
                radius *= 1e4 # convert [cm] to [um]
                length = np.pi * radius**2 * vec_coupling[ii] / (fake_r_a * 1e4) # convert [MOhm*cm] to [MOhm*um]
                # compute the pt3d points
                point0 = [0., 0., 0., 2.*radius]
                point1 = [length, 0., 0., 2.*radius]
                point2 = [length*(1.+delta), 0., 0., 2.*radius*factor_r]
                point3 = [length*(2.+delta), 0., 0., 2.*radius*factor_r]
                points.append([point0, point1, point2, point3])

            return points, surfaces
        elif method == 2:
            radii = np.cbrt(fake_r_a * surfaces / (vec_coupling * (2.*np.pi)**2))
            lengths = surfaces / (2. * np.pi * radii)
            return lengths, radii
        else:
            raise ValueError('Invalid `method` argument, should be 1 or 2')


    def plotDendrogram(self, ax,
                        plotargs={}, labelargs={}, textargs={},
                        nodelabels={}, bbox=None,
                        y_max=None):
        """
        Generate a dendrogram of the NET

        Parameters
        ----------
            ax: `matplotlib.axes`
                the axes object in which the plot will be made
            plotargs : dict (string : value)
                keyword args for the matplotlib plot function, specifies the
                line properties of the dendrogram
            labelargs : dict (string : value)
                keyword args for the matplotlib plot function, specifies the
                marker properties for the node points. Or dict with keys node
                indices, and with values dicts with keyword args for the
                matplotlib function that specify the marker properties for
                specific node points. The entry under key -1 specifies the
                properties for all nodes not explicitly in the keys.
            textargs : dict (string : value)
                keyword args for matplotlib textproperties
            nodelabels: dict (int: string) or None
                labels of the nodes. If None, nodes are named by default
                according to their location indices. If empty dict, no labels
                are added.
            y_max: int, float or None
                specifies the y-scale. If None, the scale is computed from
                ``self``. By default, y=1 is added for each child of a node, so
                if y_max is smaller than the depth of the tree, part of it will
                not be plotted
        """
        # get the number of leafs to determine the dendrogram spacing
        rnode    = self.root
        n_branch  = self.degreeOfNode(rnode)
        l_spacing = np.linspace(0., 1., n_branch+1)
        if y_max is None:
            y_max = np.max([self.depthOfNode(n) for n in self.leafs]) + 1.5
        y_min = .5
        # plot the dendrogram
        self._expandDendrogram(rnode, 0.5, None, 0.,
                    l_spacing, y_max, ax,
                    plotargs=plotargs, labelargs=labelargs, textargs=textargs,
                    nodelabels=nodelabels, bbox=bbox)
        # limits
        ax.set_ylim((y_min, y_max))
        ax.set_xlim((0.,1.))

        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        # ax.axison = False

        return y_max

    def _expandDendrogram(self, node, x0, xprev, y0,
                                        l_spacing, y_max, ax,
                                        plotargs={}, labelargs={}, textargs={},
                                        nodelabels={}, bbox=None):
        # impedance of layer
        ynew = y0 + 1.
        # plot vertical connection line
        # ax.vlines(x0, y0, ynew, **plotargs)
        if xprev is not None:
            ax.plot([xprev, x0], [y0, ynew], **plotargs)
        # get the child nodes for recursion
        l0 = 0
        for i, cnode in enumerate(node.child_nodes):
            # attribute space on xaxis
            deg = self.degreeOfNode(cnode)
            l1 = l0 + deg
            # new quantities
            xnew = (l_spacing[l0] + l_spacing[l1]) / 2.
            # horizontal connection line limits
            if i == 0:
                xnew0 = xnew
            if i == len(node.child_nodes)-1:
                xnew1 = xnew
            # recursion
            self._expandDendrogram(cnode, xnew, x0, ynew,
                    l_spacing[l0:l1+1], y_max, ax,
                    plotargs=plotargs, labelargs=labelargs, textargs=textargs,
                    nodelabels=nodelabels, bbox=None)
            # next index
            l0 = l1
        # add label and maybe text annotation to node
        if node.index in labelargs:
            ax.plot([x0], [ynew], **labelargs[node.index])
        elif -1 in labelargs:
            ax.plot([x0], [ynew], **labelargs[-1])
        else:
            try:
                ax.plot([x0], [ynew], **labelargs)
            except TypeError as e:
                pass
        if textargs:
            if nodelabels != None:
                if node.index in nodelabels:
                    if labelargs == {}:
                        ax.plot([x0], [ynew], **nodelabels[node.index][1])
                        ax.annotate(nodelabels[node.index][0],
                                    xy=(x0, ynew), xytext=(x0+0.06, ynew),#+y_max*0.04),
                                    bbox=bbox,
                                    **textargs)
                    else:
                        ax.annotate(nodelabels[node.index],
                                    xy=(x0, ynew), xytext=(x0+0.06, ynew),#+y_max*0.04),
                                    bbox=bbox,
                                    **textargs)
            else:
                ax.annotate(r'$N='+''.join([str(ind) for ind in node.loc_inds])+'$',
                                 xy=(x0, ynew), xytext=(x0+0.06, ynew),#+y_max*0.04),
                                 bbox=bbox,
                                 **textargs)










