"""
File contains:

    - :class:`CompartmentNode`
    - :class:`CompartmentTree`

Author: W. Wybo
"""


import numpy as np
import scipy.linalg as la
import scipy.optimize as so

from stree import SNode, STree
from neat.channels import channelcollection

import copy


class CompartmentNode(SNode):
    '''
    Implements a node for :class:`CompartmentTree`

    Attributes
    ----------
        ca: float
            capacitance of the compartment (uF)
        g_l: float
            leak conductance at the compartment (uS)
        g_c: float
            Coupling conductance of compartment with parent compartment (uS).
            Ignore if node is the root
    '''
    def __init__(self, index, loc_ind=None, ca=1., g_c=0., g_l=1e-2, e_eq=-75.):
        super(CompartmentNode, self).__init__(index)
        # location index this node corresponds to
        self.loc_ind = loc_ind
        # compartment params
        self.ca = ca   # capacitance (uF)
        self.g_c = g_c # coupling conductance (uS)
        # self.g_l = g_l # leak conductance (uS)
        self.e_eq = e_eq # equilibrium potential (mV)
        self.currents = {'L': [g_l, e_eq]} # ion channel currents and reversals
        self.concmechs = {}
        self.expansion_points = {}

    def __str__(self, with_parent=False, with_children=False):
        node_string = super(CompartmentNode, self).__str__()
        if self.parent_node is not None:
            node_string += ', Parent: ' + super(CompartmentNode, self.parent_node).__str__()
        node_string += ' --- (g_c = %.12f uS, '%self.g_c + \
                       ', '.join(['g_' + cname + ' = %.12f uS'%cpar[0] \
                            for cname, cpar in self.currents.iteritems()]) + \
                       ', c = %.12f uF)'%self.ca
        return node_string

    def addCurrent(self, channel_name, e_rev=None, channel_storage=None):
        if channel_name is not 'L':
            if e_rev is None:
                e_rev = channelcollection.E_REV_DICT[channel_name]
            self.currents[channel_name] = [0., e_rev]
            if channel_storage is not None and channel_name not in channel_storage:
                channel_storage[channel_name] = \
                                eval('channelcollection.' + channel_name + '()')
            self.expansion_points[channel_name] = None

    def addConcMech(self, ion, params={}):
        '''
        Add a concentration mechanism at this node.

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism (only used for NEURON model)
        '''
        self.concmechs[ion] = params

    def getCurrent(self, channel_name, channel_storage=None):
        '''
        Returns an ``::class::neat.channels.ionchannels.IonChannel`` object. If
        `channel_storage` is given,

        Parameters
        ----------
        channel_name: string
            the name of the ion channel
        channel_storage: dict of ionchannels (optional)
            keys are the names of the ion channels, and values the channel
            instances
        '''
        try:
            return channel_storage[channel_name]
        except (KeyError, TypeError):
            return eval('channelcollection.' + channel_name + '()')

    def setExpansionPoint(self, channel_name, statevar='asymptotic', channel_storage=None):
        '''
        Set the choice for the state variables of the ion channel around which
        to linearize.

        Note that when adding an ion channel to the node, the
        default expansion point setting is to linearize around the asymptotic values
        for the state variables at the equilibrium potential store in `self.e_eq`.
        Hence, this function only needs to be called to change that setting.

        Parameters
        ----------
        channel_name: string
            the name of the ion channel
        statevar: `np.ndarray`, `'max'` or `'asymptotic'` (default)
            If `np.ndarray`, should be of the same shape as the ion channels'
            state variables array, if `'max'`, the point at which the
            linearized channel current is maximal for the given equilibirum potential
            `self.e_eq` is used. If `'asymptotic'`, linearized around the asymptotic values
            for the state variables at the equilibrium potential
        channel_storage: dict of ion channels (optional)
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized

        Raises
        ------
        KeyError: if `channel_name` is not in `self.currents`
        '''
        if isinstance(statevar, str):
            if statevar == 'asymptotic':
                statevar = None
            elif statevar == 'max':
                channel = self.getCurrent(channel_name, channel_storage=channel_storage)
                statevar = channel.findMaxCurrentVGiven(self.e_eq, self.freqs,
                                                        self.currents[channel_name][1])
        self.expansion_points[channel_name] = statevar

    def calcMembraneConductanceTerms(self, freqs=0.,
                                     channel_names=None, channel_storage=None):
        '''
        Compute the membrane impedance terms and return them as a `dict`

        Parameters
        ----------
        freqs: np.ndarray (ndim = 1, dtype = complex or float) or float or complex
            The frequencies at which the impedance terms are to be evaluated
        channel_storage: dict of ion channels (optional)
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized

        Returns
        -------
        dict of np.ndarray or float or complex
            Each entry in the dict is of the same type as ``freqs`` and is the
            conductance term of a channel
        '''
        if channel_names is None: channel_names = self.currents.keys()
        cond_terms = {'L': 1.} # leak conductance has 1 as prefactor
        for channel_name in set(channel_names) - set('L'):
            if channel_name not in self.currents:
                self.addCurrent(channel_name, channel_storage=channel_storage)
            e = self.currents[channel_name][1]
            # create the ionchannel object
            channel = self.getCurrent(channel_name, channel_storage=channel_storage)
            # check if needs to be computed around expansion point
            sv = self.expansion_points[channel_name]
            # add channel contribution to membrane impedance
            cond_terms[channel_name] = - channel.computeLinSum(self.e_eq, freqs, e,
                                                               statevars=sv)

        return cond_terms

    def getGTot(self, v=None, channel_names=None, channel_storage=None):
        if channel_names is None: channel_names = self.currents.keys()
        g_tot = self.currents['L'][0] if 'L' in channel_names else 0.
        v = self.e_eq if v is None else v
        for channel_name in channel_names:
            if channel_name != 'L':
                g, e = self.currents[channel_name]
                # create the ionchannel object
                channel = self.getCurrent(channel_name, channel_storage=channel_storage)
                # check if needs to be computed around expansion point
                sv = self.expansion_points[channel_name]
                g_tot += g * channel.computePOpen(v, statevars=sv)

        return g_tot

    def setGTot(self, illegal):
        raise AttributeError("`g_tot` is a read-only attribute, set the leak " + \
                             "conductance by calling ``func:addCurrent`` with " + \
                             " \'L\' as `current_type`")

    g_tot = property(getGTot, setGTot)

    def getITot(self, v=None, channel_names=None, channel_storage=None):
        if channel_names is None: channel_names = self.currents.keys()
        v = self.e_eq if v is None else v
        i_tot = self.currents['L'][0] * (v - self.currents['L'][1]) if 'L' in channel_names else 0.
        for channel_name in channel_names:
            if channel_name != 'L':
                g, e = self.currents[channel_name]
                # create the ionchannel object
                channel = self.getCurrent(channel_name, channel_storage=channel_storage)
                # check if needs to be computed around expansion point
                sv = self.expansion_points[channel_name]
                i_tot += g * channel.computePOpen(v, statevars=sv) * (v - e)

        return i_tot

    def getDrive(self, channel_name, v=None, channel_storage=None):
        v = self.e_eq if v is None else v
        _, e = self.currents[channel_name]
        # create the ionchannel object
        channel = self.getCurrent(channel_name, channel_storage=channel_storage)
        sv = self.expansion_points[channel_name]
        return channel.computePOpen(v, statevars=sv) * (v - e)

    # def getITot(self, v=None, channel_names=None, channel_storage=None):
    #     if channel_names is None: channel_names = self.currents.keys()
    #     v = self.e_eq if v is None else v
    #     i_tot = self.currents['L'][0] * (v - self.currents['L'][1]) \
    #             if 'L' in channel_names else 0.
    #     for channel_name in channel_names:
    #         if channel_name != 'L':
    #             g, e = self.currents[channel_name]
    #             # create the ionchannel object
    #             if channel_storage is not None:
    #                 channel = channel_storage[channel_name]
    #             else:
    #                 channel = eval('channelcollection.' + channel_name + '()')
    #             i_tot += g * channel.computePOpen(v) * (v - e)

    #     return i_tot

    def fitEL(self, channel_storage=None):
        i_eq = 0.
        for channel_name in set(self.currents.keys()) - set('L'):
            g, e = self.currents[channel_name]
            # create the ionchannel object
            if channel_storage is not None:
                channel = channel_storage[channel_name]
            else:
                channel = eval('channelcollection.' + channel_name + '()')
            # compute channel conductance and current
            p_open = channel.computePOpen(self.e_eq)
            i_chan = g * p_open * (e - self.e_eq)
            i_eq += i_chan
        e_l = self.e_eq - i_eq / self.currents['L'][0]
        self.currents['L'][1] = e_l


class CompartmentTree(STree):
    def __init__(self, root=None):
        super(CompartmentTree, self).__init__(root=root)
        self.channel_storage = {}

    def createCorrespondingNode(self, index, ca=1., g_c=0., g_l=1e-2):
        '''
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        '''
        return CompartmentNode(index, ca=ca, g_c=g_c, g_l=g_l)

    def setEEq(self, e_eq):
        if isinstance(e_eq, float) or isinstance(e_eq, int):
            e_eq = e_eq * np.ones(len(self), dtype=float)
        else:
            e_eq = self._permuteToTree(np.array(e_eq))
        for ii, node in enumerate(self): node.e_eq = e_eq[ii]

    def getEEq(self):
        return np.array([node.e_eq for node in self])

    def setExpansionPoints(self, expansion_points):
        to_tree_inds = self._permuteToTreeInds()
        for channel_name, expansion_point in expansion_points.iteritems():
            # if one set of state variables, set throughout neuron
            if isinstance(expansion_point, str) or \
               expansion_point is None:
                svs = np.array([expansion_point for _ in self])
            elif isinstance(expansion_point, np.ndarray):
                if expansion_point.ndim == 3:
                    svs = np.array(expansion_point)
                elif expansion_point.ndim == 2:
                    svs = np.array([expansion_point for _ in self])
            for node, sv in zip(self, svs[to_tree_inds]):
                node.setExpansionPoint(channel_name, statevar=sv,
                                       channel_storage=self.channel_storage)

    def removeExpansionPoints(self):
        for node in self:
            for channel_name in node.currents:
                node.setExpansionPoint(channel_name, statevar='asymptotic')

    def fitEL(self):
        '''
        Set the leak reversal potential to obtain the desired equilibrium
        potentials
        '''
        e_l_0 = self.getEEq()
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
            fun_vals[ii] += node.getITot()
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

    def addCurrent(self, channel_name, e_rev=None):
        '''
        Add an ion channel current to the tree

        Parameters
        ----------
        channel_name: string
            The name of the channel type
        '''
        for ii, node in enumerate(self):
            node.addCurrent(channel_name, e_rev=e_rev,
                            channel_storage=self.channel_storage)

    def addConcMech(self, ion, params={}):
        '''
        Add a concentration mechanism to the tree

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism (only used for NEURON model)
        '''
        for node in self: node.addConcMech(ion, params=params)

    def _permuteToTreeInds(self):
        return np.array([node.loc_ind for node in self])

    def _permuteToTree(self, mat):
        '''
        give index list that can be used to permutate the axes of the impedance
        and system matrix to correspond to the associated set of locations
        '''
        index_arr = self._permuteToTreeInds()
        if mat.ndim == 1:
            return mat[index_arr]
        elif mat.ndim == 2:
            return mat[index_arr,:][:,index_arr]
        elif mat.ndim == 3:
            return mat[:,index_arr,:][:,:,index_arr]

    def _permuteToLocsInds(self):
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
        loc_inds = [node.loc_ind for node in self]
        index_arr = np.argsort(loc_inds)
        locs_unordered = [(node.index, .5) for node in self]
        return [locs_unordered[ind] for ind in index_arr]


    def calcImpedanceMatrix(self, freqs=0., channel_names=None, indexing='locs'):
        return np.linalg.inv(self.calcSystemMatrix(freqs=freqs,
                             channel_names=channel_names, indexing=indexing))

    def calcConductanceMatrix(self, indexing='locs'):
        '''
        Constructs the conductance matrix of the model

        Returns
        -------
            np.ndarray (dtype = float, ndim = 2)
                the conductance matrix
        '''
        g_mat = np.zeros((len(self), len(self)))
        for node in self:
            ii = node.index
            g_mat[ii, ii] += node.g_tot + node.g_c
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

    def calcSystemMatrix(self, freqs=0., channel_names=None, with_ca=True,
                               indexing='locs'):
        '''
        Constructs the matrix of conductance and capacitance terms of the model
        for each frequency provided in ``freqs``. this matrix is evaluated at
        the equilibrium potentials stored in each node

        Parameters
        ----------
            freqs: np.array (dtype = complex)
                Frequencies at which the matrix is evaluated [Hz]
            channel_names: `None` or `list` of `str`
                The channels to be included in the matrix
            with_ca: `bool`
                Whether or not to include the capacitive currents
            indexing: 'tree' or 'locs'
                Whether the indexing order of the matrix corresponds to the tree
                nodes (order in which they occur in the iteration) or to the
                locations on which the reduced model is based

        Returns
        -------
            np.ndarray (ndim = 3, dtype = complex)
                The first dimension corresponds to the
                frequency, the second and third dimension contain the impedance
                matrix for that frequency
        '''
        no_freq_dim = False
        if isinstance(freqs, float) or isinstance(freqs, complex):
            freqs = np.array([freqs])
            no_freq_dim = True
        if channel_names is None:
            channel_names = ['L'] + self.channel_storage.keys()
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
            g_terms = node.calcMembraneConductanceTerms(freqs=freqs,
                                                    channel_names=channel_names)
            s_mat[:,ii,ii] += sum([node.currents[c_name][0] * g_term \
                                   for c_name, g_term in g_terms.iteritems()])
        if indexing == 'locs':
            return self._permuteToLocs(s_mat[0,:,:]) if no_freq_dim else \
                   self._permuteToLocs(s_mat)
        elif indexing == 'tree':
            return s_mat[0,:,:] if no_freq_dim else s_mat
        else:
            raise ValueError('invalid argument for `indexing`, ' + \
                             'has to be \'tree\' or \'locs\'')


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
            e_eqs = np.array([self.getEEq()])
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


    def computeGMC(self, z_mat_arg, e_eqs=None, channel_names=None):
        '''
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
        '''
        z_mat_arg = self._preprocessZMatArg(z_mat_arg)
        e_eqs, _ = self._preprocessEEqs(e_eqs)
        assert len(z_mat_arg) == len(e_eqs)
        if channel_names is None:
            channel_names = ['L'] + self.channel_storage.keys()
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
        res = la.lstsq(mat_feature, vec_target)
        g_vec = res[0].real
        # set the conductances
        self._toTreeGMC(g_vec, channel_names)

    def _toStructureTensorGMC(self, channel_names):
        g_vec = self._toVecGMC(channel_names)
        g_struct = np.zeros((len(self), len(self), len(g_vec)))
        kk = 0 # counter
        for node in self:
            ii = node.index
            g_terms = node.calcMembraneConductanceTerms(0.,
                                        channel_storage=self.channel_storage,
                                        channel_names=channel_names)
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
        '''
        Place all conductances to be fitted in a single vector
        '''
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

    def _preprocessExpansionPoints(self, svs, e_eqs):
        if svs is None:
            svs = [None for _ in e_eqs]
        elif isinstance(svs, list):
            svs = np.array(svs)
            assert svs.shape[0] == e_eqs.shape[0]
        elif isinstance(svs, np.ndarray):
            assert svs.shape[0] == e_eqs.shape[0]
        else:
            raise ValueError('wrong state variable array')
        return svs

    def computeGMv2(self, z_mat_arg, e_eqs=None, freqs=0., svs=None,
                    w_e_eqs=None, w_freqs=None,
                    channel_name=None):
        '''
        Fit the models' conductances to a given impedance matrix.

        Parameters
        ----------
        z_mat_arg: np.ndarray (ndim = 2 or 3, dtype = float or complex) or
                   list of np.ndarray (ndim = 2 or 3, dtype = float or complex)
            If a single array, represents the steady state impedance matrix,
            If a list of arrays, represents the steady state impedance
            matrices for each equilibrium potential in ``e_eqs``
        e_eqs: np.ndarray (ndim = 1, dtype = float) or float
            The equilibirum potentials in each compartment for each
            evaluation of ``z_mat``
        freqs: ``None`` or `np.array` of `complex
            Frequencies at which the impedance matrices are evaluated. If None,
            assumes that the steady state impedance matrices are provides
        channel_names: ``None`` or `list` of `string`
            The channel types to be included in the fit. If ``None``, all channel
            types that have been added to the tree are included.
        other_channel_names: ``None`` or `list` of `string`
            The channels that are not to be included in the fit
        '''
        z_mat_arg = self._preprocessZMatArg(z_mat_arg)
        e_eqs, w_e_eqs = self._preprocessEEqs(e_eqs, w_e_eqs)
        assert len(z_mat_arg) == len(e_eqs)
        freqs, w_freqs, z_mat_arg = self._preprocessFreqs(freqs, w_freqs=w_freqs, z_mat_arg=z_mat_arg)
        svs = self._preprocessExpansionPoints(svs, e_eqs)
        channel_names, other_channel_names = [channel_name], ['L']
        # do the fit
        mats_feature = []
        vecs_target = []
        for z_mat, e_eq, sv, w_e_eq in zip(z_mat_arg, e_eqs, svs, w_e_eqs):
            # set equilibrium conductances
            self.setEEq(e_eq)
            # set channel expansion point
            self.setExpansionPoints({channel_name: sv})
            # feature matrix
            g_struct = self._toStructureTensorGM(freqs=freqs, channel_names=channel_names)
            tensor_feature = np.einsum('oij,ojkl->oikl', z_mat, g_struct)
            tensor_feature *= w_freqs[:,np.newaxis,np.newaxis,np.newaxis]
            tshape = tensor_feature.shape
            mat_feature_aux = np.reshape(tensor_feature,
                                         (tshape[0]*tshape[1]*tshape[2], tshape[3]))
            # target vector
            g_mat = self.calcSystemMatrix(freqs, channel_names=other_channel_names,
                                                 indexing='tree')
            zg_prod = np.einsum('oij,ojk->oik', z_mat, g_mat)
            mat_target_aux = np.eye(len(self))[np.newaxis,:,:] - zg_prod
            mat_target_aux *= w_freqs[:,np.newaxis,np.newaxis]
            vec_target_aux = np.reshape(mat_target_aux, (tshape[0]*tshape[1]*tshape[2],))
            # store feature matrix and target vector for this voltage
            mats_feature.append(mat_feature_aux * np.sqrt(w_e_eq))
            vecs_target.append(vec_target_aux * np.sqrt(w_e_eq))
        mat_feature = np.concatenate(mats_feature)
        vec_target = np.concatenate(vecs_target)
        # linear regression fit
        res = la.lstsq(mat_feature, vec_target)
        g_vec = res[0].real
        # set the conductances
        self._toTreeGM(g_vec, channel_names=channel_names)

    def computeGM(self, z_mat_arg, e_eqs=None, freqs=0.,
                    w_e_eqs=None, w_freqs=None,
                    channel_names=None, other_channel_names=None):
        '''
        Fit the models' conductances to a given impedance matrix.

        Parameters
        ----------
        z_mat_arg: np.ndarray (ndim = 2 or 3, dtype = float or complex) or
                   list of np.ndarray (ndim = 2 or 3, dtype = float or complex)
            If a single array, represents the steady state impedance matrix,
            If a list of arrays, represents the steady state impedance
            matrices for each equilibrium potential in ``e_eqs``
        e_eqs: np.ndarray (ndim = 1, dtype = float) or float
            The equilibirum potentials in each compartment for each
            evaluation of ``z_mat``
        freqs: ``None`` or `np.array` of `complex
            Frequencies at which the impedance matrices are evaluated. If None,
            assumes that the steady state impedance matrices are provides
        channel_names: ``None`` or `list` of `string`
            The channel types to be included in the fit. If ``None``, all channel
            types that have been added to the tree are included.
        other_channel_names: ``None`` or `list` of `string`
            The channels that are not to be included in the fit
        '''

        z_mat_arg = self._preprocessZMatArg(z_mat_arg)
        e_eqs, w_e_eqs = self._preprocessEEqs(e_eqs, w_e_eqs)
        assert len(z_mat_arg) == len(e_eqs)
        freqs, w_freqs, z_mat_arg = self._preprocessFreqs(freqs, w_freqs=w_freqs, z_mat_arg=z_mat_arg)
        if channel_names is None:
            channel_names = ['L'] + self.channel_storage.keys()
        if other_channel_names == None:
            other_channel_names = list(set(self.channel_storage.keys()) - set(channel_names))
        # do the fit
        mats_feature = []
        vecs_target = []
        for z_mat, e_eq, w_e_eq in zip(z_mat_arg, e_eqs, w_e_eqs):
            # set equilibrium conductances
            self.setEEq(e_eq)
            # feature matrix
            g_struct = self._toStructureTensorGM(freqs=freqs, channel_names=channel_names)
            tensor_feature = np.einsum('oij,ojkl->oikl', z_mat, g_struct)
            tensor_feature *= w_freqs[:,np.newaxis,np.newaxis,np.newaxis]
            tshape = tensor_feature.shape
            mat_feature_aux = np.reshape(tensor_feature,
                                         (tshape[0]*tshape[1]*tshape[2], tshape[3]))
            # target vector
            g_mat = self.calcSystemMatrix(freqs, channel_names=other_channel_names,
                                                 indexing='tree')
            zg_prod = np.einsum('oij,ojk->oik', z_mat, g_mat)
            mat_target_aux = np.eye(len(self))[np.newaxis,:,:] - zg_prod
            mat_target_aux *= w_freqs[:,np.newaxis,np.newaxis]
            vec_target_aux = np.reshape(mat_target_aux, (tshape[0]*tshape[1]*tshape[2],))
            # store feature matrix and target vector for this voltage
            mats_feature.append(mat_feature_aux * np.sqrt(w_e_eq))
            vecs_target.append(vec_target_aux * np.sqrt(w_e_eq))
        mat_feature = np.concatenate(mats_feature)
        vec_target = np.concatenate(vecs_target)
        # linear regression fit
        res = la.lstsq(mat_feature, vec_target)
        g_vec = res[0].real
        # set the conductances
        self._toTreeGM(g_vec, channel_names=channel_names)

    def _toStructureTensorGM(self, freqs, channel_names):
        g_vec = self._toVecGM(channel_names)
        g_struct = np.zeros((len(freqs), len(self), len(self), len(g_vec)), dtype=freqs.dtype)
        kk = 0 # counter
        for node in self:
            ii = node.index
            g_terms = node.calcMembraneConductanceTerms(freqs,
                                        channel_storage=self.channel_storage,
                                        channel_names=channel_names)
            # membrance conductance elements
            for channel_name in channel_names:
                g_struct[:,ii,ii,kk] += g_terms[channel_name]
                kk += 1
        return g_struct

    def _toVecGM(self, channel_names):
        '''
        Place all conductances to be fitted in a single vector
        '''
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

    def computeC(self, freqs, z_mat_arg, e_eqs=None, channel_names=None):
        '''
        Fit the models' capacitances to a given impedance matrix.

        !!! This function assumes that the conductances are already fitted!!!

        Parameters
        ----------
            freqs: np.array (dtype = complex)
                Frequencies at which the impedance matrix is evaluated
            zf_mat: np.ndarray (ndim = 3, dtype = complex)
                The impedance matrix. The first dimension corresponds to the
                frequency, the second and third dimension contain the impedance
                matrix for that frequency
        '''
        z_mat_arg = self._preprocessZMatArg(z_mat_arg)
        if isinstance(freqs, float) or isinstance(freqs, complex):
            freqs = np.array([freqs])
        if e_eqs is None:
            e_eqs = [self.getEEq() for _ in z_mat_arg]
        elif isinstance(e_eqs, float):
            self.setEEq(e_eq)
            e_eqs = [self.getEEq() for _ in z_mat_arg]
        if channel_names is None:
            channel_names = ['L'] + self.channel_storage.keys()
        # convert to 3d matrices if they are two dimensional
        z_mat_arg_ = []
        for z_mat in z_mat_arg:
            if z_mat.ndim == 2:
                z_mat_arg_.append(z_mat[np.newaxis,:,:])
            else:
                z_mat_arg_.append(z_mat)
            assert z_mat_arg_[-1].shape[0] == freqs.shape[0]
        # do the fit
        mats_feature = []
        vecs_target = []
        for zf_mat, e_eq in zip(z_mat_arg, e_eqs):
            # set equilibrium conductances
            self.setEEq(e_eq)
            # compute c structure tensor
            c_struct = self._toStructureTensorC(freqs)
            # feature matrix
            tensor_feature = np.einsum('oij,ojkl->oikl', zf_mat, c_struct)
            tshape = tensor_feature.shape
            mat_feature_aux = np.reshape(tensor_feature, (tshape[0]*tshape[1]*tshape[2], tshape[3]))
            # target vector
            g_mat = self.calcSystemMatrix(freqs, channel_names=channel_names,
                                                 with_ca=False, indexing='tree')
            zg_prod = np.einsum('oij,ojk->oik', zf_mat, g_mat)
            mat_target = np.eye(len(self))[np.newaxis,:,:] - zg_prod
            vec_target_aux = np.reshape(mat_target,(tshape[0]*tshape[1]*tshape[2],))
            # store feature matrix and target vector for this voltage
            mats_feature.append(mat_feature_aux)
            vecs_target.append(vec_target_aux)
        mat_feature = np.concatenate(mats_feature, 0)
        vec_target = np.concatenate(vecs_target)
        # linear regression fit
        res = la.lstsq(mat_feature, vec_target)
        c_vec = res[0].real
        # set the capacitances
        self._toTreeC(c_vec)

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

    def computeGC(self, freqs, zf_mat, z_mat=None):
        '''
        Fit the models' conductances and capacitances to a given impedance matrix
        evaluated at a number of frequency points in the Fourrier domain.

        Parameters
        ----------
            freqs: np.array (dtype = complex)
                Frequencies at which the impedance matrix is evaluated
            zf_mat: np.ndarray (ndim = 3, dtype = complex)
                The impedance matrix. The first dimension corresponds to the
                frequency, the second and third dimension contain the impedance
                matrix for that frequency
            z_mat:  np.ndarray (ndim = 2, dtype = float) or None (default)
                The steady state impedance matrix. If ``None`` is given, the
                function tries to find index of freq = 0 in ``freqs`` to
                determine ``z_mat``. If no such element is found, a
                ``ValueError`` is raised

        Raises
        ------
            ValueError: if no freq = 0 is found in ``freqs`` and no steady state
                impedance matrix is given
        '''
        if z_mat is None:
            try:
                ind0 = np.where(np.abs(freqs) < 1e-12)[0]
                z_mat = zf_mat[ind0,:,:].real
            except IndexError:
                raise ValueError("No zero frequency in `freqs`")
        # compute leak and coupling conductances
        self.computeG(z_mat)
        # compute capacitances
        self.computeC(freqs, zf_mat)

    # def computeGC_(self, freqs, zf_mat):
    #     '''
    #     Trial to fit the models' conductances and capacitances at once.
    #     So far unsuccesful.
    #     '''
    #     gc_struct = self._toStructureTensorGC(freqs)
    #     # fitting matrix for linear model
    #     tensor_feature = np.einsum('oij,ojkl->oikl', zf_mat, gc_struct)
    #     tshape = tensor_feature.shape
    #     mat_feature = np.reshape(tensor_feature,
    #                              (tshape[0]*tshape[1]*tshape[2], tshape[3]))
    #     vec_target = np.reshape(np.array([np.eye(len(self), dtype=complex) for _ in freqs]),
    #                             (len(self)*len(self)*len(freqs),))
    #     # linear regression fit
    #     res = la.lstsq(mat_feature, vec_target)
    #     gc_vec = res[0].real
    #     # set conductances and capacitances
    #     self._toTreeGC(gc_vec)

    # def _toStructureTensorGC(self, freqs):
    #     gc_vec = self._toVecGC()
    #     gc_struct = np.zeros((len(freqs), len(self), len(self), len(gc_vec)), dtype=complex)
    #     for node in self:
    #         ii = node.index
    #         if node.parent_node == None:
    #             # leak conductance elements
    #             gc_struct[:, 0, 0, 0] += 1
    #             # capacitance elements
    #             gc_struct[:, 0, 0, 0] += freqs
    #         else:
    #             kk = 3 * node.index - 1
    #             jj = node.parent_node.index
    #             # coupling conductance elements
    #             gc_struct[:, ii, jj, kk] -= 1.
    #             gc_struct[:, jj, ii, kk] -= 1.
    #             gc_struct[:, jj, jj, kk] += 1.
    #             gc_struct[:, ii, ii, kk] += 1.
    #             # leak conductance elements
    #             gc_struct[:, ii, ii, kk+1] += 1.
    #             # capacitance elements
    #             gc_struct[:, ii, ii, kk+2] += freqs
    #     return gc_struct

    # def _toVecGC(self):
    #     gc_list = []
    #     for node in self:
    #         if node.parent_node is None:
    #             gc_list.extend([node.currents['L'][0], node.ca])
    #         else:
    #             gc_list.extend([node.g_c, node.currents['L'][0], node.ca])
    #     return np.array(gc_list)

    # def _toTreeGC(self, gc_vec):
    #     for ii, node in enumerate(self):
    #         if node.parent_node is None:
    #             node.currents['L'][0] = gc_vec[ii]
    #             node.ca  = gc_vec[ii+1]
    #         else:
    #             node.g_c = gc_vec[3*ii-2]
    #             node.currents['L'][0] = gc_vec[3*ii-1]
    #             node.ca  = gc_vec[3*ii]

    def computeGChan(self, v_mat, i_mat,
                     channel_names=None, other_channel_names=None):
        '''
        Parameters
        ----------
        v_mat: np.ndarray (n,k)
        i_mat: np.ndarray (n,k)
            n = nr. of locations, k = nr. of fit points
        '''
        # check size
        assert v_mat.shape == i_mat.shape
        assert v_mat.shape[0] == len(self)
        n_loc, n_fp, n_chan = len(self), i_mat.shape[1], len(channel_names)
        # create lin fit arrays
        i_vec = np.zeros((n_loc, n_fp))
        d_vec = np.zeros((n_loc, n_fp, n_chan))
        # iterate over number of fit points
        for jj, (i_, v_) in enumerate(zip(i_mat.T, v_mat.T)):
            i_aux, d_aux = self._toVecGChan(i_, v_,
                                            channel_names=channel_names,
                                            other_channel_names=other_channel_names)
            i_vec[:,jj] = i_aux
            d_vec[:,jj,:] = d_aux

        # iterate over locations:
        g_chan = np.zeros((n_loc, n_fp))
        for ll, (i_, d_) in enumerate(zip(i_vec, d_vec)):
            node = self[ll]
            # conductance fit at node ll
            g_ = la.lstsq(d_, i_)[0]
            # store the conductances
            for ii, channel_name in enumerate(channel_names):
                node.currents[channel_name][0] = g_[ii]

    def _toVecGChan(self, i_, v_,
                    channel_names=None, other_channel_names=None):
        self.setEEq(v_)
        i_vec = np.zeros(len(self))
        d_vec = np.zeros((len(self), len(channel_names)))
        for ii, node in enumerate(self):
            i_vec[ii] += i_[node.loc_ind]
            # add the channel terms
            i_vec[ii] -= node.getITot(channel_names=other_channel_names,
                                      channel_storage=self.channel_storage)
            # add the coupling terms
            pnode = node.parent_node
            if pnode is not None:
                i_vec[ii] += node.g_c * (v_[pnode.loc_ind] - v_[node.loc_ind])
            for cnode in node.child_nodes:
                i_vec[ii] += cnode.g_c * (v_[cnode.loc_ind] - v_[node.loc_ind])
            # drive terms
            for kk, channel_name in enumerate(channel_names):
                d_vec[ii, kk] = node.getDrive(channel_name,
                                              channel_storage=self.channel_storage)

        return i_vec, d_vec

    def computeFakeGeometry(self, fake_c_m=1., fake_r_a=100.*1e-6,
                                  factor_r_a=1e-6, delta=1e-14,
                                  method=2):
        '''
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
        '''

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












