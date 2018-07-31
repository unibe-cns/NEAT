"""
File contains:

    - :class:`CompartmentNode`
    - :class:`CompartmentTree`

Author: W. Wybo
"""


import numpy as np
import scipy.linalg as la

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

    def __str__(self, with_parent=False, with_children=False):
        node_string = super(CompartmentNode, self).__str__()
        if self.parent_node is not None:
            node_string += ', Parent: ' + super(CompartmentNode, self.parent_node).__str__()
        node_string += ' --- (g_c = ' + str(self.g_c) + ' uS, ' + \
                       ', '.join(['g_' + cname + ' = ' + str(cpar[0]) + ' uS' \
                            for cname, cpar in self.currents.iteritems()]) + \
                       ', c = ' + str(self.ca) + ' uF)'
        return node_string

    def addCurrent(self, current_type, e_rev=None, channel_storage=None):
        if current_type is not 'L':
            if e_rev is None:
                e_rev = channelcollection.E_REV_DICT[current_type]
            self.currents[current_type] = [0., e_rev]
            if channel_storage is not None and current_type not in channel_storage:
                channel_storage[current_type] = \
                    eval('channelcollection.' + current_type + '()')

    def calcMembraneConductanceTerms(self, freqs=None,
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
            impedance term of a channel
        '''
        if channel_names is None: channel_names = self.currents.keys()
        imp_terms = {'L': 1.} # leak conductance has 1 as prefactor
        for channel_name in set(channel_names) - set('L'):
            if channel_name not in self.currents:
                self.addCurrent(channel_name, channel_storage=channel_storage)
            e = self.currents[channel_name][1]
            # create the ionchannel object
            if channel_storage is not None:
                channel = channel_storage[channel_name]
            else:
                channel = eval('channelcollection.' + channel_name + '()')
            # add channel contribution to membrane impedance
            imp_aux = - (e - self.e_eq) * \
                        channel.computeLinear(self.e_eq, freqs)
            imp_aux += channel.computePOpen(self.e_eq)
            imp_terms[channel_name] = imp_aux

        return imp_terms

    def getGTot(self, v=None, channel_names=None, channel_storage=None):
        if channel_names is None: channel_names = self.currents.keys()
        g_tot = self.currents['L'][0] if 'L' in channel_names else 0.
        v = self.e_eq if v is None else v
        for channel_name in channel_names:
            if channel_name != 'L':
                g, e = self.currents[channel_name]
                # create the ionchannel object
                if channel_storage is not None:
                    channel = channel_storage[channel_name]
                else:
                    channel = eval('channelcollection.' + channel_name + '()')
                g_tot += g * channel.computePOpen(v)

        return g_tot

    def setGTot(self, illegal):
        raise AttributeError("`g_tot` is a read-only attribute, set the leak " + \
                             "conductance by calling ``func:addCurrent`` with " + \
                             " \'L\' as `current_type`")

    g_tot = property(getGTot, setGTot)


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
        if isinstance(e_eq, float):
            e_eq = e_eq * np.ones(len(self))
        for ii, node in enumerate(self): node.e_eq = e_eq[ii]

    def getEEq(self):
        return np.array([node.e_eq for node in self])

    def addCurrent(self, current_type, e_rev=None):
        '''
        Add an ion channel current to the tree

        Parameters
        ----------
        current_type: string
            The name of the channel type
        '''
        for node in self:
            node.addCurrent(current_type, e_rev=e_rev,
                            channel_storage=self.channel_storage)

    def calcImpedanceMatrix(self, freqs=0., channel_names=None):
        return np.linalg.inv(self.calcSystemMatrix(freqs=freqs,
                                                channel_names=channel_names))

    def calcConductanceMatrix(self):
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
        return g_mat

    def calcSystemMatrix(self, freqs=0., channel_names=None, with_ca=True):
        '''
        Constructs the matrix of conductance and capacitance terms of the model
        for each frequency provided in ``freqs``

        Parameters
        ----------
            freqs: np.array (dtype = complex)
                Frequencies at which the matrix is evaluated

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
        if no_freq_dim:
            s_mat = s_mat[0,:,:]
        return s_mat

    def computeGMC(self, z_mat_arg, e_eqs=None, channel_names=None):
        '''
        Fit the models' conductances to a given steady state impedance matrix.

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
        if isinstance(z_mat_arg, np.ndarray):
            z_mat_arg = [z_mat_arg]
        if e_eqs is None:
            e_eqs = [self.getEEq() for _ in z_mat_arg]
        elif isinstance(e_eqs, float):
            self.setEEq(e_eq)
            e_eqs = [self.getEEq() for _ in z_mat_arg]
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
        # g_struct = self._toStructureTensorG()
        # # fitting matrix for linear model
        # tensor_feature = np.einsum('ij,jkl->ikl', z_mat, g_struct)
        # tshape = tensor_feature.shape
        # mat_feature = np.reshape(tensor_feature, (tshape[0]*tshape[1], tshape[2]))
        # # linear regression fit
        # res = la.lstsq(mat_feature, vec_target)
        # g_vec = res[0]
        # # set the conductances
        # self._toTreeG(g_vec)

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
            # if node.parent_node is None:
            #     node.currents['L'][0] = g_vec[ii]
            # else:
            #     node.g_c = g_vec[2*ii-1]
            #     node.currents['L'][0] = g_vec[2*ii]


    def computeGM(self, z_mat_arg, e_eqs=None, freqs=0., channel_names=None,
                        other_channel_names=None):
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
        '''
        if isinstance(freqs, float) or isinstance(freqs, complex):
            freqs = np.array([freqs])
        if isinstance(z_mat_arg, np.ndarray):
            z_mat_arg = [z_mat_arg]
        # convert to 3d matrices if they are two dimensional
        z_mat_arg_ = []
        for z_mat in z_mat_arg:
            if z_mat.ndim == 2:
                z_mat_arg_.append(z_mat[np.newaxis,:,:])
            else:
                z_mat_arg_.append(z_mat)
            assert z_mat_arg_[-1].shape[0] == freqs.shape[0]
        z_mat_arg = z_mat_arg_
        if e_eqs is None:
            e_eqs = [self.getEEq() for _ in z_mat_arg]
        elif isinstance(e_eqs, float):
            self.setEEq(e_eq)
            e_eqs = [self.getEEq() for _ in z_mat_arg]
        if channel_names is None:
            channel_names = ['L'] + self.channel_storage.keys()
        if other_channel_names == None:
            other_channel_names = list(set(self.channel_storage.keys()) - set(channel_names))
        # do the fit
        mats_feature = []
        vecs_target = []
        for z_mat, e_eq in zip(z_mat_arg, e_eqs):
            # set equilibrium conductances
            self.setEEq(e_eq)
            # feature matrix
            g_struct = self._toStructureTensorGM(freqs=freqs, channel_names=channel_names)
            tensor_feature = np.einsum('oij,ojkl->oikl', z_mat, g_struct)
            tshape = tensor_feature.shape
            mat_feature_aux = np.reshape(tensor_feature,
                                         (tshape[0]*tshape[1]*tshape[2], tshape[3]))
            # target vector
            g_mat = self.calcSystemMatrix(freqs, channel_names=other_channel_names)
            zg_prod = np.einsum('oij,ojk->oik', z_mat, g_mat)
            mat_target_aux = np.eye(len(self))[np.newaxis,:,:] - zg_prod
            vec_target_aux = np.reshape(mat_target_aux, (tshape[0]*tshape[1]*tshape[2],))
            # store feature matrix and target vector for this voltage
            mats_feature.append(mat_feature_aux)
            vecs_target.append(vec_target_aux)
        mat_feature = np.concatenate(mats_feature, 0)
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

    def computeC(self, freqs, zf_mat, channel_names=None):
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
        c_struct = self._toStructureTensorC(freqs)
        # feature matrix
        tensor_feature = np.einsum('oij,ojkl->oikl', zf_mat, c_struct)
        tshape = tensor_feature.shape
        mat_feature = np.reshape(tensor_feature, (tshape[0]*tshape[1]*tshape[2], tshape[3]))
        # target vector
        g_mat = self.calcSystemMatrix(freqs, channel_names=channel_names,
                                             with_ca=False)
        zg_prod = np.einsum('oij,ojk->oik', zf_mat, g_mat)
        mat_target = np.eye(len(self))[np.newaxis,:,:] - zg_prod
        vec_target = np.reshape(mat_target,(tshape[0]*tshape[1]*tshape[2],))
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

    def computeGC_(self, freqs, zf_mat):
        '''
        Trial to fit the models' conductances and capacitances at once.
        So far unsuccesful.
        '''
        gc_struct = self._toStructureTensorGC(freqs)
        # fitting matrix for linear model
        tensor_feature = np.einsum('oij,ojkl->oikl', zf_mat, gc_struct)
        tshape = tensor_feature.shape
        mat_feature = np.reshape(tensor_feature,
                                 (tshape[0]*tshape[1]*tshape[2], tshape[3]))
        vec_target = np.reshape(np.array([np.eye(len(self), dtype=complex) for _ in freqs]),
                                (len(self)*len(self)*len(freqs),))
        # linear regression fit
        res = la.lstsq(mat_feature, vec_target)
        gc_vec = res[0].real
        # set conductances and capacitances
        self._toTreeGC(gc_vec)

    def _toStructureTensorGC(self, freqs):
        gc_vec = self._toVecGC()
        gc_struct = np.zeros((len(freqs), len(self), len(self), len(gc_vec)), dtype=complex)
        for node in self:
            ii = node.index
            if node.parent_node == None:
                # leak conductance elements
                gc_struct[:, 0, 0, 0] += 1
                # capacitance elements
                gc_struct[:, 0, 0, 0] += freqs
            else:
                kk = 3 * node.index - 1
                jj = node.parent_node.index
                # coupling conductance elements
                gc_struct[:, ii, jj, kk] -= 1.
                gc_struct[:, jj, ii, kk] -= 1.
                gc_struct[:, jj, jj, kk] += 1.
                gc_struct[:, ii, ii, kk] += 1.
                # leak conductance elements
                gc_struct[:, ii, ii, kk+1] += 1.
                # capacitance elements
                gc_struct[:, ii, ii, kk+2] += freqs
        return gc_struct

    def _toVecGC(self):
        gc_list = []
        for node in self:
            if node.parent_node is None:
                gc_list.extend([node.currents['L'][0], node.ca])
            else:
                gc_list.extend([node.g_c, node.currents['L'][0], node.ca])
        return np.array(gc_list)

    def _toTreeGC(self, gc_vec):
        for ii, node in enumerate(self):
            if node.parent_node is None:
                node.currents['L'][0] = gc_vec[ii]
                node.ca  = gc_vec[ii+1]
            else:
                node.g_c = gc_vec[3*ii-2]
                node.currents['L'][0] = gc_vec[3*ii-1]
                node.ca  = gc_vec[3*ii]

    def computeFakeGeometry(self, fake_c_m=1., fake_r_a=100.*1e-6,
                                  factor_r_a=1e-6, delta=1e-14):
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
        radii, lengths, surfaces: np.array of floats
            The radii, lengths, resp. surfaces for the section in NEURON. Array
            index corresponds to NEURON index

        Raises
        ------
        AssertionError
            If the node indices are not ordered consecutively when iterating
        '''

        # c_m = 1.
        # # r_a = 100.
        # gL = 150.

        # L1 = 10.
        # L2 = 10.
        # delta = 1e-10
        # R1 = 1.
        # R2 = 100.

        # factor_r = R2 / R1

        # surfaces = np.ones(len(self)) * 2. * np.pi * R1 * L1 + 2. * np.pi * R2 * L2 + np.pi * (R2**2 - R1**2)
        # vec_coupling = np.ones(len(self)) * 3.18309886184
        # print 'L1 computed =', np.pi * (R1*1e-4)**2 * vec_coupling[0] / (fake_r_a) *1e4
        # print 'surfaces =', surfaces

        assert self.checkOrdered()
        factor_r = 1. / np.sqrt(factor_r_a)
        # compute necessary vectors for calculating
        surfaces = np.array([node.ca / fake_c_m for node in self])
        vec_coupling = np.array([1.] + [1./node.g_c for node in self if \
                                            node.parent_node is not None])

        p0s = -surfaces
        p1s = np.zeros_like(p0s)
        p2s = np.pi * (factor_r**2 - 1.) * np.ones_like(p0s)
        p3s = 2. * np.pi**2 * vec_coupling / fake_r_a * (1. + factor_r)

        # print 1e8*(p3s * (R1*1e-4)**3 + p2s * (R1*1e-4)**2 + p1s * (R1*1e-4))

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

    #     c_aux = fake_r_a / (2.*np.pi)**2
    #     vec_aux = np.array([c_aux * surfaces[node.index] for node in self])
    #     vec_coupling = np.array([0.] + [1./node.g_c for node in self if \
    #                                         node.parent_node is not None])
    #     vec_fac = np.ones_like(vec_coupling)
    #     vec_sol = copy.deepcopy(vec_coupling)
    #     # find a solution for the radii of the compartments
    #     self._solveRadii(self.root, vec_coupling, vec_sol, vec_fac)
    #     x_min = np.max(-vec_sol[np.where(vec_fac > 0.)[0]])
    #     x_max = np.min(vec_sol[np.where(vec_fac < 0.)[0]])
    #     x_sol = (x_min+x_max) / 2.
    #     # compute solution with all positive values
    #     res = (vec_sol + vec_fac * x_sol) / vec_aux
    #     # compute radii and corresponding lengths
    #     radii = (1./res)**(1./3.)
    #     lengths = surfaces / (2. * np.pi * radii)
    #     return radii, lengths, surfaces

    # def _solveRadii(self, node, vec_coupling, vec_sol, vec_fac):
    #     if node.parent_node is not None:
    #         vec_sol[node.index] -= vec_sol[node.parent_node.index]
    #         vec_fac[node.index] = -1. * vec_fac[node.parent_node.index]
    #     for cnode in node.child_nodes:
    #         self._solveRadii(cnode, vec_coupling, vec_sol, vec_fac)













