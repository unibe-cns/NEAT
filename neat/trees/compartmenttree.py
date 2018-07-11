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
        self.currents = {'L': (g_l, e_eq)} # ion channel currents and reversals

    def __str__(self, with_parent=False, with_children=False):
        node_string = super(CompartmentNode, self).__str__()
        if self.parent_node is not None:
            node_string += ', Parent: ' + super(CompartmentNode, self.parent_node).__str__()
        node_string += ' --- (g_c = ' + str(self.g_c) + \
                        ' uS, g_l = ' + str(self.currents['L'][0]) + \
                        ' uS, c = ' + str(self.ca) + ' uF)'
        return node_string

    def addCurrent(self, current_type, e_rev=None, channel_storage=None):
        if e_rev is None:
            e_rev = channelcollection.E_REV_DICT[current_type]
        self.currents[current_type] = (g_max, e_rev)
        if channel_storage is not None and current_type not in channel_storage:
            channel_storage[current_type] = \
                eval('channelcollection.' + current_type + '()')

    def calcMembraneConductanceTerms(self, freqs=None, channel_storage=None):
        '''
        Compute the membrane impedance terms and return them as a list

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
        imp_terms = {'L': 1.} # leak conductance has 1 as prefactor
        for channel_name in set(self.currents.keys()) - set('L'):
            e = self.currents[channel_name][1]
            # create the ionchannel object
            if channel_storage is not None:
                channel = channel_storage[channel_name]
            else:
                channel = eval('channelcollection.' + channel_name + '()')
            # add channel contribution to membrane impedance
            imp_aux -= (e - self.e_eq) * \
                       channel.computeLinear(self.e_eq, freqs)
            imp_aux += channel.computePOpen(self.e_eq)
            imp_terms[channel_name] = imp_aux

        return imp_terms


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

    def getEEq(self, e_eq):
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

    def calcImpedanceMatrix(self, freqs=None):
        return np.linalg.inv(self.calcSystemMatrix(freqs=freqs))

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
            g_mat[ii, ii] += node.currents['L'][0] + node.g_c
            if node.parent_node is not None:
                jj = node.parent_node.index
                g_mat[jj,jj] += node.g_c
                g_mat[ii,jj] -= node.g_c
                g_mat[jj,ii] -= node.g_c
        return g_mat

    def calcSystemMatrix(self, freqs=None):
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
        if freqs is None:
            gc_mat = self.calcConductanceMatrix()
        else:
            gc_mat = self.calcConductanceMatrix().astype(complex)[np.newaxis,:,:] * \
                     np.ones(len(freqs), dtype=complex)[:,np.newaxis,np.newaxis]
            for node in self:
                gc_mat[:, node.index, node.index] += freqs * node.ca
        return gc_mat

    def computeG(self, z_mat_arg, e_eqs=None):
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
        # do the fit
        mats_feature = []
        vecs_target = []
        for z_mat, e_eq in zip(z_mat_arg, e_eqs):
            # set equilibrium conductances
            self.setEEq(e_eq)
            # create the matrices for linear fit
            g_struct = self.toStructureTensorG()
            tensor_feature = np.einsum('ij,jkl-ikl', z_mat, g_struct)
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
        g_vec = res[0]
        # set the conductances
        self._toTreeG(g_vec)
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

    def _toStructureTensorG(self):
        g_vec = self._toVecG()
        g_struct = np.zeros((len(self), len(self), len(g_vec)))
        kk = 0 # counter
        for node in self:
            ii = node.index
            g_terms = node.calcMembraneConductanceTerms(0.,
                                        channel_storage=self.channel_storage)
            if node.parent_node == None:
                # membrance conductance elements
                for channel_name in self.currents:
                    g_struct[0, 0, kk] += g_term[channel_name]
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
                for g_term in imp_terms.itervalues():
                    g_struct[ii, ii, kk] += g_term[channel_name]
                    kk += 1
        return g_struct

    def _toVecG(self):
        '''
        Place all conductances to be fitted in a single vector
        '''
        g_list = []
        for node in self:
            if node.parent_node is None:
                g_list.extend([current[0] for current in node.currents.itervalues()])
            else:
                g_list.extend([node.g_c] + \
                              [current[0] for current in node.currents.itervalues()])
        return np.array(g_list)

    def _toTreeG(self, g_vec):
        kk = 0 # counter
        for ii, node in enumerate(self):
            if node.parent_node is None:
                for channel_name in node.currents:
                    node.currents[channel_name][0] = g_vec[kk]
                    kk += 1
            else:
                node.g_c = g_vec[kk]
                kk += 1
                for channel_name in node.currents:
                    node.currents[channel_name][0] = g_vec[kk]
                    kk += 1
            # if node.parent_node is None:
            #     node.currents['L'][0] = g_vec[ii]
            # else:
            #     node.g_c = g_vec[2*ii-1]
            #     node.currents['L'][0] = g_vec[2*ii]

    def computeC(self, freqs, zf_mat):
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
        g_mat = self.calcConductanceMatrix()
        zg_prod = np.einsum('oij,jk->oik', zf_mat, g_mat)
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







