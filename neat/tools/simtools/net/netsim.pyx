"""
File contains:

    - :class:`NETSim`

Author: W. Wybo
"""


cimport numpy as np
import numpy as np

from libcpp.string cimport string
from libc.stdint cimport int16_t, int32_t, int64_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.deque cimport deque
from libcpp.map cimport map
from libcpp.pair cimport pair

import copy
import time
import warnings

from neat.channels.channelcollection import channelcollection


def c2r(arr_c):
    return np.concatenate((arr_c.real[:, np.newaxis], arr_c.imag[:, np.newaxis]), 1)

cdef extern from "NETC.h":
    cdef cppclass NETSimulator:
        NETSimulator(int n_loc, double *v_eq)

        # initialize from python
        void init_from_python(double dt, double integ_mode, bool print_tree)
        void add_node_from_python(int node_index, int parent_index,
                               int64_t *child_indices, int n_children,
                               int64_t *loc_idxices, int n_locinds,
                               int64_t *newloc_idxices, int n_newlocinds,
                               double *alphas, double *gammas, int n_exp)
        void add_lin_term_from_python(int loc_idxex,
                                double *alphas, double *gammas, int n_exp)
        void add_ionchannel_from_python(string channel_name, int loc_idx,
                                     double g_bar, double e_rev,
                                     bool instantaneous, double *vs, int v_size);
        void add_synapse_from_type(int loc_idx, int syn_type)
        void add_synapse_from_params(int loc_idx, double e_r,
                                double *params, int p_size)
        void reset()
        # other structure functions
        void remove_synapse_from_index(int loc_idx, int syn_ind)
        # getter functions for voltage
        void add_v_loc_to_arr(double *v_arr, int v_size)
        void add_v_node_to_arr(double *v_arr, int v_size)
        double get_v_single_loc(int loc_idxex)
        double get_v_single_node(int node_index)
        double get_g_single_syn(int loc_idxex, int syn_index)
        double get_surface_single_syn(int loc_idxex, int syn_index)
        void set_v_node_from_v_loc(double *v_arr, int v_size)
        void set_v_node_from_v_node(double *v_arr, int v_size)
        # integration functions
        void solve_matrix()
        void construct_matrix(double dt, double* mat, double* arr, int n_node);
        void set_inputs_to_zero()
        void construct_input_syn_1_loc(int loc_idx, double v_m,
                                double *g_syn, int g_size)
        void construct_input_chan_1_loc(int loc_idx, double v_m)
        void advance(double dt)
        void feedSpike(int loc_idx, int syn_ind, double g_max, int n_spike)


cdef class NETSim:
    cdef NETSimulator *net_ptr
    # numbers
    cdef public int n_loc
    cdef public int n_node
    cdef public np.ndarray n_syn
    # equilibirum potentials
    cdef public np.ndarray v_eq
    # flag indicating Newton solver (0) or integration with full kernels (1)
    cdef int mode
    # store the synapses, conductances and spikes
    cdef public list syn_map_py
    cdef public list spike_times_py

    def __cinit__(self, net, lin_terms=None, v_eq=-75., print_tree=False):
        self.n_loc = len(net.root.loc_idxs)
        # equibrium potentials
        if isinstance(v_eq, float):
            self.v_eq = v_eq * np.ones(self.n_loc)
        elif len(v_eq) == self.n_loc:
            self.v_eq = np.array(v_eq)
        else:
            raise IOError("Invalid value for input argument [v_eq] should be float " + \
                          "or iterable of floats")
        cdef np.ndarray[np.double_t, ndim=1] v_eq_arr = self.v_eq
        # temporary python objects
        self.syn_map_py = []
        self.spike_times_py = []
        # create the cnet
        self.n_node = len(net)
        for node in net:
            if node.parent_node is not None:
                pnode_ind = node.parent_node.index
            else:
                pnode_ind = -1
                # initialize the C++ object
                self.net_ptr = new NETSimulator(self.n_loc, &v_eq_arr[0])
                self.n_syn = np.zeros(self.n_loc, dtype=int)
            cnode_inds = np.array([cnode.index for cnode in node.child_nodes], dtype=np.int64)
            if len(cnode_inds) == 0: cnode_inds = np.array([-1], dtype=np.int64)
            locinds = np.array(list(node.loc_idxs), dtype=np.int64) \
                                if len(node.loc_idxs) > 0 else np.array([-1], dtype=np.int64)
            newlocinds = np.array(list(node.newloc_idxs), dtype=np.int64) \
                                if len(node.newloc_idxs) > 0 else np.array([-1], dtype=np.int64)
            (alphas, gammas) = (-node.z_kernel.a, node.z_kernel.c)
            alphas = alphas.astype(complex); gammas = gammas.astype(complex)
            self._add_node(node.index, pnode_ind,
                            cnode_inds,
                            locinds,
                            newlocinds,
                            c2r(alphas), c2r(gammas))
        # add the linear terms
        if lin_terms is not None:
            if isinstance(lin_terms, dict):
                for ind, lin_term in lin_terms.iteritems():
                    alphas, gammas = -lin_term.a, lin_term.c
                    alphas, gammas = alphas.astype(complex), gammas.astype(complex)
                    self._add_lin_term(ind, c2r(alphas), c2r(gammas))
            else:
                raise IOError('`lin_terms` should be dict')
        self.initialize(mode=0, print_tree=print_tree)

    def initialize(self, dt=.1, mode=1, print_tree=False):
        self.mode = mode
        self.net_ptr.init_from_python(dt, mode, print_tree)

    def _add_node(self, node_index, parent_index,
                       np.ndarray[np.int64_t, ndim=1] child_indices,
                       np.ndarray[np.int64_t, ndim=1] loc_idxices,
                       np.ndarray[np.int64_t, ndim=1] newloc_idxices,
                       np.ndarray[np.double_t, ndim=2] alphas,
                       np.ndarray[np.double_t, ndim=2] gammas):
        self.net_ptr.add_node_from_python(node_index, parent_index,
                                   &child_indices[0], child_indices.shape[0],
                                   &loc_idxices[0], loc_idxices.shape[0],
                                   &newloc_idxices[0], newloc_idxices.shape[0],
                                   &alphas[0,0], &gammas[0,0], alphas.shape[0])

    def _add_lin_term(self, loc_idxex,
                       np.ndarray[np.double_t, ndim=2] alphas,
                       np.ndarray[np.double_t, ndim=2] gammas):
        self.net_ptr.add_lin_term_from_python(loc_idxex,
                                    &alphas[0,0], &gammas[0,0], alphas.shape[0])

    def set_v_node_from_v_node(self, v_node):
        """
        Set the node voltage

        Parameters
        ----------
        v_loc : ndarray (`ndim` = 1, `size` = ``n_node``)
            node voltage
        """
        if v_node.shape[0] != self.n_node:
            raise ValueError('Input has incorrect size')
        cdef np.ndarray[np.double_t, ndim=1] vc_arr = v_node
        self.net_ptr.set_v_node_from_v_node(&vc_arr[0], vc_arr.shape[0])

    def set_v_node_from_v_loc(self, v_loc):
        """
        Set the node voltage starting from the location voltage array. This
        transformation is not unique. The implementation chosen here is so that
        nodes which integrate new locations receive all of that locations
        voltage and the underlying nodes none

        Parameters
        ----------
        v_loc : ndarray (`ndim` = 1, `size` = ``n_loc``)
            location voltage
        """
        if v_loc.shape[0] != self.n_loc:
            raise ValueError('Input has incorrect size')
        cdef np.ndarray[np.double_t, ndim=1] vc_arr = v_loc
        self.net_ptr.set_v_node_from_v_loc(&vc_arr[0], vc_arr.shape[0])

    def get_v_loc(self):
        """
        Returns the location voltages

        Returns
        -------
        numpy.ndarray (`ndim` = 1, `size` = ``n_loc``)
        """
        cdef np.ndarray[np.double_t, ndim=1] vc_arr = np.zeros((self.n_loc,), dtype=float)
        self.net_ptr.add_v_loc_to_arr(&vc_arr[0], vc_arr.shape[0])
        return vc_arr

    def add_v_loc_to_arr(self, v_arr):
        """
        Add the location voltages to the input array

        Parameters
        ----------
        v_arr : ndarray (`ndim` = 1, `size` = ``n_loc``)
        """
        if v_arr.shape[0] != self.n_loc:
            raise ValueError('Input has incorrect size')
        cdef np.ndarray[np.double_t, ndim=1] vc_arr = v_arr
        self.net_ptr.add_v_loc_to_arr(&vc_arr[0], vc_arr.shape[0])

    def get_v_node(self):
        """
        Returns the NET node voltages

        Returns
        -------
        numpy.ndarray (`ndim` = 1, `size` = ``n_node``)
        """
        cdef np.ndarray[np.double_t, ndim=1] vc_arr = np.zeros((self.n_node,), dtype=float)
        self.net_ptr.add_v_node_to_arr(&vc_arr[0], vc_arr.shape[0])
        return vc_arr

    def add_v_node_to_arr(self, v_arr):
        """
        Add the NET node voltages to the input array

        Parameters
        ----------
        v_arr : ndarray (`ndim` = 1, `size` = ``n_node``)
        """
        if v_arr.shape[0] != self.n_node:
            raise ValueError('Input array has incorrect size')
        cdef np.ndarray[np.double_t, ndim=1] vc_arr = v_arr
        self.net_ptr.add_v_node_to_arr(&vc_arr[0], vc_arr.shape[0])

    def add_channel(self, chan, loc_idxex, g_max, e_rev,
                         instantaneous=False, v_const={}):
        """
        Add an ion channel to the NET model

        Parameters
        ----------
        chan: `neat.channels.IonChannel
            name of the ion channel
        loc_idxex: int
            index of the location
        g_max: float
            maximal conductance of the channel
        e_rev: float
            reversal potential (if not provide, default value is taken)
        instantaneous: bool (optional, default to 'False')
            whether the activation should be approximated as instantaneous if
            such a state variable is present
        v_const: dict {string: float} (optional)
            state variables for which a voltage value is provided (< 1000 mV)
            are fixed at their activation levels for this voltage value during
            the Newton iteration. If the voltage value exceeds 1000 mV or is not
            provided, state variable activations are computed based on actual
            voltage during Newton iteration
        """
        cdef string cname = chan.__class__.__name__.encode('UTF-8')
        # voltage values
        cdef np.ndarray[np.double_t, ndim=1] vs_arr = np.zeros(len(chan.statevars))
        try:
            for ii, svar in enumerate(chan.ordered_statevars):
                sv = str(svar)
                vs_arr[ii] = v_const[sv] if sv in v_const else 1e4
        except AttributeError:
            # backwards compatible with old style channels
            for ii, svar in enumerate(chan.varnames.flatten()):
                vs_arr[ii] = v_const[svar] if svar in v_const else 1e4
        # add the channel
        self.net_ptr.add_ionchannel_from_python(cname, loc_idxex, g_max, e_rev,
                                             instantaneous, &vs_arr[0], vs_arr.shape[0])

    def add_synapse(self, loc_idxex, synarg, g_max=0.001, nmda_ratio=1.6):
        """
        Add a synapse to the NET model

        Parameters
        ----------
        loc_idxex : int
            index of the synapse location
        synarg : float, string or dict
            for `float`, specifies the synaptic reversal potential (mV)
            for `string`, choos 'AMPA', 'NMDA' or 'GABA' or 'AMPA+NMDA'
            for `dict`, contains either three entries ('e_r' (reversal potential),
                'tau_r' (rise time), 'tau_d' (decay time)) or two entries ('e_r'
                (reversal potential), 'tau' (decay time))
        """
        cdef np.ndarray[np.double_t, ndim=1] tau_arr
        if loc_idxex < 0 or loc_idxex >= self.n_loc:
            raise IndexError('`loc_idxex` out of range')
        if isinstance(synarg, str):
            if synarg == 'AMPA+NMDA':
                # add the synapses
                self.net_ptr.add_synapse_from_type(loc_idxex, 0)
                self.net_ptr.add_synapse_from_type(loc_idxex, 1)
                # synapse parameters for simulation
                syn_map = {'loc_idxex': loc_idxex,
                            'syn_index_at_loc': self.n_syn[loc_idxex],
                            'n_syn_at_loc': 2,
                            'g_max': [g_max, nmda_ratio*g_max]}
                self.n_syn[loc_idxex] += 2
            else:
                if synarg == 'AMPA':
                    synarg = 0
                elif synarg == 'NMDA':
                    synarg = 1
                elif synarg == 'GABA':
                    synarg = 2
                else:
                    raise ValueError('``synarg`` should be \'AMPA\', ' \
                                        '\'NMDA\' or \'GABA\' or \'AMPA+NMDA\'')
                # add the synapse
                self.net_ptr.add_synapse_from_type(loc_idxex, synarg)
                # synapse parameters for simulation
                syn_map = {'loc_idxex': loc_idxex,
                            'syn_index_at_loc': self.n_syn[loc_idxex],
                            'n_syn_at_loc': 1,
                            'g_max': [g_max]}
                self.n_syn[loc_idxex] += 1
        elif isinstance(synarg, float):
            tau_arr = np.array([.2,3.])
            # add the synapse
            self.net_ptr.add_synapse_from_params(loc_idxex, synarg,
                                              &tau_arr[0], tau_arr.shape[0])
            # synapse parameters for simulation
            syn_map = {'loc_idxex': loc_idxex,
                        'syn_index_at_loc': self.n_syn[loc_idxex],
                        'n_syn_at_loc': 1,
                        'g_max': [g_max]}
            self.n_syn[loc_idxex] += 1
        elif isinstance(synarg, dict):
            if 'tau_r' in synarg and 'tau_d' in synarg:
                tau_arr = np.array([synarg['tau_r'], synarg['tau_d']])
            elif 'tau' in synarg:
                tau_arr = np.array([synarg['tau']])
            else:
                raise KeyError('No time scale keys found in `dict` ``synarg`` ' + \
                                'use \'tau\' or \'tau_r\' and \'tau_d\'')
            # add the synapse
            self.net_ptr.add_synapse_from_params(loc_idxex, synarg['e_r'],
                                              &tau_arr[0], tau_arr.shape[0])
            # synapse parameters for simulation
            syn_map = {'loc_idxex': loc_idxex,
                        'syn_index_at_loc': self.n_syn[loc_idxex],
                        'n_syn_at_loc': 1,
                        'g_max': [g_max]}
            self.n_syn[loc_idxex] += 1
        else:
            raise TypeError('``synarg`` should be `string` or `float` or `dict`')

        self.syn_map_py.append(syn_map)
        self.spike_times_py.append([])

    def _get_syn_list_index(self, loc_idxex, syn_index):
        if loc_idxex < 0 or loc_idxex >= self.n_loc:
            raise IndexError('`loc_idxex` out of range')
        if syn_index < 0 or syn_index >= self.n_syn[loc_idxex]:
            raise IndexError('`syn_index` out of range')
        # find the synapse index and return it
        for ii, syn_map in enumerate(self.syn_map_py):
            if syn_map['loc_idxex'] == loc_idxex and \
               syn_map['syn_index_at_loc'] == syn_index:
                return ii
        return None

    def get_cond_surf_from_loc(self, loc_idxex, syn_index):
        return self.net_ptr.get_surface_single_syn(loc_idxex, syn_index)

    def get_cond_surf(self, index):
        return self.net_ptr.get_surface_single_syn(self.syn_map[index]['loc_idxex'], \
                                               self.syn_map[index]['syn_index_at_loc'])

    def remove_synapse_from_loc(self, loc_idxex, syn_index):
        """
        Remove synapse at location ``loc_idxex``, and at position ``syn_index`` in
        the synapse array of that location

        Parameters
        ----------
        loc_idxex : int
            index of the synapse location
        syn_index : int
            index of the synapse
        """
        index = self._get_syn_list_index(loc_idxex, syn_index)
        self.remove_synapse(index)

    def remove_synapse(self, index):
        """
        Remove synapse at position ``index`` in synapse stack

        Parameters
        ----------
        index : int
            index of the synapse
        """
        if not isinstance(index, int):
            raise ValueError('Expected int')
        elif index < 0 or index > len(self.syn_map_py):
            raise IndexError('\'index\' out of range')
        syn_map_del = self.syn_map_py.pop(index)
        self.n_syn[syn_map_del['loc_idxex']] -= syn_map_del['n_syn_at_loc']
        for _ in xrange(syn_map_del['n_syn_at_loc']):
            self.net_ptr.remove_synapse_from_index(syn_map_del['loc_idxex'],
                                                syn_map_del['syn_index_at_loc'])
        # reassign indices of other spike times
        del self.spike_times_py[index]

    def _construct_input(self, v_arr, g_syn):
        """
        constructs the NET solver input for the given conductance and voltage

        Parameters
        ----------
        g_syn : list of np.ndarray (`ndim` = 1)
            list contains ``self.n_loc`` arrays, each containing the
            conductances at the corresponding location
        v_arr : np.ndarray (`ndim` = 1)
            the location voltage
        """
        if v_arr.shape[0] != self.n_loc:
            raise ValueError('`v_arr` has incorrect size')
        if len(g_syn) != self.n_loc:
            raise ValueError('`g_syn` has incorrect size')
        # declare c array
        cdef np.ndarray[np.double_t, ndim=1] g_c
        # reset the arrays
        self.net_ptr.set_inputs_to_zero()
        for ii, (v_a, g_s) in enumerate(zip(v_arr, g_syn)):
            if g_s.size != self.n_syn[ii]:
                raise ValueError('`g_syn[%d]` has incorrect size'%ii)
            if g_s.size > 0:
                g_c = g_s
                self.net_ptr.construct_input_syn_1_loc(ii, v_a, &g_c[0], g_c.shape[0])
            self.net_ptr.construct_input_chan_1_loc(ii, v_a)

    def recast_input(self, g_syn):
        """
        executes the Newton iteration to find the steady state location voltage

        Parameters
        ----------
        g_syn : list of np.ndarray (`ndim`=1) or single iterable over floats
            if list, contains ``self.n_loc`` arrays (`ndim`=1), each containing
            the conductances of synapses at the corresponding location
            if single iterable, contains conductances of each synapse on the
            synapse stack

        Returns
        -------
        list of np.ndarray (`ndim`=1)

        """
        # recast input in other form
        if not hasattr(g_syn[0], '__iter__'):
            g_syn_ = g_syn
            g_syn = [[] for _ in xrange(self.n_loc)]
            for ii, syn_map in enumerate(self.syn_map_py):
                g_syn[syn_map['loc_idxex']].append(g_syn_[ii])
                # automatically assumes second synapse (if present) is NMDA
                # synapse, multiplies maximal conductance with NMDA ratio
                surf_ampa = self.net_ptr.get_surface_single_syn(\
                            syn_map['loc_idxex'], syn_map['syn_index_at_loc'])
                for jj, g_s in enumerate(syn_map['g_max'][1:]):
                    surf_nmda = self.net_ptr.get_surface_single_syn(\
                            syn_map['loc_idxex'], syn_map['syn_index_at_loc']+jj+1)
                    g_syn[syn_map['loc_idxex']].append( \
                                g_syn_[ii] * g_s / syn_map['g_max'][0] * \
                                surf_nmda / surf_ampa)
            g_syn = [np.array(g_s) for g_s in g_syn]
            return g_syn
        else:
            return g_syn

    def get_mat_and_vec(self, dt=0.1):
        # construct the data arrays
        cdef np.ndarray[np.double_t, ndim=2] mat = np.zeros((self.n_node, self.n_node))
        cdef np.ndarray[np.double_t, ndim=1] vec = np.zeros(self.n_node)
        # fill matrix and vector
        self.net_ptr.construct_matrix(dt, &mat[0,0], &vec[0], self.n_node)
        return mat, vec

    def invert_matrix(self):
        self.net_ptr.solve_matrix()

    def solve_newton(self, g_syn,
                           v_eps=.1, n_max=100,
                           v_0=None, v_alt=None,
                           n_iter=0):
        """
        executes the Newton iteration to find the steady state location voltage

        Parameters
        ----------
        g_syn : list of np.ndarray (`ndim` = 1) or single iterable over floats
            if list, contains ``self.n_loc`` arrays (`ndim` = 1), each containing
            the conductances of synapses at the corresponding location
            if single iterable, contains conductances of synapses synapses
            on the synapse stack
        v_eps : float (mV), optional
            the tolerance for convergence
        n_max : int, optional
            maximum iteration number
        v_0 : np.ndarray, optional
            the initial location voltage. De
        v_alt : np.ndarray, optional
            initial location voltage when iteration starting from ``v_0`` fails
            to converge
        n_iter : int
            iteration number, for recursion, do not touch

        Returns
        -------
        np.ndarray (`ndim` = 1)
            the location voltage
        """
        # check if NET simulator is in correct integration mode
        if self.mode != 0:
            self.initialize(mode=0)
        # location voltage
        if v_0 is None:
            v_0 = self.v_eq
            v_alt = np.zeros(self.n_loc)
        if n_iter == 0:
            self.set_v_node_from_v_loc(v_0)
            # recast input in other form
            g_syn = self.recast_input(g_syn)
        # construct the inputs
        self._construct_input(v_0, g_syn)
        # compute solution
        self.net_ptr.solve_matrix()
        v_new = self.get_v_loc()
        # check for instability
        if np.max(np.abs(v_new)) > 1000.:
            # v_0 failed, so try v_alt
            if v_alt is not None:
                self.set_v_node_from_v_loc(v_alt)
                v_new = self.solve_newton(g_syn, v_eps=v_eps,
                                            n_iter=0, n_max=n_max,
                                            v_0=v_alt, v_alt=None)
            else:
                warnings.warn('Newton solver failed to converge')
                return np.array([np.nan for _ in range(self.n_loc)])
        elif np.linalg.norm(v_new - v_0) > v_eps and n_iter < n_max:
            # no convergence yet, move on to next step
            v_new = self.solve_newton(g_syn, v_eps=v_eps,
                                        n_iter=n_iter+1, n_max=n_max,
                                        v_0=v_new, v_alt=v_alt)
        return v_new

    def set_spiketimes_from_loc(self, loc_idxex, syn_index, spike_times):
        """
        Set spike times for a synapse indexed by it's location ``loc_idxex`` and
        position ``syn_index`` at that location

        Parameters
        ----------
        loc_idxex : int
            index of the synapse location
        syn_index : int
            index of the synapse
        spike_times : iterable
            iterable collection of spike times. Needs to be ordered
        """
        index = self._get_syn_list_index(loc_idxex, syn_index)
        self.set_spiketimes(index, spike_times)

    def set_spiketimes(self, index, spike_times):
        """
        Set spike times for a synapse indexed by it's position in the synapse
        stack

        Parameters
        ----------
        index : int
            position of synapse in synapse stack
        spike_times : iterable
            iterable collection of spike times. Needs to be ordered. Can be
            empty.
        """
        if not isinstance(index, int):
            raise ValueError('Expected int')
        elif index < 0 or index > len(self.syn_map_py):
            raise IndexError('\'index\' out of range')
        self.spike_times_py[index] = spike_times

    def run_sim(self, double tmax, double dt, int step_skip=1,
                    bool rec_v_node=False, list rec_g_syn_inds=[], bool pprint=False):
        """
        Run the simulation using the CNET.

        Parameters
        ----------
        tmax : float
            The simulation time
        dt : float
            The simulation time step
        step_skip : int
            The number of time steps to skip between storing the variables
        rec_v_node : bool
            whether to recorde the node voltage
        rec_g_syn_inds : list of inds
            record the conductance of the synapses indexed by the entries in
            this list
        pprint : bool
            whether or not to print information

        Returns
        -------
        dict
            The simulation results. Contains the keys:
            't' : np.ndarray (ndim = 1)
                contain the times at which the variables have been stored
            'v_loc' : np.ndarray (ndim = 2)
                the location voltages during the simulation, dim 1 are the
                locations, dim 2 the simulation time
            'v_node' : np.ndarray (ndim = 2, only if ``rec_v_node`` is ``True``)
                the node voltages during the simulation, dim 1 are the nodes,
                dim 2 the simulation time
            'g_syn' : np.ndarray (ndim = 2, only if ``rec_g_syn_inds`` is not empty)
                the synaptic conductances, dim 1 corresponds to indices in
                ``rec_g_syn_inds`` and dim 2 is the simulation time
        """
        if dt < 0: raise ValueError('time step has to be positive')
        if tmax < dt: raise ValueError('maximum time be larger than time step')
        cdef int k_max = int(tmax / dt)
        cdef int k_store = k_max // step_skip
        if step_skip < 0 or step_skip > k_max:
            raise ValueError('\'step_skip*dt\' has to be larger than \'tmax\'')
        # generic loop indices
        cdef int ii, jj, kk, ll, mm
        # initialize
        self.initialize(dt=dt, mode=1)
        # reset all state variables in the model
        self.net_ptr.reset()
        # convert syn map in efficient c vectors
        cdef int n_syn = len(self.syn_map_py)
        cdef vector[vector[double]] g_max = \
                            [sm['g_max'] for sm in self.syn_map_py]
        cdef vector[int] n_syn_at_loc = \
                            [sm['n_syn_at_loc'] for sm in self.syn_map_py]
        cdef vector[int] syn_index_at_loc = \
                            [sm['syn_index_at_loc'] for sm in self.syn_map_py]
        cdef vector[int] loc_idxex = \
                            [sm['loc_idxex'] for sm in self.syn_map_py]
        # convert spike times to format stored in vector of c deques
        cdef vector[deque[double]] spike_times
        cdef deque[double] spike_deque
        cdef double spike_time
        for ii, spike_times_aux in enumerate(self.spike_times_py):
            spike_times.push_back(spike_deque)
            for spike_time in spike_times_aux:
                spike_times[ii].push_back(spike_time)
            spike_deque.clear()
        # create arrays for storage
        cdef np.ndarray[np.double_t, ndim=1] t_sim = \
                    np.arange(0., (float(k_store*step_skip)-0.5)*dt, dt*step_skip)
        cdef np.ndarray[np.double_t, ndim=2] v_loc = \
                    np.zeros((self.n_loc, k_store), dtype=float)
        cdef np.ndarray[np.double_t, ndim=2] v_node
        if rec_v_node: v_node = np.zeros((self.n_node, k_store), dtype=float)
        cdef dict g_syn = {ii: [np.zeros(k_store) for _ in xrange(n_syn_at_loc[ii])] \
                                                  for ii in rec_g_syn_inds}

        if pprint:
            print('\n>>> Integrating NET model for ' + str(tmax) + ' ms. <<<')
            start = time.process_time()

        # cython loop
        cdef int ss = 0 # spike counter
        cdef double tt = 0.0 # time
        mm = 0 # store counter
        store_ind = step_skip - 1 # index when to store
        for kk in xrange(k_max):
            tt = float(kk) * dt
            # loop over all the synapses to feed input spikes
            for jj in xrange(n_syn):
                ss = 0
                while not spike_times[jj].empty() and spike_times[jj][0] < tt:
                    ss += 1
                    spike_times[jj].pop_front()
                if ss != 0:
                    for ll in xrange(n_syn_at_loc[jj]):
                        self.net_ptr.feedSpike(loc_idxex[jj], syn_index_at_loc[jj]+ll,
                                                g_max[jj][ll], ss)
            # advance the model one time step
            self.net_ptr.advance(dt)
            # store the variables
            if kk % step_skip == store_ind:
                for ll in xrange(self.n_loc):
                    v_loc[ll,mm] = self.net_ptr.get_v_single_loc(ll)
                if rec_v_node:
                    for ll in xrange(self.n_node):
                        v_node[ll,mm] = self.net_ptr.get_v_single_node(ll)
                for ii, g_store in g_syn.iteritems():
                    for ll in xrange(n_syn_at_loc[ii]):
                        g_store[ll][mm] = \
                                self.net_ptr.get_g_single_syn(loc_idxex[ii],
                                                           syn_index_at_loc[ii]+ll)
                mm += 1

        if pprint:
            stop = time.process_time()
            print('>>> Elapsed time: ' + str(stop-start) + ' seconds. <<<\n')

        #return the result
        resdict = {'t': t_sim, 'v_loc': v_loc}
        if rec_v_node:
            resdict['v_node'] = v_node
        if len(rec_g_syn_inds) > 0:
            resdict['g_syn'] = g_syn
        return resdict