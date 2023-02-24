import numpy as np

import matplotlib
import matplotlib.patheffects as patheffects
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from ...trees.morphtree import MorphLoc
from ...trees.phystree import PhysTree
from ...trees.greenstree import GreensTree
from ...trees.sovtree import SOVTree
from ...trees.netree import NET, NETNode, Kernel
from ...channels.ionchannels import SPDict

from ...tools import kernelextraction as ke

import warnings
import copy
import pickle
import concurrent.futures
import contextlib
import multiprocessing
import os
import ctypes

try:
    from ...tools.simtools.neuron import neuronmodel as neurm
except ModuleNotFoundError:
    warnings.warn('NEURON not available', UserWarning)


def cpu_count(use_hyperthreading=True):
    """
    Return number of available cores.
    Makes use of hypterthreading by default.
    """
    if use_hyperthreading:
        return multiprocessing.cpu_count()
    else:
        return multiprocessing.cpu_count() // 2


def consecutive(inds):
    """
    split a list of ints into consecutive sublists
    """
    return np.split(inds, np.where(np.diff(inds) != 1)[0]+1)


def nonnegative_hash(o):
    return ctypes.c_size_t(hash(o)).value


def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    From https://stackoverflow.com/questions/5884066/hashing-a-dictionary
    """

    if isinstance(o, (set, tuple, list)):

        return nonnegative_hash(tuple([make_hash(e) for e in o]))

    elif isinstance(o, np.ndarray):

        return nonnegative_hash(o.tobytes())

    elif not isinstance(o, dict):

        return nonnegative_hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return nonnegative_hash(tuple(frozenset(sorted(new_o.items()))))


def maybe_execute_funcs(
    tree, file_name,
    funcs_args_kwargs=[],
    recompute_cache=False, save_cache=True, pprint=False,
):
    try:
        # ensure that the funcs are recomputed if 'recompute' is true
        if recompute_cache:
            raise IOError

        with open(file_name, 'rb') as file:
            tree_ = pickle.load(file)

        cache_params_dict = {
            "cache_name": tree.cache_name,
            "cache_path": tree.cache_path,
            "save_cache": tree.save_cache,
            "recompute_cache": tree.recompute_cache,
        }

        tree.__dict__.update(tree_.__dict__)
        # set the original cache parameters
        tree.__dict__.update(cache_params_dict)
        del tree_

    except (Exception, IOError, EOFError, KeyError) as err:
        if pprint:
            if recompute_cache:
                logstr = '>>> Force recomputing cache...'
            else:
                logstr = '>>> No cache found, recomputing...'
            print(logstr)

        # execute the functions
        for func, args, kwargs in funcs_args_kwargs:
            func(*args, **kwargs)

        if save_cache:
            with open(file_name, 'wb') as file:
                pickle.dump(tree, file)


def _statevar_is_activating(f_statevar):
    """
    check whether a statevar is activating or inactivating

    Parameters
    ----------
    f_statevar: callable
        the activation function of the state variable
    """
    # test voltage values to check whether state variable is activating or
    # inactivating
    v_test = np.array([-43.22, -32.22])

    sv_test = f_statevar(v_test)
    return sv_test[0] < sv_test[1]


def getTwoVariableHoldingPotentials(e_hs):
    e_hs_aux_act   = list(e_hs)
    e_hs_aux_inact = list(e_hs)
    for ii, e_h1 in enumerate(e_hs):
        for jj, e_h2 in enumerate(e_hs):
            e_hs_aux_act.append(e_h1)
            e_hs_aux_inact.append(e_h2)
    e_hs_aux_act   = np.array(e_hs_aux_act)
    e_hs_aux_inact = np.array(e_hs_aux_inact)

    return e_hs_aux_act, e_hs_aux_inact


def getExpansionPoints(e_hs, channel, only_e_h=False):
    """
    Returns a list of expansion points around which to compute the impedance
    matrix given a set of holding potentials. If the channel has only one state
    variable, the returned expansion points are at the holding potentials, if
    the channels has two state variables, the returned expansions points are
    are different combinations of the state variable values around the holding
    potentials

    Parameters
    ----------
    e_hs: iterable collection
        The holding potentials around which the expansion points are computed
    channel: `neat.channels.ionchannels.IonChannel`
        The ion channels
    only_e_h: bool
        Only applicable for channels with at least two state variables.
        If True, returned expansion points are always for state variable
        combination evaluated at the same holding potential. Otherwise,
        state variable activations are evaluated at different holding potentials.

    Returns
    -------
    sv_hs: dict
        the expansion points at every holding potential
    """
    if len(channel.statevars) == 1 or only_e_h:
        sv_hs = channel.computeVarinf(e_hs)
        sv_hs['v'] = e_hs
    else:
        # create different combinations of holding potentials
        e_hs_aux_act, e_hs_aux_inact = getTwoVariableHoldingPotentials(e_hs)

        sv_hs = SPDict(v=e_hs_aux_act)
        for svar, f_inf in channel.f_varinf.items():
            # check if variable is activation
            if _statevar_is_activating(f_inf): # variable is activation
                sv_hs[str(svar)] = f_inf(e_hs_aux_act)
            else: # variable is inactivation
                sv_hs[str(svar)] = f_inf(e_hs_aux_inact)

    return sv_hs


def asPassiveDendrite(phys_tree, factor_lambda=2., t_calibrate=500.):
        """
        Set the dendrites to be passive compartments. Channel conductances at
        the resting potential are added to passive membrane conductance.

        Parameters
        ----------
        phys_tree: `neat.PhysTree()`
            the neuron model
        factor_lambda: float (optional, defaults to 2.)
            multiplies the numbers of compartments given by the lambda rule (to
            compute resting membrane potential)
        t_calibrate: float (optional, defaults to 500. ms)
            The calibration time for the model (should reach resting potential)

        Returns
        -------
        `neat.PhysTree()`
        """
        dt, t_max = .1, 1.
        # create a biophysical simulation model
        sim_tree = phys_tree.__copy__(new_tree=neurm.NeuronSimTree())
        # compute equilibrium potentials
        sim_tree.initModel(dt=dt, factor_lambda=factor_lambda, t_calibrate=t_calibrate)
        sim_tree.storeLocs([(node.index, .5) for node in phys_tree], 'rec locs')
        res = sim_tree.run(t_max)
        sim_tree.deleteModel()
        v_eqs = [v_m[-1] for v_m in res['v_m']]
        # store the equilbirum potential distribution
        phys_tree.setEEq(v_eqs)
        phys_tree.asPassiveMembrane(node_arg='basal')
        phys_tree.asPassiveMembrane(node_arg='apical')
        phys_tree.setCompTree(eps=1e-2)

        return phys_tree


class FitTreeGF(GreensTree):
    def __init__(self, *args,
            recompute_cache=False,
            save_cache=True,
            cache_name='',
            cache_path='',
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.cache_name = cache_name
        self.cache_path = cache_path
        self.save_cache = save_cache
        self.recompute_cache = recompute_cache

    def setImpedancesInTree(self, freqs, sv_h=None, pprint=False, **kwargs):
        """
        Sets the impedances in the tree.

        Parameters
        ----------
        freqs: np.ndarray of float or complex
            The frequencies at which to evaluate the impedances
        sv_hs: dict of {string: np.ndarray}
            Keys are the channel names and values are numpy arrays that contain
            the expansion point for each ion channel
        pprint: bool (optional, default is ``False``)
            Print info
        """
        if pprint:
            cname_string = ', '.join(list(self.channel_storage.keys()))
            print(f'>>> evaluating impedances with {cname_string}')

        if sv_h is not None:
            # check if exansion point for all channels is defined
            assert sv_h.keys() == self.channel_storage.keys()

            for c_name, sv in sv_h.items():

                # set the expansion point
                for node in self:
                    node.setExpansionPoint(c_name, statevar=sv)

        file_name = os.path.join(
            self.cache_path,
            f"{self.cache_name}cache_{str(make_hash([freqs, sv_h, kwargs]))}.p",
        )

        maybe_execute_funcs(
            self, file_name,
            recompute_cache=self.recompute_cache,
            pprint=pprint,
            save_cache=self.save_cache,
            funcs_args_kwargs=[
                (self.setCompTree, [], {}),
                (self.setImpedance, [freqs], {"pprint": pprint, **kwargs})
            ]
        )

    def calcNETSteadyState(self, root_loc=None, dx=5., dz=5.):
        if root_loc is None: root_loc = (1, .5)
        root_loc = MorphLoc(root_loc, self)
        # distribute locs on nodes
        st_nodes = self.gatherNodes(self[root_loc['node']])
        d2s_loc = self.pathLength(root_loc, (1,0.5))
        net_locs = self.distributeLocsOnNodes(d2s=np.arange(d2s_loc, 5000., dx),
                                   node_arg=st_nodes, name='net eval')
        # compute the impedance matrix for net calculation
        z_mat = self.calcImpedanceMatrix('net eval', explicit_method=False)[0]
        # assert np.allclose(z_mat, z_mat_)
        # derive the NET
        net = NET()
        self._addNodeToNET(0., z_mat[0,0], z_mat, np.arange(z_mat.shape[0]), None, net,
                           alpha=1., dz=dz)
        net.setNewLocInds()

        return net, z_mat

    def _addNodeToNET(self, z_min, z_max, z_mat, loc_inds, pnode, net, alpha=1., dz=20.):
        # compute mean impedance of node
        inds = [[]]
        while len(inds[0]) == 0:
            inds = np.where((z_mat > z_min) & (z_mat < z_max))
            z_max += dz
        z_node = np.mean(z_mat[inds])
        # subtract impedances of parent nodes
        gammas = np.array([z_node])
        self._subtractParentKernels(gammas, pnode)
        # add a node to the tree
        node = NETNode(len(net), loc_inds, z_kernel=(np.array([alpha]), gammas))
        if pnode != None:
            net.addNodeWithParent(node, pnode)
        else:
            net.root = node
        # recursion for following nodes
        d_inds = consecutive(np.where(np.diag(z_mat) > z_max)[0])
        for di in d_inds:
            if len(di) > 0:
                self._addNodeToNET(z_max, z_max+dz, z_mat[di,:][:,di], loc_inds[di], node, net,
                                       alpha=alpha, dz=dz)

    def _subtractParentKernels(self, gammas, pnode):
        if pnode != None:
            gammas -= pnode.z_kernel['c']
            self._subtractParentKernels(gammas, pnode.parent_node)


class FitTreeSOV(SOVTree):
    def __init__(self,
            *args,
            recompute_cache=False,
            save_cache=True,
            cache_name='',
            cache_path='',
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.cache_name = cache_name
        self.cache_path = cache_path
        self.save_cache = save_cache
        self.recompute_cache = recompute_cache

    def setSOVInTree(self, pprint=False, maxspace_freq=100.):
        if pprint:
            print(f'>>> evaluating SOV expansion')

        file_name = os.path.join(
            self.cache_path,
            f"{self.cache_name}cache_{str(make_hash([maxspace_freq]))}.p",
        )

        maybe_execute_funcs(
            self, file_name,
            recompute_cache=self.recompute_cache,
            pprint=pprint,
            save_cache=self.save_cache,
            funcs_args_kwargs=[
                (self.setCompTree, [], {"eps": 1.}),
                (self.calcSOVEquations, [], {
                    "maxspace_freq": maxspace_freq,"pprint": pprint,
                })
            ]
        )


class CompartmentFitter(object):
    """
    Helper class to streamline fitting reduced compartmental models

    Attributes
    ----------
    tree: `neat.PhysTree()`
        The full tree based on which reductions are made
    e_hs: np.array of float
        The holding potentials for which quasi active expansions are computed
    conc_hs: dict ({str: np.array of float})
        The holding concentrations for the concentration dependent channel
        expansion
    freqs: np.array of float or complex (default is ``np.array([0.])``)
        The frequencies at which impedance matrices are evaluated
    cache_name: str (default '')
        name of files in which intermediate trees required for the fit are
        cached.
    cache_path: str (default '')
        specify a path under which the intermediate files are cached.
        Default is empty string, which means that intermediate files are stored
        in the working directory.
    save_cache: bool (default `True`)
        Save the intermediate results in a cache (using `cache_path` and
        `cache_name`).
    recompute_cache: bool (default `False`)
        Forces recomputing the caches.
    """

    def __init__(self, phys_tree,
            e_hs=np.array([-75., -55., -35., -15.]),
            conc_hs={'ca': np.array([0.00010, 0.00012, 0.00014, 0.00016])},
            freqs=np.array([0.]),
            cache_name='', cache_path='',
            save_cache=True, recompute_cache=False,
        ):
        self.tree = phys_tree.__copy__(new_tree=PhysTree())
        self.tree.treetype = 'original'
        # get all channels in the tree
        self.channel_names = self.tree.getChannelsInTree()
        # frequencies for fit
        self.freqs = freqs
        # expansion point holding potentials for fit
        self.e_hs = e_hs
        self.conc_hs = conc_hs

        # cache related params
        self.cache_name = cache_name
        self.cache_path = cache_path
        self.save_cache = save_cache
        self.recompute_cache = recompute_cache

        if len(cache_path) > 0 and not os.path.isdir(cache_path):
            os.makedirs(cache_path)

        # boolean flag that is reset the first time `self.fitPassive` is called
        self.use_all_channels_for_passive = True

    def setCTree(self, loc_arg, extend_w_bifurc=True):
        """
        Store an initial `neat.CompartmentTree`, providing a tree
        structure scaffold for the fit for a given set of locations. The
        locations are also stored on ``self.tree`` under the name 'fit locs'

        Parameters
        ----------
        loc_arg: list of locations or string (see documentation of
                :func:`MorphTree._convertLocArgToLocs` for details)
            The compartment locations
        extend_w_bifurc: bool (optional, default `True`)
            To extend the compartment locations with all intermediate
            bifurcations (see documentation of
            :func:`MorphTree.extendWithBifurcationLocs`).
        """
        locs = self.tree._parseLocArg(loc_arg)
        if extend_w_bifurc:
            locs = self.tree.extendWithBifurcationLocs(locs)
        else:
            warnings.warn(
                'Not adding bifurcations to `loc_arg`, this could ' \
                'lead to inaccurate fits. To add bifurcation, set' \
                'kwarg `extend_w_bifurc` to ``True``'
            )
        self.tree.storeLocs(locs, name='fit locs')

        # create the reduced compartment tree
        self.ctree = self.tree.createCompartmentTree(locs)

        # add currents to compartmental model
        for c_name, channel in self.tree.channel_storage.items():
            e_revs = []
            for node in self.tree:
                if c_name in node.currents:
                    e_revs.append(node.currents[c_name][1])
            # reversal potential is the same throughout the reduced model
            self.ctree.addCurrent(copy.deepcopy(channel), np.mean(e_revs))

        # add concentration mechanisms
        ions = set()
        ion_params = {}
        for node in self.tree:
            for ion, concmech in node.concmechs.items():
                if ion not in ion_params:
                    ion_params[ion] = {}

                for param, val in concmech.items():
                    if param in ion_params:
                        ion_params[ion][param].append(val)
                    else:
                        ion_params[ion][param] = [val]

            ions.union(set(node.concmechs.keys()))

        for ion, params in ion_params.items():
            params = {param: np.mean(pvals) for param, pvals in params.items()}

            for node in self.ctree:
                loc_idx = node.loc_ind
                concmechs = self.tree[locs[loc_idx]['node']].concmechs

                if ion in concmechs:
                    cparams = {
                        pname: pval for pname, pval in concmechs[ion].items()
                    }
                    node.addConcMech(ion, **cparams)

                else:
                    node.addConcMech(ion, **params)

        # set the equilibirum potentials at fit locations
        self.setEEq()

    def createTreeGF(self,
            channel_names=[],
            cache_name_suffix='',
        ):
        """
        Create a `FitTreeGF` copy of the old tree, but only with the
        channels in ``channel_names``. Leak 'L' is included in the tree by
        default.

        Parameters
        ----------
        channel_names: list of strings
            List of channel names of the channels that are to be included in the
            new tree.
        recompute_cache: bool
            Whether or not to force recompute the impedance caches

        Returns
        -------
        `FitTreeGF()`

        """
        # create new tree and empty channel storage
        tree = self.tree.__copy__(
            new_tree=FitTreeGF(
                cache_path=self.cache_path,
                cache_name=self.cache_name + cache_name_suffix,
                save_cache=self.save_cache,
                recompute_cache=self.recompute_cache,
            ),
        )
        tree.channel_storage = {}
        # add the ion channel to the tree
        channel_names_newtree = set()
        for node, node_orig in zip(tree, self.tree):
            node.currents = {}
            g_l, e_l = node_orig.currents['L']
            # add the current to the tree
            node._addCurrent('L', g_l, e_l)
            for channel_name in channel_names:
                try:
                    g_max, e_rev = node_orig.currents[channel_name]
                    node._addCurrent(channel_name, g_max, e_rev)
                    channel_names_newtree.add(channel_name)
                except KeyError:
                    pass

        tree.channel_storage = {
            channel_name: self.tree.channel_storage[channel_name] \
            for channel_name in channel_names_newtree
        }
        tree.setCompTree()

        return tree

    def evalChannel(self, channel_name,
                          recompute=False, pprint=False, parallel=True, max_workers=None):
        """
        Evaluate the impedance matrix for the model restricted to a single ion
        channel type.

        Parameters
        ----------
        channel_name: string
            The name of the ion channel under consideration
        recompute: bool (optional, defaults to ``False``)
            whether to force recomputing the impedances
        pprint:  bool (optional, defaults to ``False``)
            whether to print information
        parallel:  bool (optional, defaults to ``True``)
            whether the models are evaluated in parallel

        Return
        ------
        fit_mats
        """
        locs = self.tree.getLocs('fit locs')
        # find the expansion point parameters for the channel
        channel = self.tree.channel_storage[channel_name]
        sv_h = getExpansionPoints(self.e_hs, channel)

        # create the trees with only a single channel and multiple expansion points
        fit_tree = self.createTreeGF([channel_name],
            cache_name_suffix=f"_{channel_name}_",
        )

        # set the impedances in the tree
        fit_tree.setImpedancesInTree(
            freqs=0.,
            sv_h={channel_name: sv_h},
            pprint=pprint
        )
        # compute the impedance matrix for this activation level
        z_mats = fit_tree.calcImpedanceMatrix(locs)[None,:,:,:]

        # compute the fit matrices for all holding potentials
        fit_mats = []
        for ii, e_h in enumerate(sv_h['v']):
            sv = SPDict({
                str(svar): sv_h[svar][ii] \
                for svar in channel.statevars if str(svar) != 'v'
            })

            # compute the fit matrices
            m_f, v_t = self.ctree.computeGSingleChanFromImpedance(
                channel_name, z_mats[:,ii,:,:], e_h, self.freqs,
                sv=sv, other_channel_names=['L'],
                all_channel_names=self.channel_names,
                action='return'
            )

            # compute open probability to weigh fit matrices
            po_h = channel.computePOpen(e_h, **sv)
            w_f = 1. / po_h

            fit_mats.append([m_f, v_t, w_f])

        # fit the model for this channel
        w_norm = 1. / np.sum([w_f for _, _, w_f in fit_mats])
        for _, _, w_f in fit_mats: w_f /= w_norm

        return fit_mats

    def fitChannels(self, recompute=False, pprint=False, parallel=True):
        """
        Fit the active ion channel parameters

        Parameters
        ----------
        recompute: bool (optional, defaults to ``False``)
            whether to force recomputing the impedances
        pprint:  bool (optional, defaults to ``False``)
            whether to print information
        parallel:  bool (optional, defaults to ``True``)
            whether the models are evaluated in parallel
        """
        # create the fit matrices for each channel
        n_arg = len(self.channel_names)

        if n_arg == 0:
            return self.ctree

        args_list = [self.channel_names,
                     [recompute for _ in range(n_arg)],
                     [pprint for _ in range(n_arg)],
                     [parallel for _ in range(n_arg)],
                    ]
        if parallel:
            max_workers = min(n_arg, cpu_count())
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                fit_mats_ = list(pool.map(self.evalChannel, *args_list))
        else:
            fit_mats_ = [self.evalChannel(*args) for args in zip(*args_list)]
        fit_mats = [f_m for f_ms in fit_mats_ for f_m in f_ms]
        # store the fit matrices
        for m_f, v_t, w_f in fit_mats:
            if not (
                np.isnan(m_f).any() or np.isnan(v_t).any() or np.isnan(w_f).any()
            ):
                self.ctree._fitResAction(
                    'store', m_f, v_t, w_f,
                    channel_names=self.channel_names
                )
        # run the fit
        self.ctree.runFit()

        return self.ctree

    def fitConcentration(self, ion, recompute=False, pprint=False):
        for ion, conc_hs in self.conc_hs.items():
            assert len(conc_hs) == len(self.e_hs)
        nh = len(self.e_hs)

        locs = self.tree.getLocs('fit locs')

        e_hs_aux_act, e_hs_aux_inact = getTwoVariableHoldingPotentials(self.e_hs)
        conc_ehs = {ion: np.array(
            [c_ion for c_ion in self.conc_hs[ion]] + \
            [c_ion for c_ion in self.conc_hs[ion] for _ in range(nh)]
        )}

        # only retain channels involved with the ion
        channel_names = []
        sv_hs = {}
        for cname, chan in self.tree.channel_storage.items():
            if chan.ion == ion:
                channel_names.append(cname)

                # if len(chan.statevars) > 1:
                #     sv_h = getExpansionPoints(self.e_hs, chan)
                # else:
                sv_h = getExpansionPoints(e_hs_aux_act, chan, only_e_h=True)

                sv_hs[cname] = sv_h

            elif ion in chan.conc:
                channel_names.append(cname)

                args = chan._argsAsList(
                    e_hs_aux_act, w_statevar=False, **conc_ehs
                )
                sv_h = chan.f_varinf(*args)
                sv_h[ion] = conc_ehs[ion]
                sv_h['v'] = e_hs_aux_act

                sv_hs[cname] = sv_h

        # for cname, sv in sv_hs.items():
        #     print(f'\n{cname}')
        #     for var, vals in sv.items():
        #         print(f'{var}, {vals}')

        # create the trees with the desired channels and expansion points
        fit_tree = self.createTreeGF(
            channel_names,
            cache_name_suffix=ion,
        )

        # set the impedances in the tree
        fit_tree.setImpedancesInTree(
            freqs=0., sv_h=sv_hs, pprint=pprint, use_conc=False,
        )
        # compute the impedance matrix for this activation level
        z_mats = fit_tree.calcImpedanceMatrix(locs)

        print(f"impedance matrices greenstree without conc = \n{z_mats}")


        # set the impedances in the tree
        fit_tree.setImpedancesInTree(
            freqs=0., sv_h=sv_hs, pprint=pprint, use_conc=True,
        )
        # compute the impedance matrix for this activation level
        z_mats = fit_tree.calcImpedanceMatrix(locs)

        print(f"impedance matrices greenstree with conc = \n{z_mats}")

        # # compute the fit matrices for all holding potentials
        # fit_mats = []
        # for ii, e_h in enumerate(sv_h['v']):
        #     svs = []
        #     for cname in channel_names:
        #         sv = {key: val_arr[ii] for key, val_arr in sv_hs[cname] if key != 'v'}
        #         svs.append(sv)

        #         sv = SPDict({
        #             str(svar): sv_h[svar][ii] \
        #             for svar in channel.statevars if str(svar) != 'v'
        #         })

        #     # compute the fit matrices
        #     m_f, v_t = self.ctree.computeConcMech(
        #         z_mats[:,ii,:,:], e_h, self.freqs, ion,
        #         sv=svs, channel_names=channel_names,
        #         action='fit'
        #     )

        #     fit_mats.append([m_f, v_t, w_f])

        print("\n> original tree")
        for node in fit_tree:
            str_repr = f"Node {node.index}"
            for cname, (g, e) in node.currents.items():
                A = 4. * np.pi * (node.R * 1e-4)**2
                str_repr += f" g_{cname} = {g*A} uS,"
            print(str_repr)

        tau_conc = fit_tree[1].concmechs['ca'].tau

        print("\n> fit tree")
        for node in self.ctree:
            print(node)

        self.ctree.computeConcMech(
            z_mats, self.freqs, ion,
            sv_s=sv_hs, channel_names=channel_names, action='fit',
        )

        for node in self.ctree:
            for cm in node.concmechs.values():
                print(A)
                print(cm, cm.gamma * A)
                # print(node.concmechs)


        # # fit the model for this channel
        # w_norm = 1. / np.sum([w_f for _, _, w_f in fit_mats])
        # for _, _, w_f in fit_mats: w_f /= w_norm

        for c_name in sv_hs:
            sv_hs[c_name] = {key: val_arr[:,None] for key, val_arr in sv_hs[c_name].items()}
        # freqs = np.array([1.,10.,100.,1000.])[None,:] * 1j
        # freqs = np.array([1000.])[None,:] * 1j


        from neat.tools import kernelextraction as ke
        # ft = ke.FourrierTools(tarr = np.linspace(0., 50., 1000))
        # freqs = ft.s

        freqs = np.logspace(0, 4, 100) *1j

        # print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

        # # set the impedances in the tree
        # fit_tree.setImpedancesInTree_(
        #     freqs=freqs, sv_h=sv_hs, pprint=pprint, use_conc=True,
        # )
        # # compute the impedance matrix for this activation level
        # z_mats = fit_tree.calcImpedanceMatrix(locs)

        # self.ctree.computeConcMech2(
        #     z_mats, freqs, ion,
        #     sv_s=sv_hs, channel_names=channel_names, action='fit',
        # )

        # c_vecs = np.zeros
        # for node in self.ctree:
        #     for cm in node.concmechs.values():
        #         print(cm)

        # self.ctree._toTreeConc()

    def fitPassive(self, use_all_channels=True, recompute=False, pprint=False):
        """
        Fit the steady state passive model, consisting only of leak and coupling
        conductances, but ensure that the coupling conductances takes the passive
        opening of all channels into account

        Parameters
        ----------
        use_all_channels: bool (optional)
            use leak at rest of all channels combined in the passive fit (passive
            leak has to be refit after capacitance fit)
        recompute: bool (optional, defaults to ``False``)
            whether to force recomputing the impedances
        pprint:  bool (optional, defaults to ``False``)
            whether to print information

        """
        self.use_all_channels_for_passive = use_all_channels

        # get equilibirum potentials
        v_eqs_tree = self.getEEq('tree')
        v_eqs_fit = self.getEEq('fit')

        locs = self.tree.getLocs('fit locs')
        # initialize appropriate greens tree
        channel_names = list(self.tree.channel_storage.keys()) if use_all_channels \
                                                               else []

        suffix = "_pas_"
        if use_all_channels:
            suffix = f"_passified_{'_'.join(channel_names)}_"

        fit_tree = self.createTreeGF(
            channel_names,
            cache_name_suffix=suffix,
        )
        fit_tree.setEEq(v_eqs_tree)
        # set the channels to passive
        fit_tree.asPassiveMembrane()
        # set the impedances in the tree
        fit_tree.setImpedancesInTree(freqs=0., pprint=pprint)
        # compute the steady state impedance matrix
        z_mat = fit_tree.calcImpedanceMatrix(locs)
        # fit the coupling+leak conductances to steady state impedance matrix
        self.ctree.computeGMC(z_mat, channel_names=['L'])

        # print passive impedance matrices
        if pprint:
            z_mat_fit = self.ctree.calcImpedanceMatrix(channel_names=['L'])
            np.set_printoptions(precision=2, edgeitems=10, linewidth=500, suppress=True)
            print('\n----- Impedance matrix comparison -----')
            print('> Zmat orig =')
            print(z_mat)
            print('> Zmat fit  =')
            print(z_mat_fit)
            print('> Zmat diff =')
            print(z_mat - z_mat_fit)
            print('---------------------------------------\n')
            # restore defaults
            np.set_printoptions(precision=8, edgeitems=3, linewidth=75, suppress=False)

        return self.ctree

    def fitPassiveLeak(self, pprint=True):
        """
        Fit leak only. Coupling conductances have to have been fit already.

        Parameters
        ----------
        pprint:  bool (optional, defaults to ``False``)
            whether to print information
        """
        locs = self.tree.getLocs('fit locs')
        # compute the steady state impedance matrix
        fit_tree = self.createTreeGF(
            [],
            cache_name_suffix="_only_leak_",
        )
        # set the impedances in the tree
        fit_tree.setImpedancesInTree(self.freqs, pprint=pprint)
        # compute the steady state impedance matrix
        z_mat = fit_tree.calcImpedanceMatrix(locs)
        # fit the conductances to steady state impedance matrix
        self.ctree.computeGSingleChanFromImpedance('L', z_mat, -75., self.freqs,
                                                   other_channel_names=[],
                                                   action='fit')
        # print passive impedance matrices
        if pprint:
            z_mat_fit = self.ctree.calcImpedanceMatrix(channel_names=['L'])
            np.set_printoptions(precision=2, edgeitems=10, linewidth=500, suppress=True)
            print('\n----- Impedance matrix comparison -----')
            print('> Zmat orig =')
            print(z_mat)
            print('> Zmat fit  =')
            print(z_mat_fit)
            print('> Zmat diff =')
            print(z_mat - z_mat_fit)
            print('---------------------------------------\n')
            # restore defaults
            np.set_printoptions(precision=8, edgeitems=3, linewidth=75, suppress=False)

        return self.ctree

    def createTreeSOV(self, eps=1.):
        """
        Create a `SOVTree` copy of the old tree

        Parameters
        ----------
        channel_names: list of strings
            List of channel names of the channels that are to be included in the
            new tree

        Returns
        -------
        `neat.tools.fittools.compartmentfitter.FitTreeSOV`

        """
        if self.use_all_channels_for_passive:
            cache_name_suffix = '_SOV_allchans_'
        else:
            cache_name_suffix = 'SOV_only_leak_'

        # create new tree and empty channel storage
        tree = self.tree.__copy__(
            new_tree=FitTreeSOV(
                cache_path=self.cache_path,
                cache_name=self.cache_name + cache_name_suffix,
                save_cache=self.save_cache,
                recompute_cache=self.recompute_cache,
            ),
        )
        if not self.use_all_channels_for_passive:
            tree.channel_storage = {}

            for node, node_orig in zip(tree, self.tree):
                node.currents = {}
                g_l, e_l = node_orig.currents['L']
                # add the current to the tree
                node._addCurrent('L', g_l, e_l)

        # set the computational tree
        tree.setCompTree(eps=eps)

        return tree

    def _calcSOVMats(self, locs, pprint=False):
        """
        Use a `neat.SOVTree` to compute SOV matrices for fit
        """
        # create an SOV tree
        sov_tree = self.createTreeSOV()
        # compute the SOV expansion for this tree
        sov_tree.setSOVInTree(pprint=pprint)
        # get SOV constants
        alphas, phimat, importance = sov_tree.getImportantModes(
            locarg=locs, sort_type='importance', eps=1e-12,
            return_importance=True
        )
        alphas = alphas.real
        phimat = phimat.real

        return alphas, phimat, importance, sov_tree

    def fitCapacitance(self, inds=[0], check_fit=True, force_tau_m_fit=False,
                             pprint=False, pplot=False):
        """
        Fit the capacitances of the model to the largest SOV time scale

        Parameters
        ----------
        inds: list of int (optional, defaults to ``[0]``)
            indices of eigenmodes used in the fit. Default is [0], indicating
            the largest eigenmode
        check_fit: bool (optional, default ``True``)
            Check whether the largest eigenmode of the reduced model is within
            tolerance of the largest eigenmode of the full tree. If not,
            capacitances are set to mach membrane time scale
        force_tau_m_fit: bool (optional, default ``False``)
            force capacitance fit through membrance time scale matching
        pprint:  bool (optional, defaults to ``False``)
            whether to print information
        pplot: bool (optional, defaults to ``False``)
            whether to plot the eigenmode timescales
        """
        # compute SOV matrices for fit
        locs = self.tree.getLocs('fit locs')
        alphas, phimat, importance, sov_tree = \
                self._calcSOVMats(locs, pprint=pprint)

        # fit the capacitances from SOV time-scales
        self.ctree.computeC(-alphas[inds]*1e3, phimat[inds,:],
                            weights=importance[inds])

        def calcTau():
            nm = len(locs)
            # original timescales
            taus_orig = np.sort(np.abs(1./alphas))[::-1][:nm]
            # fitted timescales
            lambdas, _, _ = self.ctree.calcEigenvalues()
            taus_fit = np.sort(np.abs(1./lambdas))[::-1]

            return taus_orig, taus_fit

        def calcTauM():
            clocs = [locs[n.loc_ind] for n in self.ctree]
            # original membrane time scales
            taus_m = []
            for l in clocs:
                g_m = sov_tree[l[0]].getGTot(channel_storage=sov_tree.channel_storage)
                taus_m.append(self.tree[l[0]].c_m / g_m *1e3)
            taus_m_orig = np.array(taus_m)
            # fitted membrance time scales
            taus_m_fit = np.array([node.ca / node.currents['L'][0]
                                   for node in self.ctree]) *1e3

            return taus_m_orig, taus_m_fit

        taus_orig, taus_fit = calcTau()
        if (check_fit and np.abs(taus_fit[0] - taus_orig[0]) > .8*taus_orig[0]) or \
           force_tau_m_fit:

            taus_m_orig, taus_m_fit = calcTauM()
            # if fit was not sane, revert to more basic membrane timescale match
            for ii, node in enumerate(self.ctree):
                node.ca = node.currents['L'][0] * taus_m_orig[ii] * 1e-3

            warnings.warn('No sane capacitance fit achieved for this configuragion,' + \
                          'reverted to more basic membrane time scale matching.')

        if pprint:
            # mode time scales
            taus_orig, taus_fit = calcTau()
            # membrane time scales
            taus_m_orig, taus_m_fit = calcTauM()

            np.set_printoptions(precision=2, edgeitems=10, linewidth=500, suppress=False)
            print('\n----- capacitances -----')
            print(('Ca (uF) =\n' + str([nn.ca for nn in self.ctree])))
            print('\n----- Eigenmode time scales -----')
            print(('> Taus original (ms) =\n' + str(taus_orig)))
            print(('> Taus fitted (ms) =\n' + str(taus_fit)))
            print('\n----- Membrane time scales -----')
            print(('> Tau membrane original (ms) =\n' + str(taus_m_orig)))
            print(('> Tau membrane fitted (ms) =\n' + str(taus_m_fit)))
            print('---------------------------------\n')
            # restore default print options
            np.set_printoptions(precision=8, edgeitems=3, linewidth=75, suppress=False)

        else:
            lambdas = None

        if pplot:
            self.plotKernels(alphas, phimat)

        return self.ctree

    def plotSOV(self, alphas=None, phimat=None, importance=None, n_mode=8, alphas2=None):
        fit_locs = self.tree.getLocs('fit locs')

        if alphas is None or phimat is None or importance is None:
            alphas, phimat, importance, _ = self._calcSOVMats(fit_locs,
                                            recompute=False, pprint=False)
        if alphas2 is None:
            alphas2, _, _ = self.ctree.calcEigenvalues()

        fit_locs = self.tree.getLocs('fit locs')
        colours = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
        loc_colours = np.array([colours[ii%len(colours)] for ii in range(len(fit_locs))])
        markers = Line2D.filled_markers

        pl.figure('SOV', figsize=(10,10))
        gs = GridSpec(2,2)
        ax1, ax2, ax3 = pl.subplot(gs[0,0]), pl.subplot(gs[0,1]), pl.subplot(gs[1,:])
        # x axis modes
        x_arr = np.arange(n_mode)
        x_loc = np.arange(len(fit_locs))
        # time scales
        ax1.semilogy(x_arr, np.abs(1./alphas[:n_mode]), 'rD--')
        if alphas2 is not None:
            ax1.semilogy(x_arr[:len(alphas2)], np.sort(np.abs(1./alphas2))[::-1], 'bo--')
        ax1.set_xlabel(r'$k$')
        ax2.set_ylabel(r'$\tau_k$ (ms)')
        # importance
        ax2.semilogy(x_arr, importance[:n_mode], 'rD--')
        ax2.set_xlabel(r'$k$')
        ax2.set_ylabel(r'$I_k$')
        # spatial modes
        for kk in range(n_mode):
            ax3.plot(x_loc, phimat[kk,:], ls='--', c='DarkGrey')
            ax3.scatter(x_loc, phimat[kk,:], c=loc_colours, marker=markers[kk%len(markers)], label=r''+str(kk))
        ax3.set_xlabel(r'$x_i$')
        ax3.set_ylabel(r'$\phi_k(x_i)$')
        ax3.legend(loc=0)

    def _constructKernels(self, a, c):
        nn = len(self.tree.getLocs('fit locs'))
        return [[Kernel((a, c[:,ii,jj])) for ii in range(nn)] for jj in range(nn)]

    def _getKernels(self, alphas=None, phimat=None,
                          pprint=False):
        """
        Returns the impedance kernels as a double nested list of "neat.Kernel".
        The element at the position i,j represents the transfer impedance kernel
        between compartments i and j.

        If one of the arguments is not given, the SOV matrices are computed

        Parameters
        ----------
        alphas: np.array
            The exponential coefficients, as follows from the SOV expansion
        phimat: np.ndarray (dim=2)
            The matrix to compute the exponential prefactors, as follows from
            the SOV expansion
        pprint: bool
            Is verbose if ``True``

        Returns
        -------
        k_orig: list of list of `neat.Kernel`
            The kernels of the full model
        k_comp: list of list of `neat.Kernel`
            The kernels of the reduced model
        """
        fit_locs = self.tree.getLocs('fit locs')
        if alphas is None or phimat is None:
            alphas, phimat, _, _ = self._calcSOVMats(
                fit_locs, recompute=self.recompute_cache, pprint=pprint
            )

        # compute eigenvalues
        alphas_comp, phimat_comp, phimat_inv_comp = \
                                self.ctree.calcEigenvalues(indexing='locs')

        # get the kernels
        k_orig = self._constructKernels(alphas, np.einsum('ik,kj->kij', phimat.T, phimat))
        k_comp = self._constructKernels(-alphas_comp, np.einsum('ik,kj->kij', phimat_comp, phimat_inv_comp))

        return k_orig, k_comp

    def getKernels(self, pprint=False):
        """
        Returns the impedance kernels as a double nested list of "neat.Kernel".
        The element at the position i,j represents the transfer impedance kernel
        between compartments i and j.

        Parameters
        ----------
        pprint: bool
            Is verbose if ``True``

        Returns
        -------
        k_orig: list of list of `neat.Kernel`
            The kernels of the full model
        k_comp: list of list of `neat.Kernel`
            The kernels of the reduced model
        """
        return self._getKernels(recompute=self.recompute_cache, pprint=pprint)

    def plotKernels(self, alphas=None, phimat=None, t_arr=None,
                          pprint=False):
        """
        Plots the impedance kernels.
        The kernel at the position i,j represents the transfer impedance kernel
        between compartments i and j.

        Parameters
        ----------
        alphas: np.array
            The exponential coefficients, as follows from the SOV expansion
        phimat: np.ndarray (dim=2)
            The matrix to compute the exponential prefactors, as follows from
            the SOV expansion
        t_arr: np.array
            The time-points at which the to be plotted kernels are evaluated.
            Default is ``np.linspace(0.,200.,int(1e3))``
        pprint: bool
            Is verbose if ``True``

        Returns
        -------
        k_orig: list of list of `neat.Kernel`
            The kernels of the full model
        k_comp: list of list of `neat.Kernel`
            The kernels of the reduced model
        """
        fit_locs = self.tree.getLocs('fit locs')
        nn = len(fit_locs)

        k_orig, k_comp = self._getKernels(alphas=alphas, phimat=phimat)

        if t_arr is None:
            t_arr = np.linspace(0.,200.,int(1e3))

        pl.figure('Kernels', figsize=(2.*nn, 1.5*nn))
        gs = GridSpec(nn, nn)
        gs.update(top=0.98, bottom=0.04, left=0.04, right=0.98)
        colours = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
        loc_colours = np.array([colours[ii%len(colours)] for ii in range(len(fit_locs))])

        for ii in range(nn):
            for jj in range(ii, nn):
                ko, kc = k_orig[ii][jj], k_comp[ii][jj]
                ax = pl.subplot(gs[ii,jj])
                ax.plot(t_arr, ko(t_arr), c='DarkGrey')
                ax.plot(t_arr, kc(t_arr), ls='--', c=loc_colours[jj])
                # limits
                ax.set_ylim((-0.5, 20.))
                # kernel label
                pstring = '%d $\leftrightarrow$ %d'%(ii,jj)
                ax.set_title(pstring, pad=-10)

    def checkPassive(self, loc_arg, alpha_inds=[0], n_modes=5,
                           use_all_channels_for_passive=True, force_tau_m_fit=False,
                           pprint=False):
        """
        Checks the impedance kernels of the passive model.

        Parameters
        ----------
        loc_arg: list of locations or string (see documentation of
                :func:`MorphTree._convertLocArgToLocs` for details)
            The compartment locations
        alpha_inds: list of ints
            Indices of all mode time-scales to be included in the fit
        n_modes: int
            The number of eigen modes that are shown
        use_all_channels_for_passive: bool
            Uses all channels in the tree to compute coupling conductances
        force_tau_m_fit: bool
            Force using the local membrane time-scale for capacitance fit
        pprint: bool
            is verbose if ``True``

        Returns
        -------
        ``None``
        """
        self.setCTree(loc_arg)
        # fit the passive steady state model
        self.fitPassive(use_all_channels=use_all_channels_for_passive,
                        pprint=pprint)
        # fit the capacitances
        self.fitCapacitance(inds=alpha_inds,
                            force_tau_m_fit=force_tau_m_fit,
                            pprint=pprint, pplot=True)

        fit_locs = self.tree.getLocs('fit locs')
        colours = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
        loc_colours = np.array([colours[ii%len(colours)] for ii in range(len(fit_locs))])

        pl.figure('tree')
        ax = pl.gca()
        locargs = [dict(marker='o', mec='k', mfc=lc, markersize=6.) for lc in loc_colours]
        self.tree.plot2DMorphology(ax, marklocs=fit_locs, locargs=locargs, use_radius=False)

        pl.tight_layout()
        pl.show()

    def getNET(self, c_loc, locs, channel_names=[], pprint=False):
        greens_tree = self.createTreeGF(
            channel_names=channel_names,
            cache_name_suffix="_for_NET_",
        )
        greens_tree.setImpedancesInTree(self.freqs, pprint=False)
        # create the NET
        net, z_mat = greens_tree.calcNETSteadyState(c_loc)
        net.improveInputImpedance(z_mat)

        # prune the NET to only retain ``locs``
        loc_inds = greens_tree.getNearestLocinds([c_loc]+locs, 'net eval')
        net_reduced = net.getReducedTree(loc_inds, indexing='locs')

        return net_reduced

    def calcEEq(self, locs, t_max=500., dt=0.1, factor_lambda=10., ions=[]):
        # create a biophysical simulation model
        sim_tree_biophys = self.tree.__copy__(new_tree=neurm.NeuronSimTree())
        # compute equilibrium potentials
        sim_tree_biophys.initModel(dt=dt, factor_lambda=factor_lambda)
        sim_tree_biophys.storeLocs(locs, 'rec locs', warn=False)
        res_biophys = sim_tree_biophys.run(t_max, record_concentrations=ions)
        sim_tree_biophys.deleteModel()

        return (
            np.array([v_m[-1] for v_m in res_biophys['v_m']]),
            {ion: np.array([ion_eq[-1] for ion_eq in res_biophys[ion]]) for ion in ions}
        )

    def setEEq(self, t_max=500., dt=0.1, factor_lambda=10.):
        """
        Set equilibrium potentials, measured from neuron simulation. Sets the
        `v_eqs_tree` and `v_eqs_fit` attributes, respectively containing the
        equilibrium potentials at (the middle of) each node in the original
        tree and at each of the fit locations

        Parameters
        ----------
        t_max: float
            duration of the neuron simulation
        dt: float
            time-step of the neuron simulation
        factor_lambda: int of float
            if int, signifies the number of segments per section. If float,
            multiplies the number of segments given by the lambda rule with this
            number
        """
        tree_locs = [MorphLoc((n.index, .5), self.tree) for n in self.tree]
        fit_locs = self.tree.getLocs('fit locs')
        # compute equilibrium potentials
        v_eqs = self.calcEEq(tree_locs + fit_locs,
                             t_max=t_max, dt=dt, factor_lambda=factor_lambda)[0]
        # store the equilibrium potentials
        self.v_eqs_tree = {n.index: v for n, v in zip(self.tree, v_eqs)}
        self.v_eqs_fit = v_eqs[len(tree_locs):]

    def getEEq(self, e_eqs_type, **kwargs):
        """
        Get equilibrium potentials. Specify
        `v_eqs_tree` and `v_eqs_fit` attributes, respectively containing the
        equilibrium potentials at (the middle of) each node in the original
        tree and at each of the fit locations

        Parameters
        ----------
        e_eqs_type: 'tree' or 'fit'
            For 'tree', returns the `v_eqs_tree` attribute, containing the
            equilibrium potentials at (the middle of) each node in the original
            tree. For 'fit', returns the `v_eqs_fit` attribute, containing the
            equilibrium potentials at each of the fit locations.
        kwargs: When `v_eqs_tree` or `v_eqs_fit`, have not been set, calls
            ::func::`self.setEEq()` with these `kwargs`

        """
        if not hasattr(self, 'v_eqs_tree') or not hasattr(self, 'v_eqs_fit'):
            self.setEEq(**kwargs)
        if e_eqs_type == 'fit':
            return self.v_eqs_fit
        elif e_eqs_type == 'tree':
            return self.v_eqs_tree
        else:
            raise IOError('``e_eqs_type`` should be \'fit\' or \'tree\'')

    def fitEEq(self, ions=[], **kwargs):
        """
        Fits the leak potentials of the reduced model to yield the same
        equilibrium potentials as the full model

        Parameters
        ----------
        kwargs: When `v_eqs_tree` or `v_eqs_fit`, have not been set, calls
            ::func::`self.setEEq()` with these `kwargs`
        """
        fit_locs = self.tree.getLocs('fit locs')

        # compute equilibrium potentials
        eqs = self.calcEEq(
            fit_locs,
            ions=ions,
            **kwargs
        )

        # set the equilibria
        self.ctree.setEEq(eqs[0])
        for ion in ions:
            self.ctree.setConcEq(ion, eqs[1][ion])

        # fit the leak
        self.ctree.fitEL()

        return self.ctree

    def fitModel(self, loc_arg, alpha_inds=[0], use_all_channels_for_passive=True,
                       pprint=False, parallel=False):
        """
        Runs the full fit for a set of locations (the location are automatically
        extended with the bifurcation locs)

        Parameters
        ----------
        loc_arg: list of locations or string (see documentation of
                :func:`MorphTree._convertLocArgToLocs` for details)
            The compartment locations
        alpha_inds: list of ints
            Indices of all mode time-scales to be included in the fit
        use_all_channels_for_passive: bool (optional, default ``True``)
            Uses all channels in the tree to compute coupling conductances
        pprint:  bool
            whether to print information
        parallel:  bool
            whether the models are evaluated in parallel

        Returns
        -------
        `neat.CompartmentTree`
            The reduced tree containing the fitted parameters
        """
        self.setCTree(loc_arg)
        # fit the passive steady state model
        self.fitPassive(pprint=pprint,
                        use_all_channels=use_all_channels_for_passive)
        # fit the capacitances
        self.fitCapacitance(inds=alpha_inds,
                            pprint=pprint, pplot=False)
        # refit with only leak
        if use_all_channels_for_passive:
            self.fitPassiveLeak(pprint=pprint)

        # fit the ion channel
        self.fitChannels(pprint=pprint, parallel=parallel)
        # fit the resting potentials
        self.fitEEq()

        return self.ctree

    def recalcImpedanceMatrix(self, locarg, g_syns,
                              channel_names=None):
        # process input
        locs = self.tree._parseLocArg(locarg)
        n_syn = len(locs)
        assert n_syn == len(g_syns)
        if n_syn == 0:
            return np.array([[]])
        if channel_names is None:
            channel_names = list(self.tree.channel_storage.keys())
        suffix = '_'.join(channel_names)

        # compute equilibirum potentials
        all_locs = [(n.index, .5) for n in self.tree]
        e_eqs = self.calcEEq(all_locs + locs)[0]
        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.createTreeGF(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_",
        )
        for ii, node in enumerate(greens_tree):
            node.setEEq(e_eqs[ii])
        greens_tree.setImpedancesInTree(self.freqs, pprint=False)
        # compute the impedance matrix of the synapse locations
        z_mat = greens_tree.calcImpedanceMatrix(locs, explicit_method=False)[0].real

        # get the reversal potentials of the synapse locations
        n_all = len(self.tree)
        e_eqs = e_eqs[n_all:]

        # compute the ZG matrix
        gd_mat = np.diag(g_syns)
        zg_mat = np.dot(z_mat, gd_mat)
        z_mat_ = np.linalg.solve(np.eye(n_syn) + zg_mat, z_mat)

        return z_mat_

    def fitSynRescale(self, c_locarg, s_locarg, comp_inds, g_syns, e_revs,
                            fit_impedance=False, channel_names=None):
        """
        Computes the rescaled conductances when synapses are moved to compartment
        locations, assuming a given average conductance for each synapse.

        Parameters
        ----------
        c_locarg: list of locations or string (see documentation of
                  :func:`MorphTree._convertLocArgToLocs` for details)
            The compartment locations
        s_locarg: list of locations or string (see documentation of
                  :func:`MorphTree._convertLocArgToLocs` for details)
            The synapse locations
        comp_inds: list or numpy.array of ints
            for each location in [s_locarg], gives the index of the compartment
            location in [c_locarg] to which the synapse is assigned
        g_syns: list or numpy.array of floats
            The average conductances for each synapse
        e_revs: list or numpy.array of floats
            The reversal potential of each synapse
        fit_impdedance: bool (optional, default `False`)
            Whether to also use the reproduction of the rescaled impedance matrix
            as target.
        channel_names: list of str or `None` (default)
            List of ion channels to be included in impedance matrix calculation.
            `None` includes all ion channels

        Returns
        -------
        g_resc: numpy.array of floats
            The rescale values for the synaptic weights
        """
        # process input
        c_locs = self.tree._parseLocArg(c_locarg)
        s_locs = self.tree._parseLocArg(s_locarg)
        n_comp, n_syn = len(c_locs), len(s_locs)
        assert n_syn == len(g_syns) and n_syn == len(e_revs)
        assert len(c_locs) > 0
        if n_syn == 0:
            return np.array([])
        if channel_names is None:
            channel_names = list(self.tree.channel_storage.keys())
        cs_locs = c_locs + s_locs
        cg_syns = np.concatenate((np.zeros(n_comp), np.array(g_syns)))
        comp_inds, g_syns, e_revs = np.array(comp_inds), np.array(g_syns), np.array(e_revs)

        # compute equilibirum potentials
        all_locs = [(n.index, .5) for n in self.tree]
        e_eqs = self.calcEEq(all_locs + cs_locs)[0]
        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.createTreeGF(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_",
        )
        for ii, node in enumerate(greens_tree):
            node.setEEq(e_eqs[ii])
        greens_tree.setImpedancesInTree(self.freqs, pprint=False)
        # compute the impedance matrix of the synapse locations
        z_mat = greens_tree.calcImpedanceMatrix(cs_locs, explicit_method=False)[0].real
        zc_mat = z_mat[:n_comp, :n_comp]

        # get the reversal potentials of the synapse locations
        n_all = len(self.tree)
        e_cs = e_eqs[n_all:n_all+n_comp]
        e_ss = e_eqs[-n_syn:]

        # compute the ZG matrix
        gd_mat = np.diag(cg_syns)
        zg_mat_ = np.dot(z_mat, gd_mat)
        zg_mat = np.linalg.solve(np.eye(n_comp+n_syn) + zg_mat_, zg_mat_)
        zg_mat = zg_mat[:n_comp,n_comp:]

        # create the compartment assignment matrix & syn index vector
        c_mat = np.array([comp_inds == cc for cc in range(n_comp)]).astype(int)
        s_inds = np.array([np.where(cc > 0)[0][0] for cc in c_mat.T])

        # compute the driving potential vectors
        es_vec = e_revs - e_ss
        ec_vec = e_revs - e_cs[s_inds]

        zc_mat = np.dot(zc_mat, c_mat)
        czg_mat = np.dot(c_mat.T, zg_mat)

        # create matrices for inverse fit
        a1_mat = np.einsum('ck,kn->cnk', zc_mat, np.diag(ec_vec))
        a2_mat = np.einsum('ck,kn->cnk', zc_mat, czg_mat*es_vec[None,:])
        b_mat = zg_mat * es_vec[None,:]

        # unravel first two indices
        a_mat = np.reshape(a1_mat-a2_mat, (n_syn*n_comp,-1))
        b_vec = np.reshape(b_mat, (n_syn*n_comp,))

        if fit_impedance:
            # fit based on impedance matrix
            zr_mat = np.linalg.solve(np.eye(n_comp+n_syn) + zg_mat_, z_mat)

            zr_mat = zr_mat[:n_comp,:n_comp]
            zc_mat = z_mat[:n_comp,:n_comp]

            # b matrix for fit
            b_mat = zc_mat - zr_mat
            # a tensor for fit
            zcc = np.dot(zc_mat, c_mat)
            czr = np.dot(c_mat.T, zr_mat)
            aa_mat = np.einsum('ik,kn->ink', zcc, czr)

            # unravel first two indices
            a_mat_ = np.reshape(aa_mat, (n_comp*n_comp,-1))
            b_vec_ = np.reshape(b_mat, (n_comp*n_comp,))

            # perfor mfit
            a_mat = np.concatenate((a_mat, a_mat_), axis=0)
            b_vec = np.concatenate((b_vec, b_vec_), axis=0)

        # compute rescaled synaptic conductances
        g_resc = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]

        b_arr = g_syns > 1e-9
        g_resc[np.logical_not(b_arr)] = 1.
        g_resc[b_arr] = g_resc[b_arr] / g_syns[b_arr]

        return g_resc

    def assignLocsToComps(self, c_locarg, s_locarg, fz=.8,
                                channel_names=None):
        """
        assumes the root node is in `c_locarg`
        """
        if channel_names is None:
            channel_names = list(self.tree.channel_storage.keys())

        # compute equilibirum potentials
        e_eqs = self.getEEq('tree')
        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.createTreeGF(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_at_rest_",
        )
        for ii, node in enumerate(greens_tree):
            node.setEEq(e_eqs[ii])
        greens_tree.setImpedancesInTree(self.freqs, pprint=False)

        # process input
        c_locs = self.tree._parseLocArg(c_locarg)
        s_locs = self.tree._parseLocArg(s_locarg)
        # find nodes corresponding to locs
        c_nodes = [self.tree[loc['node']] for loc in c_locs]
        s_nodes = [self.tree[loc['node']] for loc in s_locs]
        # compute input impedances
        c_zins = [greens_tree.calcZF(c_loc, c_loc)[0] for c_loc in c_locs]
        s_zins = [greens_tree.calcZF(s_loc, s_loc)[0] for s_loc in s_locs]
        # paths to root
        c_ptrs = [self.tree.pathToRoot(node) for node in c_nodes]
        s_ptrs = [self.tree.pathToRoot(node) for node in s_nodes]

        c_inds = []
        for s_node, s_path, s_loc, s_zin in zip(s_nodes, s_ptrs, s_locs, s_zins):
            z_diffs = []
            # check if there are compartment nodes before bifurcation nodes in up direction
            nn_inds = greens_tree.getNearestNeighbourLocinds(s_loc, c_locs)
            # print c_before_b
            c_ns = [c_nodes[ii] for ii in nn_inds]
            c_ps = [c_ptrs[ii] for ii in nn_inds]
            c_ls = [c_locs[ii] for ii in nn_inds]
            c_zs = [c_zins[ii] for ii in nn_inds]
            for c_node, c_path, c_loc, c_zin in zip(c_ns, c_ps, c_ls, c_zs):
                # find the common node as far from the root as possible
                s_p, c_p = s_path[::-1], c_path[::-1]
                kk = 0
                while kk < min(len(s_p), len(c_p)) and s_p[kk] == c_p[kk]:
                    p_node = s_p[kk]
                    kk += 1
                # distinguish cases for computing impedance different
                if p_node == s_node and p_node != c_node:
                    z_diffs.append(fz*np.abs(c_zin - s_zin))
                elif p_node == c_node and p_node != s_node:
                    z_diffs.append((1.-fz)*np.abs(s_zin - c_zin))
                elif p_node == c_node and p_node == s_node:
                    fz_ = fz if c_loc['x'] > s_loc['x'] else (1.-fz)
                    z_diffs.append(fz_*np.abs(s_zin-c_zin))
                else:
                    b_loc = (p_node.index, 1.)
                    b_z = greens_tree.calcZF(b_loc, b_loc)[0]
                    z_diffs.append((1.-fz)*(c_zin - b_z) + fz * (s_zin - b_z))
            # compartment node with minimal impedance difference
            ind_aux = np.argmin(z_diffs)
            c_inds.append(nn_inds[ind_aux])

        return c_inds

