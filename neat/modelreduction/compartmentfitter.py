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

from ..trees.netree import  Kernel
from ..channels.ionchannels import SPDict
from ..factorydefaults import DefaultFitting, DefaultMechParams
from ..tools import kernelextraction as ke
from .cachetrees import FitTreeGF, FitTreeSOV, FitTreeC, EquilibriumTree

import warnings
import copy
import pickle
import concurrent.futures
import contextlib
import multiprocessing
import os
import ctypes


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
            fit_cfg=None, concmech_cfg=None,
            cache_name='', cache_path='',
            save_cache=True, recompute_cache=False,
        ):
        # cache related params
        self.cache_name = cache_name
        self.cache_path = cache_path
        self.save_cache = save_cache
        self.recompute_cache = recompute_cache

        if len(cache_path) > 0 and not os.path.isdir(cache_path):
            os.makedirs(cache_path)

        # original tree
        self.tree = phys_tree.__copy__(
            new_tree=EquilibriumTree(
                cache_path=self.cache_path,
                cache_name=self.cache_name + "_orig_",
                save_cache=self.save_cache,
                recompute_cache=self.recompute_cache,
            )
        )
        self.tree.treetype = 'original'
        # set the equilibrium potentials in the tree
        self.tree.setEEq()
        # get all channels in the tree
        self.channel_names = self.tree.getChannelsInTree()

        self.cfg = fit_cfg
        if fit_cfg is None:
            self.cfg = DefaultFitting()
        if concmech_cfg is None:
            self.concmech_cfg = DefaultMechParams()


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

        for node in self.ctree:
            loc_idx = node.loc_ind
            concmechs = self.tree[locs[loc_idx]['node']].concmechs

            # try to set default parameters as the ones from the original tree
            # if the concmech is not present at the corresponding location,
            # use the default parameters
            for ion in self.tree.ions:
                if ion in concmechs:
                    cparams = {
                        pname: pval for pname, pval in concmechs[ion].items()
                    }
                    node.addConcMech(ion, **cparams)

                else:
                    node.addConcMech(ion, **self.concmech_cfg.exp_conc_mech)

        # set the equilibirum potentials at fit locations
        eq = self.tree.calcEEq('fit locs')
        self.v_eqs_fit = eq[0]
        self.conc_eqs_fit = eq[1]

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
        sv_h = getExpansionPoints(self.cfg.e_hs, channel)

        # create the trees with only a single channel and multiple expansion points
        fit_tree = self.createTreeGF([channel_name],
            cache_name_suffix=f"_{channel_name}_",
        )

        # set the impedances in the tree
        fit_tree.setImpedancesInTree(
            freqs=self.cfg.freqs,
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
                channel_name, z_mats[:,ii,:,:], e_h, np.array([self.cfg.freqs]),
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

    def fitConcentration(self, ion, fit_tau=False, pprint=False):
        """
        Fits the concentration mechanisms parameters associate with the `ion`
        ion type.

        Parameters
        ----------
        ion: str
            The ion type that is to be fitted (e.g. 'ca').
        fit_tau: bool (default ``False``)
            If ``True``, fits the time-scale of the concentration mechansims. If
            ``False``, tries to take the time-scale from the corresponding
            location in the original tree. However, if no concentration
            mechanism is present at the corresponding location, than the default
            time-scale from `neat.factorydefaults` is taken.
        pprint: bool (default ``False``)
            Whether to print fit information.

        Returns
        -------
        bool
            `False` when no concentration mech for `ion` was found in the tree,
            `True` otherwise
        """
        # check if concmech for ion exists in original tree,
        # if not, skip the rest
        has_concmech = False
        for node in self.tree:
            if ion in node.concmechs:
                has_concmech = True
                break
        if not has_concmech:
            return 0

        for ion, conc_hs in self.cfg.conc_hs_cm.items():
            assert len(conc_hs) == len(self.cfg.e_hs_cm)
        nh = len(self.cfg.e_hs_cm)

        locs = self.tree.getLocs('fit locs')

        # get lists of linearisation holding potentials
        e_hs_aux_act, e_hs_aux_inact = getTwoVariableHoldingPotentials(self.cfg.e_hs_cm)
        conc_ehs = {ion: np.array(
            [c_ion for c_ion in self.cfg.conc_hs_cm[ion]] + \
            [c_ion for c_ion in self.cfg.conc_hs_cm[ion] for _ in range(nh)]
        )}

        # only retain channels involved with the ion
        channel_names = []
        sv_hs = {}
        for cname, chan in self.tree.channel_storage.items():

            if chan.ion == ion:
                channel_names.append(cname)

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

        # create the trees with the desired channels and expansion points
        fit_tree = self.createTreeGF(
            channel_names,
            cache_name_suffix=ion,
        )

        # set the impedances in the tree
        fit_tree.setImpedancesInTree(
            freqs=self.cfg.freqs, sv_h=sv_hs, pprint=pprint, use_conc=False,
        )
        # compute the impedance matrix for this activation level
        z_mats = fit_tree.calcImpedanceMatrix(locs)

        # set the impedances in the tree
        fit_tree.setImpedancesInTree(
            freqs=self.cfg.freqs, sv_h=sv_hs, pprint=pprint, use_conc=True,
        )
        # compute the impedance matrix for this activation level
        z_mats = fit_tree.calcImpedanceMatrix(locs)

        # fit the concentration mechanism
        self.ctree.computeConcMechGamma(
            z_mats, self.cfg.freqs, ion,
            sv_s=sv_hs, channel_names=channel_names,
        )

        if fit_tau:
            # add dimensions for broadcasting
            freqs = self.cfg.freqs_tau[None,:]
            for c_name in sv_hs:
                sv_hs[c_name] = {
                    key: val_arr[:,None] for key, val_arr in sv_hs[c_name].items()
                }

            # set the impedances in the tree
            fit_tree.setImpedancesInTree(
                freqs=freqs, sv_h=sv_hs, pprint=pprint, use_conc=True,
            )
            # compute the impedance matrix for this activation level
            z_mats = fit_tree.calcImpedanceMatrix(locs)

            self.ctree.computeConcMechTau(
                z_mats, freqs, ion,
                sv_s=sv_hs, channel_names=channel_names,
            )

        return 1

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
        locs = self.tree.getLocs('fit locs')

        suffix = "_pas_"
        if use_all_channels:
            suffix = f"_passified_"

        if use_all_channels:
            fit_tree = self.tree.__copy__(
                new_tree=EquilibriumTree(
                    cache_path=self.cache_path,
                    cache_name=self.cache_name + suffix,
                    save_cache=self.save_cache,
                    recompute_cache=self.recompute_cache,
                )
            )
            # set the channels to passive
            fit_tree.asPassiveMembrane()
            # convert to a greens tree for further evaluation
            fit_tree = fit_tree.__copy__(
                new_tree=FitTreeGF(
                    cache_path=self.cache_path,
                    cache_name=self.cache_name + suffix,
                    save_cache=self.save_cache,
                    recompute_cache=self.recompute_cache
                )
            )
            fit_tree.setCompTree()
        else:
            fit_tree = self.createTreeGF(
                [], # empty list of channel to include
                cache_name_suffix=suffix,
            )

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
        fit_tree.setImpedancesInTree(self.cfg.freqs, pprint=pprint)
        # compute the steady state impedance matrix
        z_mat = fit_tree.calcImpedanceMatrix(locs)[None,:,:]
        # fit the conductances to steady state impedance matrix
        self.ctree.computeGSingleChanFromImpedance('L', z_mat, -75., np.array([self.cfg.freqs]),
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

    def fitCapacitanceFromZ(self):
        """
        Fit the capacitance of the reduced model by a fit derived from the
        matrix exponential, i.e. by requiring that the derivatives of the
        impedance kernels equal the matrix product of the system matrix of
        the reduced model with the impedance kernel matrix of the full model
        """
        # create a `GreensTreeTime` to compute response kernels
        tree = self.tree.__copy__(
            new_tree=FitTreeC(
                cache_path=self.cache_path,
                cache_name=self.cache_name + "_Zkernels_",
                save_cache=self.save_cache,
                recompute_cache=self.recompute_cache,
            ),
        )
        tree.setCompTree(eps=1e-2)
        # set the impedances for kernel calculation
        tree.setImpedance(self.cfg.t_fit)
        # compute the response kernel matrices necessary for the fit
        zt_mat, dzt_dt_mat = tree.calcImpulseResponseMatrix(
            'fit locs',
            compute_time_derivative=True,
        )
        crt_mat = tree.calcChannelResponseMatrix(
            'fit locs',
            compute_time_derivative=False,
        )
        # perform the capacitance fit
        self.ctree.setEEq(self.v_eqs_fit)
        self.ctree.computeCfromZ(zt_mat, dzt_dt_mat, crt_mat)

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
            alphas, phimat, importance, _ = self._calcSOVMats(
                fit_locs, pprint=False
            )
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
                fit_locs, pprint=pprint
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
        return self._getKernels(pprint=pprint)

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

        if alphas is None or phimat is None:
            alphas, phimat, _, _ = self._calcSOVMats(
                fit_locs, pprint=False
            )

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

    def _storeSOVMats(self):
        fit_locs = self.tree.getLocs('fit locs')
        self.alphas, self.phimat, _, _ = self._calcSOVMats(
            fit_locs, pprint=False
        )

    def kernelObjective(self, t_arr=None):
        fit_locs = self.tree.getLocs('fit locs')
        nn = len(fit_locs)

        if t_arr is None:
            t_arr = np.concatenate(
                (np.logspace(-2,0,200), np.linspace(1., 200., 400)[1:])
            )

        k_orig, k_comp = self._getKernels(alphas=self.alphas, phimat=self.phimat)

        res = 0.
        for ii in range(nn):
            for jj in range(ii, nn):
                ko, kc = k_orig[ii][jj], k_comp[ii][jj]
                res += np.sum((ko(t_arr) - kc(t_arr))**2)

        return res

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
                            pprint=pprint, pplot=False)

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
        greens_tree.setImpedancesInTree(self.cfg.freqs, pprint=False)
        # create the NET
        net, z_mat = greens_tree.calcNETSteadyState(c_loc)
        net.improveInputImpedance(z_mat)

        # prune the NET to only retain ``locs``
        loc_inds = greens_tree.getNearestLocinds([c_loc]+locs, 'net eval')
        net_reduced = net.getReducedTree(loc_inds, indexing='locs')

        return net_reduced

    def fitEEq(self, **kwargs):
        """
        Fits the leak potentials of the reduced model to yield the same
        equilibrium potentials as the full model

        Parameters
        ----------
        ions: List[str]
            The ions that are included in the fit
        kwargs:
            arguments to the `CompartmentFitter.calcEEq()` function
        """
        fit_locs = self.tree.getLocs('fit locs')

        # set the equilibria
        self.ctree.setEEq(self.v_eqs_fit)
        for ion in self.tree.ions:
            self.ctree.setConcEq(ion, self.conc_eqs_fit[ion])

        # fit the leak
        self.ctree.fitEL()

        return self.ctree

    def fitModel(self,
        loc_arg,
        alpha_inds=[0], use_all_channels_for_passive=True,
        pprint=False, parallel=False,
    ):
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
        self.fitPassive(
            pprint=pprint,
            use_all_channels=use_all_channels_for_passive
        )
        # fit the capacitances
        self.fitCapacitance(inds=alpha_inds, pprint=pprint, pplot=False)
        # refit with only leak
        if use_all_channels_for_passive:
            self.fitPassiveLeak(pprint=pprint)

        # fit the ion channels
        self.fitChannels(pprint=pprint, parallel=parallel)

        # fit the concentration mechansims
        for ion in self.tree.ions:
            found = self.fitConcentration(ion, fit_tau=False, pprint=pprint)

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

        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.createTreeGF(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_",
        )
        greens_tree.setImpedancesInTree(self.cfg.freqs, pprint=False)
        # compute the impedance matrix of the synapse locations
        z_mat = greens_tree.calcImpedanceMatrix(locs, explicit_method=False)

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

        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.createTreeGF(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_",
        )
        greens_tree.setImpedancesInTree(self.cfg.freqs, pprint=False)
        # compute the impedance matrix of the synapse locations
        z_mat = greens_tree.calcImpedanceMatrix(cs_locs, explicit_method=False)
        zc_mat = z_mat[:n_comp, :n_comp]

        # get the reversal potentials of the synapse locations
        e_eqs = self.tree.calcEEq(cs_locs)[0]
        e_cs = e_eqs[:n_comp]
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

        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.createTreeGF(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_at_rest_",
        )
        greens_tree.setImpedancesInTree(self.cfg.freqs, pprint=False)

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

