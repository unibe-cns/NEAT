"""
File contains:

    - `neat.EquilibriumTree`

Authors: W. Wybo
"""
import dill
import numpy as np

import os
import pickle


from ..trees.morphtree import computational_tree_decorator
from ..trees.phystree import PhysTree
from ..trees.greenstree import GreensTree, GreensTreeTime
from ..trees.sovtree import SOVTree


try:
    from ..tools.simtools.neuron import neuronmodel as neurm
except ModuleNotFoundError:
    warnings.warn('NEURON not available, equilibrium evaluation not working', UserWarning)


class FitTree(PhysTree):
    def set_cache_params(self,
            recompute_cache=False,
            save_cache=True,
            cache_name='',
            cache_path='',
        ):
        self.cache_name = cache_name
        self.cache_path = cache_path
        self.save_cache = save_cache
        self.recompute_cache = recompute_cache

    def maybe_execute_funcs(self,
        funcs_args_kwargs=[],
        pprint=False,
    ):
        file_name = os.path.join(
            self.cache_path,
            f"{self.cache_name}cache_{self.unique_hash()}.p",
        )

        if pprint:
            print(f"\n>>>> Cache file for {self.__class__.__name__}:\n    {file_name}\n<<<<")

        try:
            # ensure that the funcs are recomputed if 'recompute' is true
            if self.recompute_cache:
                raise IOError

            with open(file_name, 'rb') as file:
                tree_ = dill.load(file)

            cache_params_dict = {
                "cache_name": self.cache_name,
                "cache_path": self.cache_path,
                "save_cache": self.save_cache,
                "recompute_cache": self.recompute_cache,
            }

            self.__dict__.update(tree_.__dict__)
            # set the original cache parameters
            self.__dict__.update(cache_params_dict)
            del tree_

        except (Exception, IOError, EOFError, KeyError) as err:
            if pprint:
                if self.recompute_cache:
                    logstr = '>>> Force recomputing cache...'
                else:
                    logstr = '>>> No cache found, recomputing...'
                print(logstr)

            # execute the functions
            for func, args, kwargs in funcs_args_kwargs:
                func(*args, **kwargs)

            if self.save_cache:
                with open(file_name, 'wb') as file:
                    dill.dump(self, file)


class EquilibriumTree(FitTree):
    """
    Subclass of `neat.PhysTree` that allows for the calculation of the
    equilibrium potential at each node.

    Uses the NEURON simulator to evaluate the equibrium potentials. Can cache
    the results of the computation.
    """
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

    def _calcEEq(self, locarg, ions=None, t_max=500., dt=0.1, factor_lambda=10.):
        """
        Calculates equilibrium potentials and concentrations in the tree.
        Computes the equilibria through a NEURON simulations without inputs.

        Parameters
        ----------
        locarg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            equilibrium state evaluated, if ``string``, specifies the
            name under which a set of locations is stored
        ions: `iterable` of `str
            the names of the ions for which the concentration needs to be measured
        t_max: float
            duration of the simulation
        dt: float
            time-step of the simulation
        factor_lambda: `float`
            multiplies the number of compartments suggested by the lambda-rule
        """
        locs = self._convertLocArgToLocs(locarg)
        if ions is None: ions = self.ions
        # use longer simulation for Eeq fit if concentration mechansims are present
        t_max = t_max*20. if len(ions) > 0 else t_max

        # create a biophysical simulation model
        sim_tree_biophys = self.__copy__(new_tree=neurm.NeuronSimTree())
        # compute equilibrium potentials
        sim_tree_biophys.initModel(dt=dt, factor_lambda=factor_lambda)
        sim_tree_biophys.storeLocs(locs, 'rec locs', warn=False)
        res_biophys = sim_tree_biophys.run(t_max, dt_rec=20., record_concentrations=ions)
        sim_tree_biophys.deleteModel()

        return (
            np.array([v_m[-1] for v_m in res_biophys['v_m']]),
            {ion: np.array([ion_eq[-1] for ion_eq in res_biophys[ion]]) for ion in ions}
        )

    def calcEEq(self, locarg, ions=None, method="interp", L_eps=50., pprint=False, **kwargs):
        """
        Calculates equilibrium potentials and concentrations in the tree.

        Uses either linear interpolations between the stored equilibria at the
        midpoints of the nodes or computes the equilibria through a NEURON
        simulation without inputs.

        Parameters
        ----------
        locarg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            equilibrium state evaluated, if ``string``, specifies the
            name under which a set of locations is stored
        ions: `iterable` of `str
            the names of the ions for which the concentration needs to be measured
        method: Literal: 'interp' or 'sim'
            whether to use interpolation or simulation. Defaults to simulation if
            distance is larger than `L_eps`
        L_eps: float
            maximum distance (um) above which the method defaults to interpolation
        pprint: bool
            Whether or not to print additional information
        """
        if ions is None: ions = self.ions
        locs = self._convertLocArgToLocs(locarg)
        ref_locs = [(n.index, .5) for n in self]
        self.storeLocs(ref_locs, name="ref locs")

        e_eqs = []
        conc_eqs = {ion: [] for ion in ions}

        if method == "interp":
            if pprint:
                print("> computing e_eq through interpolation")

            idxs0 = self.getNearestLocinds(locs, "ref locs", direction=1)
            idxs1 = self.getNearestLocinds(locs, "ref locs", direction=2)

            for loc, idx0, idx1 in zip(locs, idxs0, idxs1):

                if idx0 is None or idx1 is None:
                    if idx0 is None and idx1 is None:
                        # ref locs probably not defined, computations should be redone
                        break

                    idx = idx0 if idx0 is not None else idx1

                    # locs[idx0] more distal than leaf ref loc
                    e_eqs.append(self[ref_locs[idx][0]].v_ep)

                    for ion in ions:
                        conc_eqs[ion].append(self[ref_locs[idx][0]].conc_eps[ion])

                else:
                    L0 = self.pathLength(loc, ref_locs[idx0])
                    L1 = self.pathLength(loc, ref_locs[idx1])

                    if L0 < 1e-10 or L1 < 1e-10:
                        idx = idx0 if L0 < L1 else idx1
                        # both neighbour locations are the same
                        e_eqs.append(self[ref_locs[idx][0]].v_ep)

                        for ion in ions:
                            # linear interpolation to compute the equilibrium concentration
                            conc_eqs[ion].append(self[ref_locs[idx][0]].conc_eps[ion])

                    elif L0 < L_eps and L1 < L_eps:
                        v_ep0 = self[ref_locs[idx0][0]].v_ep
                        v_ep1 = self[ref_locs[idx1][0]].v_ep

                        # linear interpolation to compute the equilibrium potential
                        e_eqs.append((v_ep0 * L1 + v_ep1 * L0) / (L1 + L0))

                        for ion in ions:
                            c_ep0 = self[ref_locs[idx0][0]].conc_eps[ion]
                            c_ep1 = self[ref_locs[idx0][0]].conc_eps[ion]

                            # linear interpolation to compute the equilibrium concentration
                            conc_eqs[ion].append((c_ep0 * L1 + c_ep1 * L0) / (L1 + L0))

                    else:
                        break

        if len(e_eqs) < len(locs):
            if pprint:
                print("> computing e_eq through interpolation failed, simulating")
            return self._calcEEq(locarg, ions=ions, **kwargs)

        else:
            if pprint:
                print("> equilibria:")
                for ii, loc in enumerate(locs):
                    conc_eq_str = str({ion: f"{conc_eq[ii]:.8f}" for ion, conc_eq in conc_eqs.items()})
                    print(f"    loc {loc}: e_eq = {e_eqs[ii]:.2f} mV, {conc_eq_str}")
            return (
                np.array(e_eqs),
                {ion: np.array(conc_eq) for ion, conc_eq in conc_eqs.items()}
            )

    def _setEEq(self, ions=None, t_max=500., dt=0.1, factor_lambda=10.):
        if ions is None: ions = self.ions

        locs = [(n.index, .5) for n in self]
        res = self._calcEEq(locs, ions=ions, t_max=t_max, dt=dt, factor_lambda=factor_lambda)

        for ii, n in enumerate(self):
            n.setVEP(res[0][ii])
            for ion, conc in res[1].items():
                n.setConcEP(ion, conc[ii])

    def setEEq(self, ions=None, t_max=500., dt=0.1, factor_lambda=10., pprint=False):
        """
        Set equilibrium potentials and concentrations in the tree. Computes
        the equilibria through a NEURON simulation without inputs.

        Parameters
        ----------
        ions: `list` of `str
            the names of the ions for which the concentration needs to be measured
        t_max: float
            duration of the simulation
        dt: float
            time-step of the simulation
        factor_lambda: `float`
            multiplies the number of compartments suggested by the lambda-rule
        """
        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (
                    self._setEEq,
                    (),
                    dict(ions=ions, t_max=t_max, dt=dt, factor_lambda=factor_lambda)
                ),
            ]
        )


class FitTreeGF(GreensTree, FitTree):
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

        # we set freqs here already because it needs to be included in the
        # representation to generate a hash
        self.freqs = np.array(freqs)

        if sv_h is not None:
            # check if exansion point for all channels is defined
            assert sv_h.keys() == self.channel_storage.keys()

            for c_name, sv in sv_h.items():

                # set the expansion point
                for node in self:
                    node.setExpansionPoint(c_name, statevar=sv)

        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (self.setCompTree, [], {}),
                (self.setImpedance, [freqs], {"pprint": pprint, **kwargs})
            ]
        )

    @computational_tree_decorator
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


class FitTreeC(GreensTreeTime, FitTree):
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

    def setImpedancesInTree(self, t_arr, pprint=False):
        """
        Sets the impedances in the tree that are necessary for the evaluation
        of the response kernels.

        Parameters
        ----------
        t_arr: np.ndarray of float
            The time-points at which to evaluate the response kernels
        pprint: bool (optional, default is ``False``)
            Print info
        """
        if pprint:
            cname_string = ', '.join(list(self.channel_storage.keys()))
            print(f'>>> evaluating response kernels with {cname_string}')

        self._setFreqAndTimeArrays(t_arr)

        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (self.setCompTree, [], {}),
                (self.setImpedance, [t_arr], {})
            ]
        )


class FitTreeSOV(SOVTree, FitTree):
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

    def setSOVInTree(self, maxspace_freq=100., pprint=False):
        if pprint:
            print(f'>>> evaluating SOV expansion')

        self.maxspace_freq = maxspace_freq

        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (self.setCompTree, [], {"eps": 1.}),
                (self.calcSOVEquations, [], {
                    "maxspace_freq": maxspace_freq,"pprint": pprint,
                })
            ]
        )
