import os
import time
import copy
import warnings

import numpy as np

from ....trees.morphtree import MorphLoc
from ....trees.phystree import PhysTree, PhysNode

try:
    import neuron
    from neuron import h
except ModuleNotFoundError:
    warnings.warn('NEURON not available, importing non-functional h module only for doc generation', UserWarning)
    # universal iterable mock object
    class H(object):
        def __init__(self):
            pass

        def __getattr__(self,attr):
            try:
                return super(H, self).__getattr__(attr)
            except AttributeError:
                return self.__global_handler

        def __global_handler(self, *args, **kwargs):
            return H()

        def __iter__(self):  # make iterable
            return self

        def __next__(self):
            raise StopIteration

        def __mul__(self, other):  # make multipliable
            return 1.0

        def __rmul__(self, other):
            return self * other

        def __call__(self, *args, **kwargs): # make callable
            return H()
    h = H()
    neuron = H()
    np_array = np.array
    def array(*args, **kwargs):
        if isinstance(args[0], H):
            print(args)
            print(kwargs)
            return np.eye(2)
        else:
            return np_array(*args, **kwargs)
    np.array = array


h.load_file("stdlib.hoc") # contains the lambda rule
h.nrn_load_dll(os.path.join(os.path.dirname(__file__),
                            'x86_64/.libs/libnrnmech.so')) # load all mechanisms


class MechName(object):
    def __init__(self):
        self.names = {'L': 'pas', 'ca': 'CaDyn'}

    def __getitem__(self, key):
        if key in self.names:
            return self.names[key]
        else:
            return 'I' + key
mechname = MechName()


class NeuronSimNode(PhysNode):
    def __init__(self, index, p3d=None):
        super().__init__(index, p3d)

    def _makeSection(self, factorlambda=1., pprint=False):
        compartment = h.Section(name=str(self.index))
        compartment.push()
        # create the compartment
        if self.index == 1:
            compartment.diam = 2. * self.R # um (NEURON takes diam=2*r)
            compartment.L = 2. * self.R    # um (to get correct surface)
            compartment.nseg = 1
        else:
            compartment.diam = 2. * self.R  # section radius [um] (NEURON takes diam = 2*r)
            compartment.L = self.L # section length [um]
            # set number of segments
            if type(factorlambda) == float:
                # nseg according to NEURON bookint
                compartment.nseg = int(((compartment.L / (0.1 * h.lambda_f(100.)) + 0.9) / 2.) * 2. + 1.) * int(factorlambda)
            else:
                compartment.nseg = factorlambda

        # set parameters
        compartment.cm = self.c_m # uF/cm^2
        compartment.Ra = self.r_a*1e6 # MOhm*cm --> Ohm*cm
        # insert membrane currents
        for key, current in self.currents.items():
            if current[0] > 1e-10:
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.g = ' + str(current[0]) + '*1e-6') # uS/cm^2 --> S/cm^2
                    exec('seg.' + mechname[key] + '.e = ' + str(current[1])) # mV
        # insert concentration mechanisms
        for ion, params in self.concmechs.items():
            compartment.insert(mechname[ion])
            for seg in compartment:
                for param, value in params.items():
                    exec('seg.' + mechname[ion] + '.' + param + ' = ' + str(value))
        h.pop_section()

        if pprint:
            print(self)
            print(('>>> compartment length = %.2f um'%compartment.L))
            print(('>>> compartment diam = %.2f um'%compartment.diam))
            print(('>>> compartment nseg = ' + str(compartment.nseg)))

        return compartment

    def _makeShunt(self, compartment):
        if self.g_shunt > 1e-10:
            shunt = h.Shunt(compartment(1.))
            shunt.g = self.g_shunt # uS
            shunt.e = self.e_eq    # mV
            return shunt
        else:
            return None


class NeuronSimTree(PhysTree):
    """
    Tree class to define NEURON (Carnevale & Hines, 2004) based on `neat.PhysTree`.

    Attributes
    ----------
    sections: dict of hoc sections
        Storage for hoc sections. Keys are node indices.
    shunts: list of hoc mechanisms
        Storage container for shunts
    syns: list of hoc mechanisms
        Storage container for synapses
    iclamps: list of hoc mechanisms
        Storage container for current clamps
    vclamps: lis of hoc mechanisms
        Storage container for voltage clamps
    vecstims: list of hoc mechanisms
        Storage container for vecstim objects
    netcons: list of hoc mechanisms
        Storage container for netcon objects
    vecs: list of hoc vectors
        Storage container for hoc spike vectors
    dt: float
        timestep of the simulator ``[ms]``
    t_calibrate: float
        Time for the model to equilibrate``[ms]``. Not counted as part of the
        simulation.
    factor_lambda : int or float
        If int, the number of segments per section. If float, multiplies the
        number of segments given by the standard lambda rule (Carnevale, 2004)
        to give the number of compartments simulated (default value 1. gives
        the number given by the lambda rule)
    v_init: float
        The initial voltage at which the model is initialized ``[mV]``

    A `NeuronSimTree` can be extended easily with custom point process mechanisms.
    Just make sure that you store the point process in an existing appropriate
    storage container or in a custom storage container, since if all references
    to the hocobject disappear, the object itself will be deleted as well.

    .. code-block:: python
        class CustomSimTree(NeuronSimTree):
            def addCustomPointProcessMech(self, loc, **kwargs):
                loc = MorphLoc(loc, self)

                # create the point process
                pp = h.custom_point_process(self.sections[loc['node']](loc['x']))
                pp.arg1 = kwargs['arg1']
                pp.arg2 = kwargs['arg2']
                ...

                self.storage_container_for_point_process.append(pp)

    If you define a custom storage container, make sure that you overwrite the
    `__init__()` and `deleteModel()` functions to make sure it is created and
    deleted properly.
    """
    def __init__(self, file_n=None, types=[1,3,4],
                       factor_lambda=1., t_calibrate=0., dt=0.025, v_init=-75.):
        super().__init__(file_n=file_n, types=types)
        # neuron storage
        self.sections = {}
        self.shunts = []
        self.syns = []
        self.iclamps = []
        self.vclamps = []
        self.vecstims = []
        self.netcons = []
        self.vecs = []
        # simulation parameters
        self.dt = dt # ms
        self.t_calibrate = t_calibrate # ms
        self.factor_lambda = factor_lambda
        self.v_init = v_init # mV
        self.indstart = int(t_calibrate / dt)

    def _createCorrespondingNode(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return NeuronSimNode(node_index, p3d=p3d)

    def initModel(self, dt=0.025, t_calibrate=0., v_init=-75., factor_lambda=1.,
                        pprint=False):
        """
        Initialize hoc-objects to simulate the neuron model implemented by this
        tree.

        Parameters
        ----------
        dt: float (default is ``.025`` ms)
            Timestep of the simulation
        t_calibrate: float (default ``0.`` ms)
            The calibration time; time model runs without input to reach its
            equilibrium state before the true simulation starts
        v_init: float (default ``-75.`` mV)
            The initial voltage at which the model is initialized
        factor_lambda: float or int (default 1.)
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated (default value 1. gives
            the number given by the lambda rule)
        pprint: bool (default ``False``)
            Whether or not to print info on the NEURON model's creation
        """
        self.t_calibrate = t_calibrate
        self.dt = dt
        self.indstart = int(self.t_calibrate / self.dt)
        self.v_init = v_init
        self.factor_lambda = factor_lambda
        # reset all storage
        self.deleteModel()
        # create the NEURON model
        self._createNeuronTree(pprint=pprint)

    def deleteModel(self):
        '''
        Delete all stored hoc-objects
        '''
        # reset all storage
        self.sections = {}
        self.shunts = []
        self.syns = []
        self.iclamps = []
        self.vclamps = []
        self.vecstims = []
        self.netcons = []
        self.vecs = []
        self.storeLocs([{'node': 1, 'x': 0.}], 'rec locs')

    def _createNeuronTree(self, pprint):
        for node in self:
            # create the NEURON section
            compartment = node._makeSection(self.factor_lambda, pprint=pprint)
            # connect with parent section
            if not self.isRoot(node):
                compartment.connect(self.sections[node.parent_node.index], 1, 0)
            # store
            self.sections.update({node.index: compartment})
            # create a static shunt
            shunt = node._makeShunt(compartment)
            if shunt is not None:
                self.shunts.append(shunt)
        # if pprint:
        #     print(h.topology())


    def addShunt(self, loc, g, e_r):
        """
        Adds a static conductance at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the shunt.
        g: float
            The conductance of the shunt (uS)
        e_r: float
            The reversal potential of the shunt (mV)
        """
        loc = MorphLoc(loc, self)
        # create the shunt
        shunt = h.Shunt(self.sections[loc['node']](loc['x']))
        shunt.g = g
        shunt.e = e_r
        # store the shunt
        self.shunts.append(shunt)

    def addDoubleExpCurrent(self, loc, tau1, tau2):
        """
        Adds a double exponential input current at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the current waveform (ms)
        tau2: float
            Decay time of the current waveform (ms)
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.epsc_double_exp(self.sections[loc['node']](loc['x']))
        syn.tau1 = tau1
        syn.tau2 = tau2
        # store the synapse
        self.syns.append(syn)

    def addExpSynapse(self, loc, tau, e_r):
        """
        Adds a single-exponential conductance-based synapse

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        tau: float
            Decay time of the conductance window (ms)
        e_r: float
           Reversal potential of the synapse (mV)
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.exp_AMPA_NMDA(self.sections[loc['node']](loc['x']))
        syn.tau = tau
        syn.e = e_r
        # store the synapse
        self.syns.append(syn)

    def addDoubleExpSynapse(self, loc, tau1, tau2, e_r):
        """
        Adds a double-exponential conductance-based synapse

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the conductance window (ms)
        tau2: float
            Decay time of the conductance window (ms)
        e_r: float
            Reversal potential of the synapse (mV)
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.Exp2Syn(self.sections[loc['node']](loc['x']))
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.e = e_r
        # store the synapse
        self.syns.append(syn)

    def addNMDASynapse(self, loc, tau, tau_nmda, e_r=0., nmda_ratio=1.7):
        """
        Adds a single-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        tau: float
            Decay time of the AMPA conductance window (ms)
        tau_nmda: float
            Decay time of the NMDA conductance window (ms)
        e_r: float (optional, default ``0.`` mV)
           Reversal potential of the synapse (mV)
        nmda_ratio: float (optional, default 1.7)
            The ratio of the NMDA over AMPA component. Means that the maximum of
            the NMDA conductance window is ``nmda_ratio`` times the maximum of
            the AMPA conductance window.
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.exp_AMPA_NMDA(self.sections[loc['node']](loc['x']))
        syn.tau = tau
        syn.tau_NMDA = tau_nmda
        syn.e = e_r
        syn.NMDA_ratio = nmda_ratio
        # store the synapse
        self.syns.append(syn)

    def addDoubleExpNMDASynapse(self, loc, tau1, tau2, tau1_nmda, tau2_nmda,
                                     e_r=0., nmda_ratio=1.7):
        """
        Adds a double-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the AMPA conductance window (ms)
        tau2: float
            Decay time of the AMPA conductance window (ms)
        tau1_nmda: float
            Rise time of the NMDA conductance window (ms)
        tau2_nmda: float
            Decay time of the NMDA conductance window (ms)
        e_r: float (optional, default ``0.`` mV)
           Reversal potential of the synapse (mV)
        nmda_ratio: float (optional, default 1.7)
            The ratio of the NMDA over AMPA component. Means that the maximum of
            the NMDA conductance window is ``nmda_ratio`` times the maximum of
            the AMPA conductance window.
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.double_exp_AMPA_NMDA(self.sections[loc['node']](loc['x']))
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.tau1_NMDA = tau1_nmda
        syn.tau2_NMDA = tau2_nmda
        syn.e = e_r
        syn.NMDA_ratio = nmda_ratio
        # store the synapse
        self.syns.append(syn)

    def addIClamp(self, loc, amp, delay, dur):
        """
        Injects a DC current step at a given lcoation

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        amp: float
            The amplitude of the current (nA)
        delay: float
            The delay of the current step onset (ms)
        dur: float
            The duration of the current step (ms)
        """
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.IClamp(self.sections[loc['node']](loc['x']))
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.amp = amp # nA
        # store the iclamp
        self.iclamps.append(iclamp)

    def addSinClamp(self, loc, amp, delay, dur, bias, freq, phase):
        """
        Injects a sinusoidal current at a given lcoation

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        amp: float
            The amplitude of the current (nA)
        delay: float
            The delay of the current onset (ms)
        dur: float
            The duration of the current (ms)
        bias: float
            Constant baseline added to the sinusoidal waveform (nA)
        freq: float
            Frequency of the sinusoid (Hz)
        phase: float
            Phase of the sinusoid (rad)
        """
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.SinClamp(self.sections[loc['node']](loc['x']))
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.pkamp = amp # nA
        iclamp.bias = bias # nA
        iclamp.freq = freq # Hz
        iclamp.phase = phase # rad
        # store the iclamp
        self.iclamps.append(iclamp)

    def addOUClamp(self, loc, tau, mean, stdev, delay, dur, seed=None):
        """
        Injects a Ornstein-Uhlenbeck current at a given lcoation

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the current.
        tau: float
            Time-scale of the OU process (ms)
        mean: float
            Mean of the OU process (nA)
        stdev: float
            Standard deviation of the OU process (nA)
        delay: float
            The delay of current onset from the start of the simulation (ms)
        dur: float
            The duration of the current input (ms)
        seed: int, optional
            Seed for the random number generator
        """
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        if tau > 1e-9:
            iclamp = h.OUClamp(self.sections[loc['node']](loc['x']))
            iclamp.tau = tau
        else:
            iclamp = h.WNclamp(self.sections[loc['node']](loc['x']))
        iclamp.mean = mean # nA
        iclamp.stdev = stdev # nA
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.seed_usr = seed # ms
        iclamp.dt_usr = self.dt # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def addOUconductance(self, loc, tau, mean, stdev, e_r, delay, dur, seed=None):
        """
        Injects a Ornstein-Uhlenbeck conductance at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the conductance.
        tau: float
            Time-scale of the OU process (ms)
        mean: float
            Mean of the OU process (uS)
        stdev: float
            Standard deviation of the OU process (uS)
        e_r: float
            Reversal of the current (mV)
        delay: float
            The delay of current onset from the start of the simulation (ms)
        dur: float
            The duration of the current input (ms)
        seed: int, optional
            Seed for the random number generator
        """
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.OUConductance(self.sections[loc['node']](loc['x']))
        iclamp.tau = tau
        iclamp.mean = mean # uS
        iclamp.stdev = stdev # uS
        iclamp.e_r = e_r # mV
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.seed_usr = seed # ms
        iclamp.dt_usr = self.dt # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def addOUReversal(self, loc, tau, mean, stdev, g_val, delay, dur, seed=None):
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.OUReversal(self.sections[loc['node']](loc['x']))
        iclamp.tau = tau # ms
        iclamp.mean = mean # mV
        iclamp.stdev = stdev # mV
        iclamp.g = g_val # uS
        iclamp.delay = delay + self.t_calibrate # ms
        iclamp.dur = dur # ms
        iclamp.seed_usr = seed # ms
        iclamp.dt_usr = self.dt # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def addVClamp(self, loc, e_c, dur):
        """
        Adds a voltage clamp at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`neat.MorphLoc`
            The location of the conductance.
        e_c: float
            The clamping voltage (mV)
        dur: float, ms
            The duration of the voltage clamp
        """
        loc = MorphLoc(loc, self)
        # add the voltage clamp
        vclamp = h.SEClamp(self.sections[loc['node']](loc['x']))
        vclamp.rs = 0.01
        vclamp.dur1 = dur
        vclamp.amp1 = e_c
        # store the vclamp
        self.vclamps.append(vclamp)

    # def addRecLoc(self, loc):
    #     self.addLoc(loc, 'rec locs')

    def setSpikeTrain(self, syn_index, syn_weight, spike_times):
        """
        Each hoc point process that receive spikes through should by appended to
        the synapse stack (stored under the list `self.syns`).

        Default :class:`NeuronSimTree` point processes that are added to
        `self.syns` are:
        - `self.addDoubleExpCurrent()`
        - `self.addExpSyn()`
        - `self.addDoubleExpSyn()`
        - `self.addDoubleExpSyn()`
        - `self.addNMDASynapse()`
        - `self.addDoubleExpNMDASynapse()`

        With this function, these synapse can be set to receive a specific spike
        train.

        Parameters
        ----------
        syn_index: int
            index of the point process in the synapse stack
        syn_weight: float
            weight of the synapse (maximal value of the conductance window)
        spike_times: list or `np.array` of floats
            the spike times
        """
        # add spiketrain
        spks = np.array(spike_times) + self.t_calibrate
        spks_vec = h.Vector(spks.tolist())
        vecstim = h.VecStim()
        vecstim.play(spks_vec)
        netcon = h.NetCon(vecstim, self.syns[syn_index], 0, self.dt, syn_weight)
        # store the objects
        self.vecs.append(spks_vec)
        self.vecstims.append(vecstim)
        self.netcons.append(netcon)

    def run(self, t_max, downsample=1,
            record_from_syns=False, record_from_iclamps=False, record_from_vclamps=False,
            record_from_channels=False, record_v_deriv=False, record_concentrations=[],
            pprint=False):
        """
        Run the NEURON simulation. Records at all locations stored
        under the name 'rec locs' on `self` (see `MorphTree.storeLocs()`)

        Parameters
        ----------
        t_max: float
            Duration of the simulation
        downsample: int (> 0)
            Records the state of the model every `downsample` time-steps
        record_from_syns: bool (default ``False``)
            Record currents of synapstic point processes (in `self.syns`).
            Accessible as `np.ndarray` in the output dict under key 'i_syn'
        record_from_iclamps: bool (default ``False``)
            Record currents of iclamps (in `self.iclamps`)
            Accessible as `np.ndarray` in the output dict under key 'i_clamp'
        record_from_vclamps: bool (default ``False``)
            Record currents of vclamps (in `self.vclamps`)
            Accessible as `np.ndarray` in the output dict under key 'i_vclamp'
        record_from_channels: bool (default ``False``)
            Record channel state variables from `neat` defined channels in `self`,
            at locations stored under 'rec locs'
            Accessible as `np.ndarray` in the output dict under key 'chan'
        record_v_deriv: bool (default ``False``)
            Record voltage derivative at locations stored under 'rec locs'
            Accessible as `np.ndarray` in the output dict under key 'dv_dt'
        record_from_concentrations: bool (default ``False``)
            Record ion concentration at locations stored under 'rec locs'
            Accessible as `np.ndarray` in the output dict with as key the ion's
            name

        Returns
        -------
        dict
            Dictionary with the results of the simulation. Contains time and
            voltage as `np.ndarray` at locations stored under the name '
            rec locs', respectively with keys 't' and 'v_m'. Also contains
            traces of other recorded variables if the option to record them was
            set to ``True``
        """
        assert isinstance(downsample, int) and downsample > 0
        # simulation time recorder
        res = {'t': h.Vector()}
        res['t'].record(h._ref_t)
        # voltage recorders
        res['v_m'] = []
        for loc in self.getLocs('rec locs'):
            res['v_m'].append(h.Vector())
            res['v_m'][-1].record(self.sections[loc['node']](loc['x'])._ref_v)
        # synapse current recorders
        if record_from_syns:
            res['i_syn'] = []
            for syn in self.syns:
                res['i_syn'].append(h.Vector())
                res['i_syn'][-1].record(syn._ref_i)
        # current clamp current recorders
        if record_from_iclamps:
            res['i_clamp'] = []
            for iclamp in self.iclamps:
                res['i_clamp'].append(h.Vector())
                res['i_clamp'][-1].record(iclamp._ref_i)
        # voltage clamp current recorders
        if record_from_vclamps:
            res['i_vclamp'] = []
            for vclamp in self.vclamps:
                res['i_vclamp'].append(h.Vector())
                res['i_vclamp'][-1].record(vclamp._ref_i)
        # channel state variable recordings
        if record_from_channels:
            res['chan'] = {}
            channel_names = self.getChannelsInTree()
            for channel_name in channel_names:
                res['chan'][channel_name] = {str(var): [] for var in self.channel_storage[channel_name].statevars}
                for loc in self.getLocs('rec locs'):
                    for ind, varname in enumerate(self.channel_storage[channel_name].statevars):
                        var = str(varname)
                        # assure xcoordinate is refering to proper neuron section (not endpoint)
                        xx = loc['x']
                        if xx < 1e-3: xx += 1e-3
                        elif xx > 1. - 1e-3: xx -= 1e-3
                        # create the recorder
                        try:
                            rec = h.Vector()
                            exec('rec.record(self.sections[loc[0]](xx).' + mechname[channel_name] + '._ref_' + str(var) +')')
                            res['chan'][channel_name][var].append(rec)
                        except AttributeError:
                            # the channel does not exist here
                            res['chan'][channel_name][var].append([])
        if len(record_concentrations) > 0:
            for c_ion in record_concentrations:
                res[c_ion] = []
                for loc in self.getLocs('rec locs'):
                    res[c_ion].append(h.Vector())
                    exec('res[c_ion][-1].record(self.sections[loc[\'node\']](loc[\'x\'])._ref_' + c_ion + 'i)')
        # record voltage derivative
        if record_v_deriv:
            res['dv_dt'] = []
            for ii, loc in enumerate(self.getLocs('rec locs')):
                res['dv_dt'].append(h.Vector())
                # res['dv_dt'][-1].deriv(res['v_m'][ii], self.dt)

        # initialize
        # neuron.celsius=37.
        h.finitialize(self.v_init)
        h.dt = self.dt

        # simulate
        if pprint: print('>>> Simulating the NEURON model for ' + str(t_max) + ' ms. <<<')
        start = time.process_time()
        neuron.run(t_max + self.t_calibrate)
        stop = time.process_time()
        if pprint: print('>>> Elapsed time: ' + str(stop-start) + ' seconds. <<<')
        runtime = stop-start

        # compute derivative
        if 'dv_dt' in res:
            for ii, loc in enumerate(self.getLocs('rec locs')):
                res['dv_dt'][ii].deriv(res['v_m'][ii], h.dt, 2)
                res['dv_dt'][ii] = np.array(res['dv_dt'][ii])[self.indstart:][::downsample]
            res['dv_dt'] = np.array(res['dv_dt'])
        # cast recordings into numpy arrays
        res['t'] = np.array(res['t'])[self.indstart:][::downsample] - self.t_calibrate
        for key in set(res.keys()) - {'t', 'chan', 'dv_dt'}:
            if key in res and len(res[key]) > 0:
                res[key] = np.array([np.array(reslist)[self.indstart:][::downsample] \
                                     for reslist in res[key]])
                if key in ('i_syn', 'i_clamp', 'i_vclamp'):
                    res[key] *= -1.
        # cast channel recordings into numpy arrays
        if 'chan' in res:
            for channel_name in channel_names:
                channel = self.channel_storage[channel_name]
                for ind0, varname in enumerate(channel.statevars):
                    var = str(varname)
                    for ind1 in range(len(self.getLocs('rec locs'))):
                        res['chan'][channel_name][var][ind1] = \
                                np.array(res['chan'][channel_name][var][ind1])[self.indstart:][::downsample]
                        if len(res['chan'][channel_name][var][ind1]) == 0:
                            res['chan'][channel_name][var][ind1] = np.zeros_like(res['t'])
                    res['chan'][channel_name][var] = \
                            np.array(res['chan'][channel_name][var])
                # compute P_open
                # sv = np.zeros((len(channel.statevars), len(self.getLocs('rec locs')), len(res['t'])))
                sv = {}
                for varname in channel.statevars:
                    var = str(varname)
                    sv[var] = res['chan'][channel_name][var]
                res['chan'][channel_name]['p_open'] = channel.computePOpen(res['v_m'], **sv)

        return res

    def calcEEq(self, t_dur=100., set_e_eq=True):
        """
        Compute the equilibrium potentials in the middle (``x=0.5``) of each node.

        Parameters
        ----------
        t_dur: float (optional, default ``100.`` ms)
            The duration of the simulation
        set_e_eq: bool (optional, default ``True``)
            Store the equilibrium potential as the ``PhysNode.e_eq`` attribute
        """
        self.initModel(dt=self.dt, t_calibrate=self.t_calibrate,
                       v_init=self.v_init, factor_lambda=self.factor_lambda)
        self.storeLocs([(n.index, 0.5) for n in self], name='rec locs')
        res = self.run(t_dur)
        v_eq = res['v_m'][:-1]
        if set_e_eq:
            for (node, e) in zip(self, v_eq): node.setEEq(e_eq)


        return v_eq

    def calcImpedanceMatrix(self, locarg, i_amp=0.001, t_dur=100., pplot=False):
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
        elif isinstance(locarg, str):
            locs = self.getLocs(locarg)
        else:
            raise IOError('`locarg` should be list of locs or string')
        z_mat = np.zeros((len(locs), len(locs)))
        for ii, loc0 in enumerate(locs):
            for jj, loc1 in enumerate(locs):
                self.initModel(dt=self.dt, t_calibrate=self.t_calibrate,
                               v_init=self.v_init, factor_lambda=self.factor_lambda)
                self.addIClamp(loc0, i_amp, 0., t_dur)
                self.storeLocs([loc0, loc1], 'rec locs', warn=False)
                # simulate
                res = self.run(t_dur)
                # voltage deflections
                # v_trans = res['v_m'][1][-int(1./self.dt)] - self[loc1['node']].e_eq
                v_trans = res['v_m'][1][-int(1./self.dt)] - res['v_m'][1][0]
                # compute impedances
                z_mat[ii, jj] = v_trans / i_amp
                if pplot:
                    import matplotlib.pyplot as pl
                    pl.figure()
                    pl.plot(res['t'], res['v_m'][1])
                    pl.show()

        return z_mat

    def calcImpedanceKernelMatrix(self, locarg, i_amp=0.001,
                                                dt_pulse=0.1, t_max=100.):
        tk = np.arange(0., t_max, self.dt)
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
        elif isinstance(locarg, str):
            locs = self.getLocs(locarg)
        else:
            raise IOError('`locarg` should be list of locs or string')
        zk_mat = np.zeros((len(tk), len(locs), len(locs)))
        for ii, loc0 in enumerate(locs):
            for jj, loc1 in enumerate(locs):
                loc1 = locs[jj]
                self.initModel(dt=self.dt, t_calibrate=self.t_calibrate,
                               v_init=self.v_init, factor_lambda=self.factor_lambda)
                self.addIClamp(loc0, i_amp, 0., dt_pulse)
                self.storeLocs([loc0, loc1], 'rec locs', warn=False)
                # simulate
                res = self.run(t_max)
                # voltage deflections
                v_trans = res['v_m'][1][1:] - self[loc1['node']].e_eq
                # compute impedances
                zk_mat[:, ii, jj] = v_trans / (i_amp * dt_pulse)
        return tk, zk_mat


class NeuronCompartmentNode(NeuronSimNode):
    def __init__(self, index):
        super().__init__(index)

    def getChildNodes(self, skip_inds=[]):
        return super().getChildNodes(skip_inds=skip_inds)

    def _makeSection(self, pprint=False):
        compartment = neuron.h.Section(name=str(self.index))
        compartment.push()
        # create the compartment
        if 'points_3d' in self.content:
            points = self.content['points_3d']
            h.pt3dadd(*points[0], sec=compartment)
            h.pt3dadd(*points[1], sec=compartment)
            h.pt3dadd(*points[2], sec=compartment)
            h.pt3dadd(*points[3], sec=compartment)
        else:
            compartment.diam = 2. * self.R  # section radius [um] (NEURON takes diam = 2*r)
            compartment.L = self.L # section length [um]
        # set number of segments to one
        compartment.nseg = 1

        # set parameters
        compartment.cm = self.c_m # uF/cm^2
        compartment.Ra = self.r_a*1e6 # MOhm*cm --> Ohm*cm
        # insert membrane currents
        for key, current in self.currents.items():
            if current[0] > 1e-10:
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec('seg.' + mechname[key] + '.g = ' + str(current[0]) + '*1e-6') # uS/cm^2 --> S/cm^2
                    exec('seg.' + mechname[key] + '.e = ' + str(current[1])) # mV
        # insert concentration mechanisms
        for ion, params in self.concmechs.items():
            compartment.insert(mechname[ion])
            for seg in compartment:
                for param, value in params.items():
                    exec('seg.' + mechname[ion] + '.' + param + ' = ' + str(value))
        h.pop_section()

        if pprint:
            print(self)
            print(('>>> compartment length = %.2f um'%compartment.L))
            print(('>>> compartment diam = %.2f um'%compartment.diam))
            print(('>>> compartment nseg = ' + str(compartment.nseg)))

        return compartment


class NeuronCompartmentTree(NeuronSimTree):
    """
    Subclass of `NeuronSimTree` where sections are defined so that they are
    effectively single compartments. Should be created from a
    `neat.CompartmentTree` using `neat.createReducedCompartmentModel()`
    """
    def __init__(self, t_calibrate=0., dt=0.025, v_init=-75.):
        super().__init__(file_n=None, types=[1,3,4],
                         t_calibrate=t_calibrate, dt=dt, v_init=v_init)

    # redefinition of bunch of standard functions to not include skip inds by default
    def __getitem__(self, index, skip_inds=[]):
        return super().__getitem__(index, skip_inds=skip_inds)

    def getNodes(self, recompute_flag=0, skip_inds=[]):
        return super().getNodes(recompute_flag=recompute_flag, skip_inds=skip_inds)

    def __iter__(self, node=None, skip_inds=[]):
        return super().__iter__(node=node, skip_inds=skip_inds)

    def _findNode(self, node, index, skip_inds=[]):
        return super()._findNode(node, index, skip_inds=skip_inds)

    def _gatherNodes(self, node, node_list=[], skip_inds=[]):
        return super()._gatherNodes(node, node_list=node_list, skip_inds=skip_inds)

    def _createCorrespondingNode(self, node_index):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return NeuronCompartmentNode(node_index)

    def _createNeuronTree(self, pprint):
        for node in self:
            # create the NEURON section
            compartment = node._makeSection(pprint=pprint)
            # connect with parent section
            if not self.isRoot(node):
                compartment.connect(self.sections[node.parent_node.index], 0.5, 0)
            # store
            self.sections.update({node.index: compartment})
            # create a static shunt
            shunt = node._makeShunt(compartment)
            if shunt is not None:
                self.shunts.append(shunt)


def createReducedNeuronModel(ctree, fake_c_m=1., fake_r_a=100.*1e-6, method=2):
    """
    Creates a `neat.NeuronCompartmentTree` to simulate reduced compartmentment
    models from a `neat.CompartmentTree`.

    Parameters
    ----------
    ctree: `neat.CompartmentTree`
        The tree containing the parameters of the reduced compartmental model
        to be simulated

    Returns
    -------
    `neat.NeuronCompartmentTree`

    Notes
    -----
    The function `ctree.getEquivalentLocs()` can be used to obtain 'fake'
    locations corresponding to each compartment, which in turn can be used to
    insert hoc point process at the compartments using the same functions
    definitions as for as for a morphological `neat.NeuronSimTree`
    """
    # calculate geometry that will lead to correct constants
    arg1, arg2 = ctree.computeFakeGeometry(fake_c_m=fake_c_m, fake_r_a=fake_r_a,
                                                 factor_r_a=1e-6, delta=1e-10,
                                                 method=method)
    if method == 1:
        points = arg1; surfaces = arg2
        sim_tree = ctree.__copy__(new_tree=NeuronCompartmentTree())
        for ii, comp_node in enumerate(ctree):
            pts = points[ii]
            sim_node = sim_tree.__getitem__(comp_node.index, skip_inds=[])
            sim_node.setP3D(np.array(pts[0][:3]), (pts[0][3] + pts[-1][3]) / 2., 3)

        # fill the tree with the currents
        for ii, sim_node in enumerate(sim_tree):
            comp_node = ctree[ii]
            sim_node.currents = {chan: [g / surfaces[comp_node.index], e] \
                                         for chan, (g, e) in comp_node.currents.items()}
            sim_node.concmechs = copy.deepcopy(comp_node.concmechs)
            for concmech in sim_node.concmechs.values():
                concmech.gamma *= surfaces[comp_node.index]
            sim_node.c_m = fake_c_m
            sim_node.r_a = fake_r_a
            sim_node.content['points_3d'] = points[comp_node.index]
    elif method == 2:
        lengths = arg1 ; radii = arg2
        surfaces = 2. * np.pi * radii * lengths
        sim_tree = ctree.__copy__(new_tree=NeuronCompartmentTree())
        for ii, comp_node in enumerate(ctree):
            sim_node = sim_tree.__getitem__(comp_node.index, skip_inds=[])
            if sim_tree.isRoot(sim_node):
                sim_node.setP3D(np.array([0.,0.,0.]), radii[ii]*1e4, 1)
            else:
                sim_node.setP3D(np.array([sim_node.parent_node.xyz[0]+lengths[ii]*1e4, 0., 0.]),
                                 radii[ii]*1e4, 3)

        # fill the tree with the currents
        for ii, sim_node in enumerate(sim_tree):
            comp_node = ctree[ii]
            sim_node.currents = {chan: [g / surfaces[comp_node.index], e] \
                                         for chan, (g, e) in comp_node.currents.items()}
            sim_node.concmechs = copy.deepcopy(comp_node.concmechs)
            for concmech in sim_node.concmechs.values():
                concmech.gamma *= surfaces[comp_node.index]
            sim_node.c_m = fake_c_m
            sim_node.r_a = fake_r_a
            sim_node.R = radii[comp_node.index]*1e4    # convert to [um]
            sim_node.L = lengths[comp_node.index]*1e4  # convert to [um]
    return sim_tree

