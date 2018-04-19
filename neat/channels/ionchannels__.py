import numpy as np
import sympy as sp
import scipy.optimize
import scipy.sparse as ss
import math
import copy

def vtrap(x, y):
    if np.abs(x/y) < 1e-6:
        trap = y*(1. - x/(y*2.))
    else:
        trap = x/(np.exp(x/y) - 1.)
    return trap


## generic ion channel class ###########################################
class IonChannel(object):
    '''
    Base class for all different ion channel types. 

    The algebraic form of the membrance current is stored in three sympy arrays:
    `statevar`, `powers` and `factors`. An example of how the current is 
    computed is given below:
        `statevar` = ``[[a11,a12],[a21,a22]]``
        `powers` = ``[[n11,n12],[n21,n22]]``
        `factors` = ``[f1, f2]``
    Then the corresponding transmembrane current is
        math::I = g (f1 a11^{n11} a12^{n12} + f2 a21^{n21} a22^{n22}) (V - e)
    
    Parameters
    ----------
        g_max : float
            the maximal conductance (uS/cm^2)
        e_rev : float
            the reversal (mV)
        e_eq : float
            the equilibrium potential of the neuron (mV)
        calc : boolean
            specifies if the ion channels is to be use for impedance calculations
    
    Attributes
    ----------
        statevar : 2d numpy.ndarray of floats
            values of the state variables
        powers : 2d numpy.ndarray of floats
            powers of the corresponding state variables
        factors : numpy.array 
            factors to multiply terms in sum
    
        calc : bool 
            whether the ion channels serve to compute kernels
        V0 : float,
            resting potential of the neuron
    
        When `calc` is ``True``, we need to set some sympy variables to perform 
        the computations:
        spV : sympy.symbol 
            voltage
        statevars : sympy.ndarray of sympy.symbol
            state variables
        fun : sympy expression 
            function of statevariables in the membrane current
        fstatevar : sympy expression 
            statevariable functions
        varinf : sympy.ndarray of sympy.expression
            the asymptotic values for the sympy variables
        tau : sympy.ndarray of sympy.expressions 
            the timescales of the state variables
        fv : sympy.ndarray of sympy.expressions 
            the coefficients of the expansion of the current. Compute them by using 
            :func:`set_expansion_coeffs`.
    '''

    def __init__(self, g_max=0., e_rev=0., e_eq=0., calc=False):
        self.g = g_max
        self.e = e_rev
        self.V0 = e_eq
        self.calc = calc


    def get_current(self, V):
        '''
        Returns the current of the ionchannel in nA

        Parameters
        ----------
            V : float
                the membrane voltage

        Returns
        -------
            float
        '''
        return - self.g \
               * np.sum(self.factors \
                        * np.prod(self.statevar**self.powers, 1)[:,None]) \
               * (V - self.e)

    def get_conductance(self):
        '''
        Let the channel current be :math:`-g(V-e)`. Returns :math:`-g` and 
        :math:`-g*(E_{eq}-e)`, with :math:`E_{eq}` the equilibrium potential

        Returns
        -------
            (float, float)
                :math:`-g` and :math:`-g*(E_eq-e)`
        '''
        geff = self.g \
               * np.sum(self.factors \
                        * np.prod(self.statevar**self.powers, 1)[:,None])
        return - geff, geff * (self.e - self.V0)
        
    def fun_statevar(self, V): 
        '''
        Place holder, should be overloaded in the derived classes
        '''
        return 0.
        
    def set_expansion_point(self, E_eq=-65.):
        self.E_eq = E_eq

    def calc_passive(self, freqs):
        return - self.g0 * np.ones(freqs.shape, dtype=complex)

    def calc_linear(self, freqs):
        '''
        Computes contribution of ion channels to membrane impedance
        '''
        coeffI, coeffstatevar = self.compute_lincoeff()
        # convert units of coeffstatevar to 1/s (instead of 1/ms)
        for ind, var in np.ndenumerate(self.statevars):
            coeffstatevar[0][ind] *= 1e3
            coeffstatevar[1][ind] *= 1e3
        returndict = {}
        imp = coeffI[0] * np.ones(freqs.shape, dtype=complex)
        for ind, var in np.ndenumerate(self.statevars):
            # response function for state variable given the voltage
            returndict[var] = coeffstatevar[0][ind] / (freqs - coeffstatevar[1][ind])
            # response function contribution of state variable to membrane impedance
            imp += coeffI[1][ind] * (self.V0 - self.e) * returndict[var]
        returndict['V'] = self.g * imp
        return returndict
        
    def compute_lincoeff(self):
        '''
        computes coefficients for linear simulation
        '''
        # coefficients for computing current
        fun = self.fun #statevars**self.powers
        coeff = np.zeros(self.statevar.shape, dtype=object)
        # differentiate
        for ind, var in np.ndenumerate(self.statevars):
            coeff[ind] = sp.diff(fun, var,1)
        # substitute
        for ind, var in np.ndenumerate(self.statevars):
            fun = fun.subs(var, self.varinf[ind])
            for ind2, coe in np.ndenumerate(coeff):
                coeff[ind2] = coe.subs(var, self.varinf[ind])
        fun = fun.subs(self.spV, self.V0)
        for ind, coe in np.ndenumerate(coeff):
            coeff[ind] = coe.subs(self.spV, self.V0)
        coeffI = [np.float64(fun), coeff.astype(float)]
        
        # coefficients for state variable equations
        dfdv = np.zeros(self.statevar.shape, dtype=object)
        dfdx = np.zeros(self.statevar.shape, dtype=object)
        # differentiate
        for ind, var in np.ndenumerate(self.statevars):
            dfdv[ind] = sp.diff(self.fstatevar[ind], self.spV, 1)
            dfdx[ind] = sp.diff(self.fstatevar[ind], var, 1)
        # substitute state variables by their functions
        for ind, var in np.ndenumerate(self.statevars):
            dfdv[ind] = dfdv[ind].subs(var, self.varinf[ind])
        # substitute voltage by its value
        for ind, var in np.ndenumerate(self.statevars):
            dfdv[ind] = dfdv[ind].subs(self.spV, self.V0)
            dfdx[ind] = dfdx[ind].subs(self.spV, self.V0)

        coeffstatevar = [dfdv.astype(float), dfdx.astype(float)]
        
        return coeffI, coeffstatevar
       
    def write_mod_file(self):
        '''
        Writes a modfile of the ion channel for simulations with neuron
        '''
        f = open('../mech/I' + self.__class__.__name__ + '.mod', 'w')
        
        f.write(': This mod file is automaticaly generated by the ionc.write_mode_file() function in /source/ionchannels.py \n\n')
        
        f.write('NEURON {\n')
        f.write('    SUFFIX I' + self.__class__.__name__ + '\n')
        if self.ion == '':
            f.write('    NONSPECIFIC_CURRENT i' + '\n')
        else:
            # f.write('    USEION ' + self.ion + ' READ e' + self.ion + ' WRITE i' + self.ion + '\n')
            f.write('    USEION ' + self.ion + ' WRITE i' + self.ion + '\n')
        if len(self.concentrations) > 0:
            for concstring in self.concentrations:
                f.write('    USEION ' + concstring + ' READ ' + concstring + 'i' + '\n')
        f.write('    RANGE  g, e' + '\n')
        varstring = 'var0inf'
        taustring = 'tau0'
        for ind in range(len(self.varinf.flatten()[1:])):
            varstring += ', var' + str(ind+1) + 'inf'
            taustring += ', tau' + str(ind+1)
        f.write('    GLOBAL ' + varstring + ', ' + taustring + '\n')
        f.write('    THREADSAFE' + '\n')
        f.write('}\n\n')
        
        f.write('PARAMETER {\n')
        f.write('    g = ' + str(self.g*1e-6) + ' (S/cm2)' + '\n')
        f.write('    e = ' + str(self.e) + ' (mV)' + '\n')
        for ion in self.concentrations:
            f.write('    ' + ion + 'i (mM)' + '\n')
        f.write('}\n\n')
        
        f.write('UNITS {\n')
        f.write('    (mA) = (milliamp)' + '\n')
        f.write('    (mV) = (millivolt)' + '\n')
        f.write('    (mM) = (milli/liter)' + '\n')
        f.write('}\n\n')
        
        f.write('ASSIGNED {\n')
        f.write('    i' + self.ion + ' (mA/cm2)' + '\n')
        # if self.ion != '':
        #     f.write('    e' + self.ion + ' (mV)' + '\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + 'inf' + '\n')
            f.write('    tau' + str(ind) + ' (ms)' + '\n')
        f.write('    v (mV)' + '\n')
        f.write('}\n\n')
        
        f.write('STATE {\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + '\n')
        f.write('}\n\n')
        
        f.write('BREAKPOINT {\n')
        f.write('    SOLVE states METHOD cnexp' + '\n')
        calcstring = '    i' + self.ion + ' = g * ('
        l = 0
        for i in range(self.statevar.shape[0]):
            for j in range(self.statevar.shape[1]):
                for k in range(self.powers[i,j]):
                    calcstring += ' var' + str(l) + ' *'
                l += 1
            calcstring += str(self.factors[i,0])
            if i < self.statevar.shape[0] - 1:
                calcstring += ' + '
        # calcstring += ') * (v - e' + self.ion + ')'
        calcstring += ') * (v - e)'
        f.write(calcstring + '\n')
        f.write('}\n\n')
        
        concstring = ''
        for ion in self.concentrations:
            concstring += ', ' + ion + 'i'
        f.write('INITIAL {\n')
        f.write('    rates(v' + concstring + ')' + '\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + ' = var' + str(ind) + 'inf' + '\n')
        f.write('}\n\n')
        
        f.write('DERIVATIVE states {\n')
        f.write('    rates(v' + concstring + ')' + '\n')
        for ind in range(len(self.varinf.flatten())):
            f.write('    var' + str(ind) + '\' = (var' + str(ind) + 'inf - var' + str(ind) + ') / tau' + str(ind) + '\n')
        f.write('}\n\n')
        
        concstring = ''
        for ion in self.concentrations:
            concstring += ', ' + ion
        f.write('PROCEDURE rates(v' + concstring + ') {\n')
        for ind, varinf in enumerate(self.varinf.flatten()):
            f.write('    var' + str(ind) + 'inf = ' + sp.printing.ccode(varinf) + '\n')
            f.write('    tau' + str(ind) + ' = ' + sp.printing.ccode(self.tau.flatten()[ind]) + '\n')
        f.write('}\n\n')
        
        f.close()

    def create_py_simchannel(self, inloc_inds, Ninloc, Es_eq, g, e, flag=0, conc_eq=[], mode_ratio=.1):
        '''
        Creates a vectorized simulation object. It is assumed that ther are 'Ninloc' locations at which
        the voltage is modeled, and the channel has a nonzero conductance at the locations indicated by 
        the indices in 'inloc_inds'. 

        Input:
            [inloc_inds]: array of ints, indices where the channel has nonzero conductance
            [Ninloc]: int, the number of inlocs
            [Es_eq]: float or array of floats, in the former case a uniform equilibrium potential or 
                and in the latter case a vector of equilibrium potentials at all locations
            [g]: array of floats, maximal conductances at the locations indicated by [inloc_inds]
            [e]: array of floats, reversal potentials at the locations indicated by [inloc_inds]
            [flag]: int, mode of integration. 0 simulates the full current, 1 the non-passive current
                and 2 the non-linear current
            [conc_eq]: 2d array of floats (but only if the channel depends on concentrations), first 
                dimension has the length of [inloc_inds] and second dimension is the same as the number 
                of concentrations. Flag 2 is incompatible with concentrations dependence.
            [mode_ratio]: float between zero and one, a numpy technicality

        Output:
            [simchans]: list of vectorized simchannel objects required to simulate the channel.
        '''

        if flag == 2 and len(self.concentrations) > 0:
            raise Exception('No nonlinear simulations allowed with ion channels that depend on concentrations!')

        fraction = float(len(inloc_inds)) / float(Ninloc)
        if fraction > mode_ratio:
            mode = 0
        else:
            mode = 1

        simchans = []
        for i, varrow in enumerate(self.varinf):

            svarr = np.zeros((1, len(varrow)), dtype=object)
            tauarr = np.zeros((1, len(varrow)), dtype=object)
            if flag == 2:
                dsvarr = np.zeros((1, len(varrow)), dtype=object)
            
            for ind, expr in enumerate(varrow):
                if len(self.concentrations) > 0.:
                    svarr[0,ind] = sp.utilities.lambdify([self.spV]+[c for c in self.conc], expr, "numpy")
                else:
                    # check if svinf depends on V, if not, our own lambdify has to be implemented
                    if self.spV in expr.atoms():
                        svarr[0,ind] = sp.utilities.lambdify(self.spV, expr, "numpy")
                    else:
                        def fun(v, val=float(expr)):
                            return float(val) * np.ones(v.shape)
                        svarr[0,ind] = fun
                # check if tau depends on V, if not, our own lambdify has to be implemented
                if self.spV in self.tau[i,ind].atoms():
                    tauarr[0,ind] = sp.utilities.lambdify(self.spV, self.tau[i,ind], "numpy")
                else:
                    # tauarr[0,ind] = lambda v: float(self.tau[i,ind]) * np.ones(v.shape)
                    def fun(v, tauexpr=float(self.tau[i,ind])):
                        return tauexpr * np.ones(v.shape)
                    tauarr[0,ind] = fun
                if flag == 2:
                    dsvarr[0,ind] = sp.utilities.lambdify(self.spV, sp.diff(expr, self.spV, 1), "numpy")

            if len(self.concentrations) > 0.:
                def svfun(V, conc, sva=svarr):
                    args = [V] + [conc[:,i] for i in range(len(conc[0,:]))]
                    return np.array([svf(*args) for svf in sva[0]]).T
            else:
                def svfun(V, sva=svarr):
                    return np.array([svf(V) for svf in sva[0]]).T
            def taufun(V, taua=tauarr):
                return np.array([tauf(V) for tauf in taua[0]]).T
            if flag == 2:
                def dsvfun(V, dsva=dsvarr):
                    return np.array([dsvf(V) for dsvf in dsva[0]]).T
            else:
                dsvfun = None

            simchans.append( simchannel(inloc_inds, Ninloc, Es_eq, conc_eq,
                                self.ion, self.concentrations,
                                g*self.factors[i,0], e,
                                self.powers[i:i+1,:],
                                svfun, taufun, dsvfun,
                                flag, mode=mode) )

        return simchans


class simchannel:
    def __init__(self, inloc_inds, Ninloc, Es_eq, conc_eq,
                        ion, concentrations,
                        g, e, 
                        powers,
                        svinf, tauinf, dsvinf=None,
                        flag=0, mode=1):
        '''
        Creates a vectorized simulation object and accepts a vector of of voltages.

        Let N be the number of state variables.

        Parameters
            inloc_inds : numpy.array of ints
                see :func:`ionChannel.create_py_simchannel`
            Ninloc : int
                see :func:`ionChannel.create_py_simchannel`
            Es_eq : float or numpy.array of floats
                see :func:`ionChannel.create_py_simchannel`
            g : numpy.array of floats
                see :func:`ionChannel.create_py_simchannel`
            e : numpy.array of floats
                see :func:`ionChannel.create_py_simchannel`
            powers : numpy.array of ints (size=N)
                The powers of the state variables
            svinf : function, 
                accepts a vector V and returns the rate functions in a shape
                len(V) by N 
            tauinf : function, accepts a vector V and returns state variable 
                time-scales in a shape len(V) by N 
            dsvinf : None or function (only if flag is 2), derivatives of the rate functions,
                for non-linear simulation
            flag : int, mode of integration. 0 simulates the full current, 1 the non-passive current
                and 2 the non-linear current
            mode : int, if 0, simulates the channel at all locations. If 1, only simulates at the 
                locations indicated in [Ninloc_inds]
        '''
        self.flag = flag
        self.ion = ion
        self.concentrations = concentrations
        # integration mode
        self.mode = mode
        # inloc info
        self.Ninloc = Ninloc
        self.inloc_inds = inloc_inds
        if mode == 1:
            self.Nelem = len(inloc_inds)
            self.elem_inds = copy.copy(self.inloc_inds)
        else:
            self.Nelem = Ninloc
            self.elem_inds = np.arange(self.Ninloc)
        # equilibirum potentials
        if type(Es_eq) == float:
            self.Es_eq = Es_eq * np.ones(self.Nelem)
        else:
            if mode == 1:
                self.Es_eq = Es_eq[self.inloc_inds]
            else:
                self.Es_eq = Es_eq
        # equilibrium concentrations
        self.conc_eq = conc_eq
        # state variables
        if len(self.concentrations) > 0:
            self.sv = svinf(self.Es_eq[self.elem_inds], self.conc_eq[self.elem_inds,:])
        else:
            self.sv = svinf(self.Es_eq[self.elem_inds])
        if flag == 2:
            self.svlin = np.zeros((2, self.sv.shape[0], self.sv.shape[1]))
        # powers
        self.powers = powers
        # maximal conductance and reversal
        if mode == 1:
            self.g = g
            self.e = e
        else:
            self.g = np.zeros(Ninloc)
            self.e = np.zeros(Ninloc)
            self.g[inloc_inds] = g
            self.e[inloc_inds] = e
        # rate functions
        self.svinf = svinf
        self.tauinf = tauinf
        if flag == 2:
            self.dsvinf = dsvinf
        # set state variable values
        self.sveq = copy.deepcopy(self.sv)
        if flag == 2:
            self.dsveq = dsvinf(self.Es_eq[self.elem_inds])
        # equilibirum time scales
        self.taueq = tauinf(self.Es_eq[self.elem_inds])
        # equilibirum open probability
        self.popeneq = (self.sveq**self.powers).prod(1)
        # auxiliary arrays
        self.svinf_aux = copy.copy(self.sveq)
        self.tauinf_aux = copy.copy(self.taueq)
        self.popen_aux = copy.copy(self.popeneq)
        self.exp_aux = np.exp(-.1/self.tauinf_aux)

    def reset(self):
        self.sv = self.sveq
        if self.flag == 2:
            self.svlin = np.zeros(self.svlin.shape)

    def advance_general(self, dt, V, conc=[]):
        '''
        Advance the ion channels internal variables one timestep

        Input:
            [dt]: float, the timestep
            [V]: array, voltage (len(V)==Ninloc)
            [conc]: concentration vector (size=(len(V),number of concentrations))
        '''
        if self.flag == 2:
            self.advance_nl(dt, V, conc=conc)
        else:
            self.advance(dt, V, conc=conc)

    def advance(self, dt, V, conc=[]):
        # recast V and conc if necessary
        if self.mode == 1:
            V = V[self.inloc_inds]
            if len(self.concentrations) > 0:
                conc = conc[self.inloc_inds,:]
        # compute rate functions
        if len(self.concentrations) > 0:
            self.svinf_aux = self.svinf(V, conc)
        else:
            self.svinf_aux = self.svinf(V)
        self.tauinf_aux = self.tauinf(V)
        self.exp_aux = np.exp(-dt/self.tauinf_aux)
        # advance the variables     
        self.sv *= self.exp_aux
        self.sv += (1.-self.exp_aux) * self.svinf_aux

    def get_current_general(self, V, I_out=None):
        '''
        Get the channel current given the voltage

        Input:
            V: array, voltage (len(V)==Ninloc)
        '''
        if self.flag == 1:
            return self.get_current_np(V, I_out=I_out)
        elif self.flag == 2:
            return self.get_current_nl(V, I_out=I_out)
        else:
            return self.get_current(V, I_out=I_out)

    def get_current(self, V, I_out=None):
        if I_out == None:
            I_out = np.zeros(self.Ninloc)
        if self.mode == 1:
            I_out[self.inloc_inds] -= \
                    self.g * \
                    (self.sv**self.powers).prod(1) * \
                    (V[self.inloc_inds] - self.e)
        else:
            I_out -= \
                    self.g * \
                    (self.sv**self.powers).prod(1) * \
                    (V - self.e)
        return I_out

    def get_conductance(self, G_out=None, I_out=None):
        '''
        Let the channel current be :math:`-g(V-e)`. Returns :math:`-g` and 
        :math:`-g*(E_eq-e)`
        '''
        if I_out == None:
            G_out = np.zeros(self.Ninloc)
            I_out = np.zeros(self.Ninloc)
        self.popen_aux = (self.sv**self.powers).prod(1)
        if self.mode == 1:
            G_out[self.inloc_inds] -= self.g * self.popen_aux
            I_out[self.inloc_inds] -= self.g * self.popen_aux * (self.Es_eq - self.e)
        else:
            G_out -= self.g * self.popen_aux
            I_out -= self.g * self.popen_aux * (self.Es_eq - self.e)
        return G_out, I_out

    def get_current_np(self, V, I_out=None):
        if I_out == None:
            I_out = np.zeros(self.Ninloc)
        if self.mode == 1:
            I_out[self.inloc_inds] -= \
                    self.g * \
                    ((self.sv**self.powers).prod(1) - self.popeneq) * \
                    (V[self.inloc_inds] - self.e)
        else:
            I_out -= \
                    self.g * \
                    ((self.sv**self.powers).prod(1) - self.popeneq) * \
                    (V - self.e)
        return I_out

    
class h(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-43., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        Hcn channel from (Bal and Oertel, 2000)
        '''
        self.ion = ''
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        self.tauf = 40. # ms
        self.taus = 300. # ms
        
        self.tau_array = np.array([[self.tauf], [self.taus]])
        
        self.varnames = np.array([['hf'], ['hs']])
        self.statevar = np.array([[1./(1.+np.exp((V0+82.)/7.))], [1./(1.+np.exp((V0+82.)/7.))]])
        self.powers = np.array([[1],[1]], dtype=int)
        self.factors = np.array([[1.-self.ratio], [self.ratio]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = (1.-self.ratio)*self.statevars[0,0] + self.ratio*self.statevars[1,0]
            
            self.varinf = np.array([[1./(1.+sp.exp((self.spV+82.)/7.))], [1./(1.+sp.exp((self.spV+82.)/7.))]])
            # make array of sympy floats to render it sympy compatible
            self.tau = np.zeros(self.tau_array.shape, dtype=object)
            for ind, tau in np.ndenumerate(self.tau_array):
                self.tau[ind] = sp.Float(self.tau_array[ind])
            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.],[0.]])
            self.Vstatevar = np.array([[0.],[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        return (1./(1.+np.exp((V+82.)/7.)) - sv) / self.tau_array


class h_HAY(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-45., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        Hcn channel from (Kole, Hallermann and Stuart, 2006)

        Used in (Hay, 2011)
        '''
        self.ion = ''
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        
        self.varnames = np.array([['m']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0))]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]
            
            spalpham = 0.001 * 6.43 * (self.spV + 154.9) / (sp.exp((self.spV + 154.9) / 11.9) - 1.)
            spbetam = 0.001 * 193. * sp.exp(self.spV / 33.1)

            self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
            # make array of sympy floats to render it sympy compatible
            self.tau = np.array([[1. / (spalpham + spbetam)]])

            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def alpham(self, V):
        return 0.001 * 6.43 * (V + 154.9) / (np.exp((V + 154.9) / 11.9) - 1.)

    def betam(self, V):
        return 0.001 * 193. * np.exp(V / 33.1)

    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        svinf = np.array([[am / (am + bm)]])
        taus = np.array([[1. / (am + bm)]])
        return (svinf - sv) / taus


class Na(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=6.3):
        '''
        Sodium channel from the HH model
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2

        self.q10 = 3.**((temp - 6.3) / 10.)
        # self.q10 = 1.
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0)), \
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0))]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)

        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham = .1 * -(self.spV+40.)/(sp.exp(-(self.spV+40.)/10.) - 1.)  #1/ms
            spbetam = 4. * sp.exp(-(self.spV+65.)/18.)  #1/ms
            spalphah = .07 * sp.exp(-(self.spV+65.)/20.)   #1/ms
            spbetah = 1. / (sp.exp(-(self.spV+35.)/10.) + 1.)   #1/ms
            
            self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
            self.tau = np.array([[1. / (self.q10*(spalpham + spbetam)), 1. / (self.q10*(spalphah + spbetah))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm), ah / (ah + bh)]])
        taus = np.array([[1. / (self.q10*(am + bm)), 1. / (self.q10*(ah + bh))]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        if type(V) is np.float64 or type(V) is float:
            return .1  * vtrap(-(V+40.),10.)
        else:
            return .1  * -(V+40.) / (np.exp(-(V+40.)/10.) - 1.) 
        
    def betam(self, V):
        return  4.   * np.exp(-(V+65.)/18.)
        
    def alphah(self, V):
        return .07 * np.exp(-(V+65.)/20.)
        
    def betah(self, V):
        return 1.   / (np.exp(-(V+35.)/10.) + 1.)


class Na_Branco(ionChannel):
    def __init__(self, g=0.120*1e6, e=50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.): 
        ''' sodium channel found in (Branco, 2011) code'''
        self.ion = 'na'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0)), \
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0))]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0

    def alpham(self, V):
        if type(V) is np.float64 or type(V) is float:
            return .1  * vtrap(-(V+40.),10.)
        else:
            return .1  * -(V+40.) / (np.exp(-(V+40.)/10.) - 1.) 
        
    def betam(self, V):
        return  4.   * np.exp(-(V+65.)/18.)
        
    def alphah(self, V):
        return .07 * np.exp(-(V+65.)/20.)
        
    def betah(self, V):
        return 1.   / (np.exp(-(V+35.)/10.) + 1.)

    def trap(self, V, p1, p2, p3): pass


class Na_p(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Derived by (Hay, 2011) from (Magistretti and Alonso, 1999)

        Used in (Hay, 2011)

        !!! Does not work !!!
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ 1. / (1. + np.exp(-(V0 + 52.6) / 4.6)) ,
                                    1. / (1. + np.exp( (V0 + 48.8) / 10.)) ]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham =   0.182   * (self.spV + 38. ) / (1. - sp.exp(-(self.spV + 38. ) / 6.  ))  #1/ms
            spbetam  = - 0.124   * (self.spV + 38. ) / (1. - sp.exp( (self.spV + 38. ) / 6.  ))  #1/ms
            spalphah = - 2.88e-6 * (self.spV + 17. ) / (1. - sp.exp( (self.spV + 17. ) / 4.63))   #1/ms
            spbetah  =   6.94e-6 * (self.spV + 64.4) / (1. - sp.exp(-(self.spV + 64.4) / 2.63))   #1/ms
            
            self.varinf = np.array([[   1. / (1. + sp.exp(-(self.spV + 52.6) / 4.6)) , 
                                        1. / (1. + sp.exp( (self.spV + 48.8) / 10.)) ]])
            self.tau = np.array([[(6./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        if type(V) == np.float64 or type(V) == float:
            if V == -38. or V == -17. or V == -64.4:
                V += 0.0001
        else:
            ind = np.where(V == -38. or V == -17. or V == -64.4)[0]
            V[ind] += 0.0001
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[1. / (1. + np.exp(-(V + 52.6) / 4.6)) , 
                           1. / (1. + np.exp( (V + 48.8) / 10.)) ]])
        taus = np.array([[  (6. / (am + bm)) / 2.95 , 
                            (1. / (ah + bh)) / 2.95 ]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return 0.182 * (V + 38.) / (1. - np.exp(-(V + 38.) / 6.))
        
    def betam(self, V):
        return - 0.124 * (V + 38.) / (1. - np.exp((V + 38.) / 6.))
        
    def alphah(self, V):
        return -2.88e-6 * (V + 17.) / (1. - np.exp((V + 17.) / 4.63))
        
    def betah(self, V):
        return  6.94e-6 * (V + 64.4) / (1. - np.exp(-(V + 64.4) / 2.63))


class Na_Ta(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        (Colbert and Pan, 2002)

        Used in (Hay, 2011)
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ self.alpham(V0) / (self.alpham(V0) + self.betam(V0)) ,
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0)) ]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham =   0.182 * (self.spV + 38.) / (1. - sp.exp(-(self.spV + 38.) / 6.))  #1/ms
            spbetam  = - 0.124 * (self.spV + 38.) / (1. - sp.exp( (self.spV + 38.) / 6.))  #1/ms
            spalphah = - 0.015 * (self.spV + 66.) / (1. - sp.exp( (self.spV + 66.) / 6.))   #1/ms
            spbetah  =   0.015 * (self.spV + 66.) / (1. - sp.exp(-(self.spV + 66.) / 6.))  #1/ms
            
            self.varinf = np.array([[   spalpham / (spalpham + spbetam) , 
                                        spalphah / (spalphah + spbetah) ]])
            self.tau = np.array([[(1./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        if type(V) == np.float64 or type(V) == float:
            if V == -38. or V == -66.:
                V += 0.0001
        else:
            ind = np.where(V == -38. or V == -66.)[0]
            V[ind] += 0.0001
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm) , 
                           ah / (ah + bh) ]])
        taus = np.array([[  (1. / (am + bm)) / 2.95 , 
                            (1. / (ah + bh)) / 2.95 ]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return   0.182 * (V + 38.) / (1. - np.exp(-(V + 38.) / 6.))
        
    def betam(self, V):
        return - 0.124 * (V + 38.) / (1. - np.exp( (V + 38.) / 6.))
        
    def alphah(self, V):
        return - 0.015 * (V + 66.) / (1. - np.exp( (V + 66.) / 6.))
        
    def betah(self, V):
        return   0.015 * (V + 66.) / (1. - np.exp(-(V + 66.) / 6.))


class Na_Ta2(ionChannel):
    def __init__(self, g = 0.120*1e6, e = 50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        (Colbert and Pan, 2002) 

        Shifted by 6 mV from Na_Ta

        Used in (Hay, 2011)
        '''
        self.ion = 'na'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ self.alpham(V0) / (self.alpham(V0) + self.betam(V0)) ,
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0)) ]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            spalpham =   0.182 * (self.spV + 32.) / (1. - sp.exp(-(self.spV + 32.) / 6.))  #1/ms
            spbetam  = - 0.124 * (self.spV + 32.) / (1. - sp.exp( (self.spV + 32.) / 6.))  #1/ms
            spalphah = - 0.015 * (self.spV + 60.) / (1. - sp.exp( (self.spV + 60.) / 6.))   #1/ms
            spbetah  =   0.015 * (self.spV + 60.) / (1. - sp.exp(-(self.spV + 60.) / 6.))  #1/ms
            
            self.varinf = np.array([[   spalpham / (spalpham + spbetam) , 
                                        spalphah / (spalphah + spbetah) ]])
            self.tau = np.array([[(1./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        if type(V) == np.float64 or type(V) == float:
            if V == -32. or V == -60.:
                V += 0.0001
        else:
            ind = np.where(V == -32. or V == -60.)[0]
            V[ind] += 0.0001
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm) , 
                           ah / (ah + bh) ]])
        taus = np.array([[  (1. / (am + bm)) / 2.95 , 
                            (1. / (ah + bh)) / 2.95 ]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return   0.182 * (V + 32.) / (1. - np.exp(-(V + 32.) / 6.))
        
    def betam(self, V):
        return - 0.124 * (V + 32.) / (1. - np.exp( (V + 32.) / 6.))
        
    def alphah(self, V):
        return - 0.015 * (V + 60.) / (1. - np.exp( (V + 60.) / 6.))
        
    def betah(self, V):
        return   0.015 * (V + 60.) / (1. - np.exp(-(V + 60.) / 6.))


class K(ionChannel):
    def __init__(self,  g=0.036*1e6, e=-77., V0=-65, conc0=[], nonlinear=False, calc=False, temp=6.3):
        '''
        Potassium channel from HH model
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2

        self.q10 = 3.**((temp-6.3) / 10.)
        # self.q10 = 1.
        
        self.varnames = np.array([['n']])
        self.statevar = np.array([[self.alphan(V0) / (self.alphan(V0) + self.betan(V0))]])
        self.powers = np.array([[4]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**4
            
            spalphan = -0.01 * (self.spV + 55.) / (sp.exp(-(self.spV + 55.)/10.) - 1.)
            spbetan = .125* sp.exp(-(self.spV + 65.)/80.)
            
            self.varinf = np.array([[spalphan / (spalphan + spbetan)]])
            self.tau = np.array([[1. / (self.q10*(spalphan + spbetan))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        an = self.alphan(V); bn = self.betan(V)
        svinf = np.array([[an / (an + bn)]])
        taus = np.array([[1. / (self.q10*(an + bn))]])
        return (svinf - sv) / taus
        
    def alphan(self, V):
        if type(V) is np.float64 or type(V) is float:
            return .01 * vtrap(-(V+55.),10.)
        else:
            return .01 * -(V+55.) / (np.exp(-(V+55.)/10.) - 1)
    
    def betan(self, V): 
        return .125* np.exp(-(V+65.)/80.)


class Klva(ionChannel):
    def __init__(self, g=0.001*1e6, e=-106., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Low voltage activated potassium channel (Kv1) from (Mathews, 2010)
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.g = g #uS/cm^2
        self.e = e #mV
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[1./(1.+np.exp(-(V0+57.34)/11.7)),  0.73/(1.+np.exp((V0+67.)/6.16)) + 0.27]])
        self.powers = np.array([[4, 1]], dtype=int)
        self.factors = np.array([[1.]])
        
        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)

        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**4 * self.statevars[0,1]
            
            self.varinf = np.array([[1./(1.+sp.exp(-(self.spV+57.34)/11.7)), 0.73/(1.+sp.exp((self.spV+67.)/6.16)) + 0.27]])
            self.tau = np.array([[(21.5/(6.*sp.exp((self.spV+60.)/7.) + 24.*sp.exp(-(self.spV+60.)/50.6)) + 0.35), \
                                (170./(5.*sp.exp((self.spV+60.)/10.) + sp.exp(-(self.spV+70.)/8.)) + 10.7)]]) # ms
            # self.tau = np.array([[1.080549, 58.346879]])
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[1./(1.+np.exp(-(V+57.34)/11.7)),  0.73/(1.+np.exp((V+67.)/6.16)) + 0.27]])
        taus = np.array([[(21.5/(6.*np.exp((V+60.)/7.) + 24.*np.exp(-(V+60.)/50.6)) + 0.35), \
                            (170./(5.*np.exp((V+60.)/10.) + np.exp(-(V+70.)/8.)) + 10.7)]]) # ms
        # taus = np.array([[1.080549, \
        #                     58.346879]]) # ms
        return (svinf - sv) / taus


class m(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-80., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        M-type potassium current (Adams, 1982)

        Used in (Hay, 2011)

        !!! does not work when e > V0 !!!
        '''
        self.ion = 'k'
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        
        self.varnames = np.array([['m']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0))]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]
            
            spalpham = 3.3e-3 * sp.exp( 2.5 * 0.04 * (self.spV + 35.))
            spbetam = 3.3e-3 * sp.exp(-2.5 * 0.04 * (self.spV + 35.))

            self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
            # make array of sympy floats to render it sympy compatible
            self.tau = np.array([[(1. / (spalpham + spbetam)) / 2.95]])# 

            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def alpham(self, V):
        return 3.3e-3 * np.exp( 2.5 * 0.04 * (V + 35.))

    def betam(self, V):
        return 3.3e-3 * np.exp(-2.5 * 0.04 * (V + 35.))

    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        svinf = np.array([[am / (am + bm)]])
        taus = np.array([[(1. / (am + bm)) / 2.95]])# 
        return (svinf - sv) / taus


class Kv3_1(ionChannel):
    def __init__(self, g=0.0038*1e3, e=-80., V0=-65, conc0=[], nonlinear=False, calc=False, ratio=0.2, temp=0.):
        '''
        Shaw-related potassium channel (Rettig et al., 1992)

        Used in (Hay et al., 2011)
        '''
        self.ion = 'k'
        self.concentrations = []

        self.g = g # uS/cm2
        self.e = e # mV
        
        self.ratio = ratio
        
        self.varnames = np.array([['m']])
        self.statevar = np.array([[1. / (1. + np.exp(-(V0 - 18.7) / 9.7))]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]

            self.varinf = np.array([[ 1. / (1. + sp.exp(-(self.spV - 18.7) / 9.7)) ]])
            self.tau = np.array([[ 4. / (1. + sp.exp(-(self.spV + 46.56) / 44.14)) ]])

            self.fstatevar = (self.varinf - self.statevars) / self.tau
            
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
            
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[ 1. / (1. + np.exp(-(V - 18.7) / 9.7)) ]])
        taus = np.array([[ 4. / (1. + np.exp(-(V + 46.56) / 44.14)) ]])
        return (svinf - sv) / taus


class Kpst(ionChannel):
    def __init__(self, g=0.001*1e6, e=-106., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Persistent Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.g = g #uS/cm^2
        self.e = e #mV
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ 1. / (1. + np.exp(-(V0 + 11.) / 12.)),  
                                    1. / (1. + np.exp( (V0 + 64.) / 11.))]])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            self.varinf = np.array([[1. / (1. + sp.exp(-(self.spV + 11.) / 12.)) , 
                                     1. / (1. + sp.exp( (self.spV + 64.) / 11.)) ]])
            self.tau = np.array([[(3.04 + 17.3 * sp.exp(-((self.spV + 60.) / 15.9)**2) + 25.2 * sp.exp(-((self.spV + 60.) / 57.4)**2)) / 2.95, \
                                (360. + (1010. + 24. * (self.spV + 65.)) * sp.exp(-((self.spV + 85.) / 48.)**2)) / 2.95]]) # ms
            # self.tau = np.array([[1.080549, 58.346879]])
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[ 1. / (1. + np.exp(-(V + 11.) / 12.)) ,  
                            1. / (1. + np.exp( (V + 64.) / 11.)) ]])
        # taum fitted to:
        # if V < -50.:
        #     return (1.25 + 175.03 * np.exp( 0.026 * V)) / 2.95
        # else:
        #     return (1.25 + 13.    * np.exp(-0.026 * V)) / 2.95
        taus = np.array([[ (3.04 + 17.3 * np.exp(-((V + 60.) / 15.9)**2) + 25.2 * np.exp(-((V + 60.) / 57.4)**2)) / 2.95, 
                            (360. + (1010. + 24. * (V + 65.)) * np.exp(-((V + 85.) / 48.)**2)) / 2.95 ]]) # ms
        return (svinf - sv) / taus


class Ktst(ionChannel):
    def __init__(self, g=0.001*1e6, e=-106., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Transient Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        
        self.g = g #uS/cm^2
        self.e = e #mV
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[ 1. / (1. + np.exp(-(V0 + 10.) / 19.)),  
                                    1. / (1. + np.exp( (V0 + 76.) / 10.))]])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            self.varinf = np.array([[   1. / (1. + sp.exp(-(self.spV + 10.) / 19.)) ,  
                                        1. / (1. + sp.exp( (self.spV + 76.) / 10.)) ]])
            self.tau = np.array([[  (0.34 + 0.92 * sp.exp(-((self.spV + 81.) / 59.)**2)) / 2.95 , 
                                    (8.   + 49.  * sp.exp(-((self.spV + 83.) / 23.)**2)) / 2.95]]) # ms
            # self.tau = np.array([[1.080549, 58.346879]])
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
    
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[ 1. / (1. + np.exp(-(V + 10.) / 19.)) ,  
                            1. / (1. + np.exp( (V + 76.) / 10.)) ]])
        taus = np.array([[  (0.34 + 0.92 * np.exp(-((V + 81.) / 59.)**2)) / 2.95 , 
                            (8.   + 49.  * np.exp(-((V + 83.) / 23.)**2)) / 2.95 ]]) # ms
        return (svinf - sv) / taus


class KA(ionChannel):
    def __init__(self, g=0.0477*1e6, e=-75., V0=-65., conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        A-type potassium (Abbott, 2000) (Connor-Stevens model)
        '''
        self.ion = 'k'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['a', 'b']])
        self.statevar = np.array([[(0.0761 * np.exp(0.0314 * (V0+94.22)) / (1. + np.exp(0.0346 * (V0+1.17))))**(1./3.),
                                    (1. / (1. + np.exp(0.0688 * (V0 + 53.3))))**4]])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 * self.statevars[0,1]
            
            self.varinf = np.array([[(0.0761 * sp.exp(0.0314 * (self.spV+94.22)) / (1. + sp.exp(0.0346 * (self.spV+1.17))))**(1./3.),
                                     (1. / (1. + sp.exp(0.0688 * (self.spV + 53.3))))**4]])
            self.tau = np.array([[0.3632 + 1.158 / (1. + sp.exp(0.0497 * (self.spV+55.96))), 
                                    1.24 + 2.678 / (1. + sp.exp(0.0624 * (self.spV+50.)))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[(0.0761 * np.exp(0.0314 * (V+94.22)) / (1. + np.exp(0.0346 * (V+1.17))))**(1./3.),  
                                (1. / (1. + np.exp(0.0688 * (V + 53.3))))**4]])
        taus = np.array([[0.3632 + 1.158 / (1. + np.exp(0.0497 * (V+55.96))), \
                            1.24 + 2.678 / (1. + np.exp(0.0624 * (V+50.)))]]) # ms
        return (svinf - sv) / taus


class KA_prox(ionChannel): # TODO finish implementation
    def __init__(self, g=0.0477*1e6, e=-90., V0=-65., conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        A-type potassium (Kellems, 2010)

        !!! works in testsuite, but unstable in certain cases !!!
        '''
        self.ion = 'k'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['n', 'l']])
        self.statevar = np.array([[1. / (1.+self.alphan(V0)), 1. / (1.+self.alphal(V0))]])
        self.powers = np.array([[1,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**3 #* self.statevars[0,1]

            alphan = sp.exp(-0.038*(1.5 + 1./(1.+sp.exp((self.spV+40.)/5.))) * (self.spV-11.))
            betan = sp.exp(-0.038*(0.825 + 1./(1.+sp.exp((self.spV+40.)/5.))) * (self.spV-11.))
            alphal = sp.exp(0.11*(self.spV+56.))
            
            self.varinf = np.array([[1. / (1.+alphan), 1. / (1.+alphal)]])
            self.tau = np.array([[4.*betan / (1.+alphan), 0.2 + 27. / (1. + sp.exp(0.2-self.spV/22.))]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()

    def fun_statevar(self, V, sv, conc={}):
        taul = 0.2 + 27. / (1. + np.exp(0.2-V/22.))
        an = self.alphan(V); bn = self.betan(V); al = self.alphal(V)
        svinf = np.array([[1. / (1. + an), 1./(1. + al)]])
        taus = np.array([[4.*bn / (1+an), taul]])
        return (svinf - sv) / taus
        
    def alphan(self, V):
        return np.exp(-0.038*(1.5 + 1./(1.+np.exp((V+40.)/5.))) * (V-11.))
    
    def betan(self, V): 
        return np.exp(-0.038*(0.825 + 1./(1.+np.exp((V+40.)/5.))) * (V-11.))

    def alphal(self, V):
        return np.exp(0.11*(V+56.))


class SK(ionChannel):
    def __init__(self, g=0.00001*1e6, e=-80, V0=-65., conc0=[1e-4], nonlinear=False, calc=False, temp=0.):
        '''
        SK-type calcium-activated potassium current (Kohler et al., 1996)

        !!!Untested, not functional yet!!!

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = ['ca']

        self.e = e  # mV
        self.g = g  # uS/cm2

        self.varnames = np.array([['z']])
        self.statevar = np.array([[ 1./(1. + (0.00043/conc0[0])**4.8) ]])
        self.tau_array = np.array([[1.]])
        self.powers = np.array([[1.]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)

        self.nonlinear = nonlinear
        self.calc = calc

        self.V0 = V0
        self.conc0 = conc0

        if self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.zeros(len(self.concentrations), dtype=object)
            for ind, name in enumerate(self.concentrations):
                self.conc[ind] = sp.symbols(name)

            self.fun = self.statevars[0,0]**self.powers[0,0]

            self.varinf = np.array([[1./(1. + (0.00043/self.conc[0])**4.8)]], dtype=object)
            # make array of sympy floats to render it sympy compatible
            self.tau = np.zeros(self.tau_array.shape, dtype=object)
            for ind, tau in np.ndenumerate(self.tau_array):
                self.tau[ind] = sp.Float(self.tau_array[ind])
            self.fstatevar = (self.varinf - self.statevars) / self.tau

        if self.nonlinear:
            self.I0 = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None]) * (self.V0 - self.e)

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            # self.set_expansion_coeffs()

    def calc_linear(self, freqs):
        return {'V': np.zeros(freqs.shape, dtype=complex)} 

    def advance(self, V, dt, conc={'Ca':0.}):
        '''
        advance the state variables of the ion channel
        '''
        self.statevar += dt * self.fun_statevar(V, self.statevar, conc=conc)

    def getCurrent(self, V):
        '''
        returns the transmembrane current in nA, if self.nonlinear is True,
        returns the current without the equilibrium part
        '''
        I = - self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None]) * (V - self.e)
        if self.nonlinear:
            I -= self.I0
        return I

    def fun_statevar(self, V, sv, conc={'Ca':1e-4}):
        svinf = np.array([[1./(1. + (0.00043/conc[self.concentrations[0]])**4.8)]])
        return (svinf - sv) / self.tau_array


class Ca_LVA(ionChannel):
    def __init__(self, g=0.00001*1e6, e=50., V0=-65., conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        LVA calcium channel (Avery and Johnston, 1996; tau from Randall, 1997)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []

        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[1. / (1. + np.exp(-(V0 + 40.)/6.)), \
                                    1. / (1. + np.exp((V0 + 90.)/6.4))]])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            self.varinf = np.array([[1. / (1. + sp.exp(-(self.spV + 40.)/6.)), \
                                    1. / (1. + sp.exp((self.spV + 90.)/6.4))]])
            self.tau = np.array([[(5. + 20./(1. + sp.exp((self.spV  + 35.)/5.)))/2.95, 
                                    (20. + 50./(1. + sp.exp((self.spV + 50.)/7.)))/2.95]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        svinf = np.array([[1. / (1. + np.exp(-(V + 40.)/6.)), \
                                1. / (1. + np.exp((V + 90.)/6.4))]])
        taus = np.array([[(5. + 20./(1. + np.exp((V + 35.)/5.)))/2.95, 
                                (20. + 50./(1. + np.exp((V + 50.)/7.)))/2.95]]) # 1/ms
        return (svinf - sv) / taus


class Ca_HVA(ionChannel):
    def __init__(self, g=0.00001*1e6, e=50., V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        High voltage-activated calcium channel (Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = np.array([['m', 'h']])
        self.statevar = np.array([[self.alpham(V0) / (self.alpham(V0) + self.betam(V0)), \
                                    self.alphah(V0) / (self.alphah(V0) + self.betah(V0))]])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.zeros(self.varnames.shape, dtype=object)
            for ind, name in np.ndenumerate(self.varnames):
                self.statevars[ind] = sp.symbols(name)
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]**self.powers[0,0] * self.statevars[0,1]**self.powers[0,1]
            
            spalpham = -0.055 * (27. + self.spV) / (sp.exp(-(27. + self.spV)/3.8) - 1.)  #1/ms
            spbetam = 0.94 * sp.exp(-(75. + self.spV)/17.)  #1/ms
            spalphah = 0.000457 * sp.exp(-(13. + self.spV)/50.)   #1/ms
            spbetah = 0.0065 / (sp.exp(-(self.spV + 15.)/28.) + 1.)   #1/ms
            
            self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
            self.tau = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # 1/ms
            self.fstatevar = (self.varinf - self.statevars) / self.tau      
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.,0.]])
            self.Vstatevar = np.array([[0.,0.]])

        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, sv, conc={}):
        am = self.alpham(V); bm = self.betam(V)
        ah = self.alphah(V); bh = self.betah(V)
        svinf = np.array([[am / (am + bm), ah / (ah + bh)]])
        taus = np.array([[1. / (am + bm), 1. / (ah + bh)]])
        return (svinf - sv) / taus
        
    def alpham(self, V):
        return -0.055 * (27. + V) / (np.exp(-(27. + V)/3.8) - 1.)
        
    def betam(self, V):
        return 0.94 * np.exp(-(75. + V)/17.)
        
    def alphah(self, V):
        return 0.000457 * np.exp(-(13. + V)/50.)
        
    def betah(self, V):
        return 0.0065 / (np.exp(-(V + 15.)/28.) + 1.)

        
class L(ionChannel):
    def __init__(self,  g=0.0003*1e6, e=-54.4, V0=-65, conc0=[], nonlinear=False, calc=False, temp=0.):
        '''
        Leak current
        '''
        self.ion = ''
        self.concentrations = []
        
        self.e = e  # mV
        self.g = g  # uS/cm2
        
        self.varnames = []
        self.statevar = np.array([[1.]])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([[1.]])

        self.g0 = self.g * np.sum(self.factors * np.prod(self.statevar**self.powers, 1)[:,None])
        self.sv0 = copy.deepcopy(self.statevar)
        
        self.nonlinear = nonlinear
        self.calc = calc
        
        self.V0 = V0        
        
        if self.nonlinear or self.calc:
            # sympy expressions
            self.spV = sp.symbols('v')
            self.statevars = np.array([[sp.symbols('x')]])
            self.conc = np.array([], dtype=object)
            
            self.fun = self.statevars[0,0]
            
            self.varinf = np.array([[sp.Float(1.)]])
            self.tau = np.array([[sp.Float(1.)]]) # 1/ms
            self.fstatevar = np.array([[sp.Float(0.)]])  
                
        if self.nonlinear:
            self.lincoeffI, self.lincoeffstatevar = self.compute_lincoeff()
            self.linstatevar = copy.deepcopy(self.statevar)
            self.Istatevar = np.array([[0.]])
            self.Vstatevar = np.array([[0.]])
        if self.calc:
            self.fv = self.g * self.fun * (self.spV - self.e)
            self.set_expansion_coeffs()
            
    def fun_statevar(self, V, dt, conc={}):
        return np.array([[0.]])
        
    def set_expansion_coeffs(self, order=2): pass
    
    def calc_offset(self, freqs=None):
        if freqs==None:
            return self.g*(self.V0 - self.e)
        else:
            return self.g*(self.V0 - self.e)*np.ones(freqs.shape, dtype=complex)
        
    def calc_linear(self, freqs):
        return {'V': self.g*np.ones(freqs.shape, dtype=complex)}
    
    def calc_quadratic(self, freqs):
        return {'V': np.zeros((len(freqs), len(freqs)), dtype=complex)}


## make mod files ######################################################
if __name__ == "__main__":
    fcc = open('cython_code/ionchannels.cc', 'w')
    fh = open('cython_code/ionchannels.h', 'w')
    fstruct = open('cython_code/channelstruct.h', 'w')

    fh.write('#include <iostream>' + '\n')
    fh.write('#include <string>' + '\n')
    fh.write('#include <vector>' + '\n')
    fh.write('#include <string.h>' + '\n')
    fh.write('#include <stdlib.h>' + '\n')
    fh.write('#include <algorithm>' + '\n')
    fh.write('#include <math.h>' + '\n\n')
    fh.write('#include "memcurrent.h"' + '\n\n')
    fh.write('using namespace std;' + '\n\n')
    
    fcc.write('#include \"ionchannels.h\"' + '\n\n')

    fstruct.write('struct ionc_set{' + '\n')
    
    fstruct.close()
    fcc.close()
    fh.close()
    
    hchan = h(nonlinear=True)
    hchan.write_mod_file()
    hchan.write_cpp_code()
    
    h_HAYchan = h_HAY(nonlinear=True)
    h_HAYchan.write_mod_file()
    h_HAYchan.write_cpp_code()
    
    Nachan = Na(nonlinear=True)
    Nachan.write_mod_file()
    Nachan.write_cpp_code()
    
    Na_pchan = Na_p(nonlinear=True)
    Na_pchan.write_mod_file()
    Na_pchan.write_cpp_code()
    
    Na_Tachan = Na_Ta(nonlinear=True)
    Na_Tachan.write_mod_file()
    Na_Tachan.write_cpp_code()
    
    Na_Ta2chan = Na_Ta2(nonlinear=True)
    Na_Ta2chan.write_mod_file()
    Na_Ta2chan.write_cpp_code()
    
    Klvachan = Klva(nonlinear=True)
    Klvachan.write_mod_file()
    Klvachan.write_cpp_code()
    
    Kchan = K(nonlinear=True)
    Kchan.write_mod_file()
    Kchan.write_cpp_code()
    
    Kpstchan = Kpst(nonlinear=True)
    Kpstchan.write_mod_file()
    Kpstchan.write_cpp_code()
    
    Ktstchan = Ktst(nonlinear=True)
    Ktstchan.write_mod_file()
    Ktstchan.write_cpp_code()
    
    Kv3_1chan = Kv3_1(nonlinear=True)
    Kv3_1chan.write_mod_file()
    Kv3_1chan.write_cpp_code()
    
    mchan = m(nonlinear=True)
    mchan.write_mod_file()
    mchan.write_cpp_code()
    
    KAchan = KA(nonlinear=True)
    KAchan.write_mod_file()
    KAchan.write_cpp_code()
    
    KAproxchan = KA_prox(nonlinear=True)
    KAproxchan.write_mod_file()    
    KAproxchan.write_cpp_code()    
    
    SKchan = SK(nonlinear=True, calc=True)
    SKchan.write_mod_file()
    
    Ca_HVAchan = Ca_HVA(nonlinear=True)
    Ca_HVAchan.write_mod_file()
    Ca_HVAchan.write_cpp_code()
    
    Ca_LVAchan = Ca_LVA(nonlinear=True)
    Ca_LVAchan.write_mod_file()
    Ca_LVAchan.write_cpp_code()
    
    Lchan = L(nonlinear=True)
    Lchan.write_mod_file()
    Lchan.write_cpp_code()

    Caconc = conc_ca()
    Caconc.write_mod_file()

    fstruct = open('cython_code/channelstruct.h', 'a')
    fstruct.write('};')
    fstruct.close()
########################################################################
            
    
