import sympy as sp
import numpy as np
import scipy.optimize as so

import os


def _insert_function_prefixes(string, prefix='np',
                              functions=['exp', 'sin', 'cos', 'tan', 'pi']):
    '''
    Prefix all occurences in the input `string` of the functions in the
    `functions` list with the provided `prefix`.

    Parameters
    ----------
    string: string
        the input string
    prefix: string, optional
        the prefix that is put before each function. Defaults to `'np'`
    functions: list of strings, optional
        the list of functions that will be prefixed. Defaults to
        `['exp', 'sin', 'cos', 'tan', 'pi']`

    Returns
    -------
    string

    Examples
    --------
    >>> _insert_function_prefixes('5. * exp(0.) + 3. * cos(pi)')
    '5. * np.exp(0.) + 3. * np.cos(pi)'
    '''
    for func_name in functions:
        numpy_string = ''
        while len(string) > 0:
            ind = string.find(func_name)
            if ind == -1:
                numpy_string += string
                string = ''
            else:
                numpy_string += string[0:ind] + prefix + '.' + func_name
                string = string[ind+len(func_name):]
        string = numpy_string
    return string


class IonChannel(object):
    '''
    Base class for all different ion channel types.

    The algebraic form of the membrance current is stored in three numpy.arrays:
    `varnames`, `powers` and `factors`. An example of how the current is
    computed is given below:
        `varnames` = ``[['a00', 'a01', 'a02'],
                        ['a10', 'a11', 'a12']]``
        `powers` = ``[[n00, n01, n02],
                      [n10, n11, n12]]``
        `factors` = ``[f0, f1]``
    Then the corresponding probability that the channel is open is given by
        math::`f0 a00^{n00} a01^{n01} a02^{n02}
                + f1 a10^{n10} a11^{n11} a12^{n12})`

    Attributes
    ----------
    *Every derived class should define all these attributes in its constructor,
    before the base class constructor is called*
    varnames : 2d numpy.ndarray of strings
        The names associated with each channel state variable
    powers : 2d numpy.ndarray of floats or ints
        The powers to which each state variable is raised to compute the
        channels' open probalility
    factors : numpy.array of floats or ints
        factors which multiply each product of state variables in the sum
    varinf : 2d numpy.ndarray of sympy.expression instances
        The activation functions of the channel state variables
    tauinf : 2d numpy.ndarray of sympy.expression instances\
        The relaxation timescales of the channel state variables [ms]

    *The base class then defines the following attributes*
    statevars: 2d numpy.ndarray of sympy.symbols
        Symbols associated with the state variables
    fstatevar: 2d numpy.ndarray of sympy.expression instances
        The functions that give the time-derivative of the statevariables (i.e.
        math::`(varinf - var) / tauinf`)
    fun: sympy.expression
        The analytical form of the open probability
    coeff_curr: list of sympy.expression instances
        TODO
    coeff_statevar: list of sympy.expression instances
        TODO
    '''

    def __init__(self):
        '''
        Will give an ``AttributeError`` if initialized as is. Should only be
        initialized from its' derived classes that implement specific ion
        channel types.
        '''
        if not hasattr(self, 'ion'):
            self.ion = ''
        if not hasattr(self, 'concentrations'):
            self.concentrations = []
        # these attributes should be defined
        if not hasattr(self, 'varnames'):
            raise AttributeError('\'varnames\' is not defined')
        if not hasattr(self, 'powers'):
            raise AttributeError('\'powers\' is not defined')
        if not hasattr(self, 'factors'):
            raise AttributeError('\'factors\' is not defined')
        # define the sympy functions
        self.statevars = np.zeros(self.varnames.shape, dtype=object)
        for ind, name in np.ndenumerate(self.varnames):
            self.statevars[ind] = sp.symbols(name)
        self.fstatevar = (self.varinf - self.statevars) / self.tauinf
        # construct the sympy function for the ion channel current
        terms = []
        for ii, factor in enumerate(self.factors):
            terms.append(factor)
            for jj, var in enumerate(self.statevars[ii]):
                terms[-1] *= var**self.powers[ii,jj]
        self.p_open = sum(terms)
        # set the coefficients for the linear expansion
        # self.compute_lincoeff()
        # construct lambda functions for steady state activation
        self.f_varinf = self.lambdifyVarInf()
        # construct lambda functions for state variable time scales
        self.f_tauinf = self.lambdifyTauInf()
        # construct lambda function for passive opening
        self.f_p_open = self.lambdifyPOpen()
        # construct lambda function for linear current coefficient evaluations
        self.dp_dx, self.df_dv, self.df_dx = \
                        self.lambdifyDerivatives()

        # express statevar[0,0] as a function of the other state variables
        self.po = sp.symbols('po')
        sol = sp.solve(self.p_open - self.po, self.statevars[0,0])
        ind = np.where([sp.I not in s.atoms() for s in sol])[0][-1]
        sol = sol[ind]
        # print self.__class__.__name__
        # print self.p_open
        # print sol
        self.f_s00 = sp.lambdify((self.statevars, self.po), sol)

    def lambdifyVarInf(self):
        f_varinf = np.zeros(self.varnames.shape, dtype=object)
        for ind, varinf in np.ndenumerate(self.varinf):
            f_varinf[ind] = sp.lambdify(self.sp_v, varinf)
        return f_varinf

    def lambdifyTauInf(self):
        f_tauinf = np.zeros(self.varnames.shape, dtype=object)
        for ind, tauinf in np.ndenumerate(self.tauinf):
            f_tauinf[ind] = sp.lambdify(self.sp_v, tauinf)
        return f_tauinf

    def lambdifyPOpen(self):
        # arguments for lambda function
        args = [self.sp_v] + [statevar for ind, statevar in np.ndenumerate(self.statevars)]
        # return lambda function
        return sp.lambdify(args, self.p_open)

    def lambdifyDerivatives(self):
        # arguments for lambda function
        args = [self.sp_v] + [statevar for ind, statevar in np.ndenumerate(self.statevars)]
        # compute open probability derivatives to state vars
        dp_dx_aux = np.zeros(self.statevars.shape, dtype=object)
        for ind, var in np.ndenumerate(self.statevars):
            dp_dx_aux[ind] = sp.lambdify(args,
                                     sp.diff(self.p_open, var, 1))
        # compute state variable derivatives
        df_dv_aux = np.zeros(self.statevars.shape, dtype=object)
        df_dx_aux = np.zeros(self.statevars.shape, dtype=object)
        # differentiate
        for ind, var in np.ndenumerate(self.statevars):
            df_dv_aux[ind] = sp.lambdify(args,
                                     sp.diff(self.fstatevar[ind], self.sp_v, 1))
            df_dx_aux[ind] = sp.lambdify(args,
                                     sp.diff(self.fstatevar[ind], var, 1))
        # define convenient functions
        def dp_dx(*args):
            dp_dx_list = [[] for _ in range(self.statevars.shape[0])]
            for ind, dp_dx_ in np.ndenumerate(dp_dx_aux):
                dp_dx_list[ind[0]].append(dp_dx_aux[ind](*args))
            return np.array(dp_dx_list)
        def df_dv(*args):
            df_dv_list = [[] for _ in range(self.statevars.shape[0])]
            for ind, df_dv_ in np.ndenumerate(df_dv_aux):
                df_dv_list[ind[0]].append(df_dv_aux[ind](*args))
            return np.array(df_dv_list)
        def df_dx(*args):
            df_dx_list = [[] for _ in range(self.statevars.shape[0])]
            for ind, df_dx_ in np.ndenumerate(df_dx_aux):
                df_dx_list[ind[0]].append(df_dx_aux[ind](*args))
            return np.array(df_dx_list)

        return dp_dx, df_dv, df_dx

    def computePOpen(self, v, statevars=None):
        if statevars is None:
            args = [v] + [f_varinf(v) for _, f_varinf in np.ndenumerate(self.f_varinf)]
        else:
            args = [v] + [var0 for var0 in statevars.reshape(-1, *statevars.shape[2:])]
        return self.f_p_open(*args)

    def computeDerivatives(self, v, statevars=None):
        if statevars is None:
            args = [v] + [f_varinf(v) for _, f_varinf in np.ndenumerate(self.f_varinf)]
        else:
            args = [v] + [var0 for var0 in statevars.reshape(-1, *statevars.shape[2:])]
        return self.dp_dx(*args), self.df_dv(*args), self.df_dx(*args)

    def computeVarInf(self, v):
        if isinstance(v, np.ndarray):
            dims = tuple(list(tuple(self.f_varinf.shape)) + list(v.shape))
            slice_ind = [slice(dd) for dd in v.shape]
        else:
            dims = self.f_varinf.shape
            slice_ind = []
        res = np.zeros(dims)
        for ind, f_varinf in np.ndenumerate(self.f_varinf):
            ind_slice = tuple(list(ind) + slice_ind)
            res[ind_slice] = f_varinf(v)
        return res

    def computeTauInf(self, v):
        if isinstance(v, np.ndarray):
            dims = tuple(list(tuple(self.f_tauinf.shape)) + list(v.shape))
            slice_ind = [slice(dd) for dd in v.shape]
        else:
            dims = self.f_tauinf.shape
            slice_ind = []
        res = np.zeros(dims)
        for ind, f_tauinf in np.ndenumerate(self.f_tauinf):
            ind_slice = tuple(list(ind) + slice_ind)
            res[ind_slice] = f_tauinf(v)
        return res

    def computeLinear(self, v, freqs, statevars=None):
        dp_dx_arr, df_dv_arr, df_dx_arr = self.computeDerivatives(v, statevars=statevars)
        lin_f = np.zeros_like(freqs)
        for ind, dp_dx_ in np.ndenumerate(dp_dx_arr):
            df_dv_ = df_dv_arr[ind] * 1e3 # convert to 1 / s
            df_dx_ = df_dx_arr[ind] * 1e3 # convert to 1 / s
            # add to the impedance contribution
            lin_f += dp_dx_ * df_dv_ / (freqs - df_dx_)
        return lin_f

    def computeLinSum(self, v, freqs, e_rev, statevars=None):
        return (e_rev - v) * self.computeLinear(v, freqs, statevars=statevars) - \
               self.computePOpen(v, statevars=statevars)

    def findMaxCurrent(self, freqs, e_rev):
        def f_min(xx):
            xv = xx[1:].reshape(self.statevars.shape)
            val = 1. / np.abs(np.sum(self.computeLinSum(xx[0], freqs, e_rev, statevars=xv)))
            return val
        # optimization
        x0 = [-45.] + [0.5 for _ in xrange(self.statevars.size)]
        bounds = [(-90., 0.)] + [(0., 1.) for _ in xrange(self.statevars.size)]
        res = so.minimize(f_min, x0, bounds=bounds)
        return res['x'][0], res['x'][1:].reshape(self.statevars.shape)

    def findMaxCurrentVGiven(self, v, freqs, e_rev):
        def f_min(xx):
            xv = xx.reshape(self.statevars.shape)
            val = 1. / np.abs(np.sum(self.computeLinSum(v, freqs, e_rev, statevars=xv)))
            return val
        # optimization
        x0 = [0.5 for _ in xrange(self.statevars.size)]
        bounds = [(0., 1.) for _ in xrange(self.statevars.size)]
        res = so.minimize(f_min, x0, bounds=bounds)
        return res['x'].reshape(self.statevars.shape)

    def findStatevarsVPGiven(self, v, p_open):
        def f_object(xx):
            xv = xx.reshape(self.statevars.shape)
            return np.abs(p_open - self.computePOpen(v, statevars=xv))
        # the bounds for the optimization
        constraints = ({'type': 'ineq', 'fun': lambda xx: xx - 0.1},
                       {'type': 'ineq', 'fun': lambda xx: 0.9 - xx})
        # initialization
        x0 = np.array([0.5 for _ in xrange(self.statevars.size)])
        # optimization
        res = so.minimize(f_object, x0, method='SLSQP', constraints=constraints)
        return res['x'].reshape(self.statevars.shape)

    # def findMaxLinear(self, v, freqs, e_rev, p_open=None):
    #     # contraint function for the optimization
    #     if p_open is not None:
    #     # if False:
    #         def f_constraint(xx):
    #             xv = xx.reshape(self.statevars.shape)
    #             return p_open - self.computePOpen(v, statevars=xv)
    #         constraints = ({'type': 'ineq', 'fun': lambda xx: 0.001 - f_constraint(xx)/p_open},
    #                        {'type': 'ineq', 'fun': lambda xx: 0.001 + f_constraint(xx)/p_open},
    #                        {'type': 'ineq', 'fun': lambda xx: xx},
    #                        {'type': 'ineq', 'fun': lambda xx: 1. - xx})
    #     else:
    #         constraints = ({'type': 'ineq', 'fun': lambda xx: xx},
    #                        {'type': 'ineq', 'fun': lambda xx: 1. - xx})
    #     # the objective function for the optimization
    #     def f_object(xx):
    #         xv = xx.reshape(self.statevars.shape)
    #         val = 1. / np.abs(np.sum(self.computeLinear(v, freqs, statevars=xv)))
    #         return val
    #     # the bounds for the optimization
    #     # bounds = [(0., 1.) for _ in xrange(self.statevars.size)]
    #     # initialization
    #     # if p_open is not None:
    #     if False:
    #         x0 = self.findStatevarsVPGiven(v, p_open).reshape(self.statevars.size)
    #     else:
    #         x0 = np.array([0.5 for _ in xrange(self.statevars.size)])

    #     # print '\n>>>>>'
    #     # print self.__class__.__name__
    #     # print 'p_open =', p_open
    #     # print 'sv_0 =', x0.reshape(self.statevars.shape)
    #     # print 'f_constraint =', f_constraint(x0)
    #     # print 'f_object =', f_object(x0)
    #     # print '<<<<<\n'
    #     # optimization
    #     res_0 = so.minimize(f_object, x0, method='SLSQP', constraints=constraints)
    #     x0 = np.array([0.1 for _ in xrange(self.statevars.size)])
    #     res_1 = so.minimize(f_object, x0, method='COBYLA', constraints=constraints)
    #     # res = so.minimize(f_object, x0, method='SLSQP', bounds=bounds)
    #     if np.abs(f_object(res_0['x'])) < np.abs(f_object(res_1['x'])):
    #         return res_0['x'].reshape(self.statevars.shape)
    #     else:
    #         return res_1['x'].reshape(self.statevars.shape)
    #     # return res_1['x'].reshape(self.statevars.shape)

    # def findMaxLinear(self, v, freqs, e_rev, p_open=None):
    #     if self.statevars.size > 1:
    #         def to_statevar(xx):
    #             if self.statevars.shape[0] > 1:
    #                 xv = np.array([[self.f_s00(np.array([[np.nan], [xx]]), p_open)], [xx]])
    #             elif self.statevars.shape[1] > 1:
    #                 xv = np.array([[self.f_s00(np.array([[np.nan, xx]]), p_open), xx]])
    #             return xv
    #         # the objective function for the optimization
    #         def f_object(xx):
    #             xv = to_statevar(xx)
    #             val = 1. / np.abs(np.sum(self.computeLinear(v, freqs, statevars=xv)))
    #             return val
    #         # the bounds for the optimization
    #         bounds = [0.001, 0.999]
    #         # optimization
    #         res_0 = so.minimize_scalar(f_object, method='bounded', bounds=bounds)
    #         return to_statevar(res_0['x'])
    #     else:
    #         return np.array([[self.f_s00(np.zeros_like(self.statevars), p_open)]])

    def getStatevarsPOpen(self, p_open):
        xx = 1.
        if self.statevars.shape[0] > 1:
            xv = np.array([[self.f_s00(np.array([[np.nan], [xx]]), p_open)], [xx]])
        elif self.statevars.shape[1] > 1:
            xv = np.array([[self.f_s00(np.array([[np.nan, xx]]), p_open), xx]])
        else:
            xv = np.array([[self.f_s00(np.zeros_like(self.statevars), p_open)]])
        return xv


    def computeFreqIMax(self, v, e_rev, f_bounds=(0.,10000.)):
        '''
        Computes the frequency of voltage fluctuation at which the channel current,
        linearized around the holding potential `v`, is maximal

        Parameters
        ----------
        v: `float` or `np.ndarray`
            the holding potential around which the voltage is linearized [mV]
        e_rev: `float`
            the reversal potential of the channel [mV]
        f_bounds: `tuple` `(f_min, f_max)`
            the minimal resp. maximal frequencies [Hz] of the search interval

        Returns
        -------
        `np.ndarray`
            the frequency value as a real number (same shape as `v`)
        '''
        # optimization function
        def f_min(freq, u, e_r):
            return -np.abs(self.computeLinSum(u, 1j*freq, e_r))
        # find minima
        if hasattr(v, '__iter__') or hasattr(v, '__getitem__'):
            freq_vals = np.zeros_like(np.array(v))
            for ind, vv in np.ndenumerate(v):
                res = so.minimize_scalar(f_min, bounds=f_bounds, args=(vv, e_rev))
                freq_vals[ind] = res['x']
            return freq_vals
        else:
            return so.minimize_scalar(f_min, bounds=f_bounds, args=(v, e_rev))['x']

    def writeModFile(self, path, g=0., e=0.):
        '''
        Writes a modfile of the ion channel for simulations with neuron
        '''
        file = open(os.path.join(path, 'I' + self.__class__.__name__ + '.mod'), 'w')

        file.write(': This mod file is automaticaly generated by the ' +
                    '``neat.channels.ionchannels`` module\n\n')

        file.write('NEURON {\n')
        file.write('    SUFFIX I' + self.__class__.__name__ + '\n')
        if self.ion == '':
            file.write('    NONSPECIFIC_CURRENT i' + '\n')
        else:
            file.write('    USEION ' + self.ion + ' WRITE i' + self.ion + '\n')
        if len(self.concentrations) > 0:
            for concstring in self.concentrations:
                file.write('    USEION ' + concstring + ' READ ' \
                                      + concstring + 'i' + '\n')
        file.write('    RANGE  g, e' + '\n')
        varstring = 'var0inf'
        taustring = 'tau0'
        for ind in range(len(self.varinf.flatten()[1:])):
            varstring += ', var' + str(ind+1) + 'inf'
            taustring += ', tau' + str(ind+1)
        file.write('    GLOBAL ' + varstring + ', ' + taustring + '\n')
        file.write('    THREADSAFE' + '\n')
        file.write('}\n\n')

        file.write('PARAMETER {\n')
        file.write('    g = ' + str(g*1e-6) + ' (S/cm2)' + '\n')
        file.write('    e = ' + str(e) + ' (mV)' + '\n')
        for ion in self.concentrations:
            file.write('    ' + ion + 'i (mM)' + '\n')
        file.write('}\n\n')

        file.write('UNITS {\n')
        file.write('    (mA) = (milliamp)' + '\n')
        file.write('    (mV) = (millivolt)' + '\n')
        file.write('    (mM) = (milli/liter)' + '\n')
        file.write('}\n\n')

        file.write('ASSIGNED {\n')
        file.write('    i' + self.ion + ' (mA/cm2)' + '\n')
        # if self.ion != '':
        #     f.write('    e' + self.ion + ' (mV)' + '\n')
        for ind in range(len(self.varinf.flatten())):
            file.write('    var' + str(ind) + 'inf' + '\n')
            file.write('    tau' + str(ind) + ' (ms)' + '\n')
        file.write('    v (mV)' + '\n')
        file.write('}\n\n')

        file.write('STATE {\n')
        for ind in range(len(self.varinf.flatten())):
            file.write('    var' + str(ind) + '\n')
        file.write('}\n\n')

        file.write('BREAKPOINT {\n')
        file.write('    SOLVE states METHOD cnexp' + '\n')
        calcstring = '    i' + self.ion + ' = g * ('
        ll = 0
        for ii in range(self.statevars.shape[0]):
            for jj in range(self.statevars.shape[1]):
                for kk in range(self.powers[ii,jj]):
                    calcstring += ' var' + str(ll) + ' *'
                ll += 1
            calcstring += str(self.factors[ii])
            if ii < self.statevars.shape[0] - 1:
                calcstring += ' + '
        # calcstring += ') * (v - e' + self.ion + ')'
        calcstring += ') * (v - e)'
        file.write(calcstring + '\n')
        file.write('}\n\n')

        concstring = ''
        for ion in self.concentrations:
            concstring += ', ' + ion + 'i'
        file.write('INITIAL {\n')
        file.write('    rates(v' + concstring + ')' + '\n')
        for ind in range(len(self.varinf.flatten())):
            file.write('    var' + str(ind) + ' = var' + str(ind) + 'inf' + '\n')
        file.write('}\n\n')

        file.write('DERIVATIVE states {\n')
        file.write('    rates(v' + concstring + ')' + '\n')
        for ind in range(len(self.varinf.flatten())):
            file.write('    var' + str(ind) + '\' = (var' + str(ind) \
                        + 'inf - var' + str(ind) + ') / tau' + str(ind) + '\n')
        file.write('}\n\n')

        concstring = ''
        for ion in self.concentrations:
            concstring += ', ' + ion
        file.write('PROCEDURE rates(v' + concstring + ') {\n')
        for ind, varinf in enumerate(self.varinf.flatten()):
            file.write('    var' + str(ind) + 'inf = ' \
                            + sp.printing.ccode(varinf) + '\n')
            file.write('    tau' + str(ind) + ' = ' \
                            + sp.printing.ccode(self.tauinf.flatten()[ind]) + '\n')
        file.write('}\n\n')

        file.close()


    # def computeLin(self, v):
    #     '''
    #     computes coefficients for linear simulation
    #     '''
    #     # coefficients for computing current
    #     fun = self.fun #statevars**self.powers
    #     coeff = np.zeros(self.statevars.shape, dtype=object)
    #     # differentiate
    #     for ind, var in np.ndenumerate(self.statevars):
    #         coeff[ind] = sp.diff(fun, var,1)
    #     # substitute
    #     for ind, var in np.ndenumerate(self.statevars):
    #         fun = fun.subs(var, self.varinf[ind])
    #         for ind2, coe in np.ndenumerate(coeff):
    #             coeff[ind2] = coe.subs(var, self.varinf[ind])
    #     fun = fun.subs(self.spV, self.V0)
    #     for ind, coe in np.ndenumerate(coeff):
    #         coeff[ind] = coe.subs(self.spV, self.V0)
    #     self.coeff_curr = [np.float64(fun), coeff.astype(float)]

    #     # coefficients for state variable equations
    #     dfdv = np.zeros(self.statevar.shape, dtype=object)
    #     dfdx = np.zeros(self.statevar.shape, dtype=object)
    #     # differentiate
    #     for ind, var in np.ndenumerate(self.statevars):
    #         dfdv[ind] = sp.diff(self.fstatevar[ind], self.spV, 1)
    #         dfdx[ind] = sp.diff(self.fstatevar[ind], var, 1)
    #     # substitute state variables by their functions
    #     for ind, var in np.ndenumerate(self.statevars):
    #         dfdv[ind] = dfdv[ind].subs(var, self.varinf[ind])
    #     # substitute voltage by its value
    #     for ind, var in np.ndenumerate(self.statevars):
    #         dfdv[ind] = dfdv[ind].subs(self.spV, self.V0)
    #         dfdx[ind] = dfdx[ind].subs(self.spV, self.V0)

    #     self.coeff_statevar = [dfdv.astype(float), dfdx.astype(float)]

    # def write_to_py_file(self):
    #     file = open('pychannels.py', 'a')
    #     file.write('\n\n')
    #     # append the new class
    #     file.write('class ' + self.__class__.__name__ + 'Sim(SimChannel):\n')
    #     # write the initialization function
    #     file.write('    def __init__(self, inloc_inds, Ninloc, es_eq, ' \
    #                     + 'g_max, e_rev, ' \
    #                     + 'flag=0, mode=1):\n')
    #     # write the specific attributes of the class
    #     power_string = '        self.powers = np.array(['
    #     for powers_row in self.powers:
    #         power_string += '['
    #         for power in powers_row:
    #             power_string += str(power) + ', '
    #         power_string += '], '
    #     power_string += '])\n'
    #     file.write(power_string)
    #     factor_string = '        self.factors = np.array(['
    #     for factor in self.factors:
    #         factor_string += str(factor) + ', '
    #     factor_string += '])\n'
    #     file.write(factor_string)
    #     # write call to base class constructor
    #     file.write('        super(' + self.__class__.__name__ + 'Sim, self)' \
    #                     + '.__init__(self, inloc_inds, Ninloc, es_eq, ' \
    #                     + 'g_max, e_rev, ' \
    #                     + 'flag=flag, mode=mode)\n\n')
    #     # write the functions for the asymptotic values of the state variables
    #     file.write('    def svinf(self, V):\n')
    #     file.write('        V = V[self.inloc_inds] if self.mode == 1 ' \
    #                                                   + 'else V\n')
    #     file.write('        sv_inf = np.zeros((%d, %d, '%self.varinf.shape \
    #                                            + 'self.Nelem))\n')
    #     for ind, var in np.ndenumerate(self.varinf):
    #         try:
    #             if self.spV in var.atoms():
    #                 file.write('        sv_inf[%d,%d,:] = '%ind
    #                                     + _insert_function_prefixes(str(var)) \
    #                                     + '\n')
    #             else:
    #                 file.write('        sv_inf[%d,%d,:] = '%ind
    #                                                        + str(float(var)) \
    #                                                        + '\n')
    #         except AttributeError:
    #             file.write('        sv_inf[%d,%d,:] = '%ind
    #                                                    + str(float(var)) \
    #                                                    + '\n')

    #     file.write('        return sv_inf \n\n')
    #     # write the functions to evaluate relaxation times
    #     file.write('    def tauinf(self, V):\n')
    #     file.write('        V = V[self.inloc_inds] if self.mode == 1 ' \
    #                                                   + 'else V\n')
    #     file.write('        sv_inf = np.zeros((%d, %d, '%self.varinf.shape \
    #                                            + 'self.Nelem))\n')
    #     for ind, tau in np.ndenumerate(self.tauinf):
    #         try:
    #             if self.spV in tau.atoms():
    #                 file.write('        tau_inf[%d,%d,:] = '%ind \
    #                                     + _insert_function_prefixes(str(tau)) \
    #                                     + '\n')
    #             else:
    #                 file.write('        tau_inf[%d,%d,:] = '%ind \
    #                                                         + str(float(tau)) \
    #                                                         + '\n')
    #         except AttributeError:
    #             file.write('        tau_inf[%d,%d,:] = '%ind
    #                                                     + str(float(tau)) \
    #                                                     + '\n')

    #     file.write('        return tau_inf \n\n')
    #     file.close()




# class SimChannel(object):
#     def __init__(self, inloc_inds, Ninloc, es_eq, conc_eq,
#                         g_max, e_rev,
#                         powers,
#                         flag=0, mode=1):
#         '''
#         Creates a vectorized simulation object and accepts a vector of voltages.

#         Let N be the number of state variables.

#         Parameters
#         ----------
#         inloc_inds : numpy.array of ints
#             indices of locations where ionchannel has to be simulated
#         Ninloc : int
#             the total number of input locations
#         es_eq : float or numpy.array of floats
#             The equilibrium potential. As float, signifies that the
#             equilibirum potential is the same everywhere. As numpy.array
#             (number of elements equal to `Ninloc`), signifies the
#             equilibrium as each location
#         g_max : numpy.array of floats
#             The maximal conductance of the ion channel at each location
#         e_rev : numpy.array of floats
#             The reversal potential of the ion channel at each location
#         flag : {0, 1}, optional
#             Mode of simulation. `0` simulates the full current, `1` the
#             non-passive current. Defaults to 1
#         mode : {0, 1}, optional
#             If 0, simulates the channel at all locations. If 1, only
#             simulates at the locations indicated in `inloc_inds`
#         '''
#         # integration mode
#         self.flag = flag
#         self.mode = mode
#         # inloc info
#         self.Ninloc = Ninloc
#         self.inloc_inds = inloc_inds
#         if mode == 1:
#             self.Nelem = len(inloc_inds)
#             self.elem_inds = copy.copy(self.inloc_inds)
#         else:
#             self.Nelem = Ninloc
#             self.elem_inds = np.arange(self.Ninloc)
#         # equilibirum potentials
#         if type(es_eq) == float:
#             self.es_eq = es_eq * np.ones(self.Ninloc)
#         else:
#             self.es_eq = es_eq
#         # maximal conductance and reversal
#         if mode == 1:
#             self.g_max = g_max[inloc_inds]
#             self.e_rev = e_rev[inloc_inds]
#         else:
#             self.g_max = g_max
#             self.e_rev = e_rev
#         # state variables array (initialized to equilibirum)
#         self.sv = self.svinf(es_eq)
#         # set equilibirum state variable values
#         self.sv_eq = copy.deepcopy(self.sv)
#         self.tau_eq = self.tauinf(self.es_eq[self.elem_inds])
#         # equilibirum open probability
#         self.p_open_eq = self.get_p_open(self.sv_eq)

#     def reset(self):
#         self.sv = self.sveq

#     def get_p_open(self, sv=None):
#         if sv == None: sv = self.sv
#         self._p_open = np.sum(self.factors[:,np.newaxis] * \
#                               np.product(sv**self.powers[:,:,np.newaxis],
#                                          1),
#                               0)
#         return self._p_open

#     def set_p_open(self, illegal):
#         raise AttributeError("`popen` is a read-only attribute")

#     p_open = property(get_p_open, set_p_open)

#     def advance(self, dt, V):
#         '''
#         Advance the ion channels internal variables one timestep

#         Parameters
#         ----------
#             dt : float
#                 the timestep
#             V : numpy.array of floats
#                 Voltage at each location
#         '''
#         svinf = self.svinf(V)
#         tauinf = self.tauinf(V)
#         prop1 = np.exp(-dt/tauinf)
#         # advance the variables
#         self.sv *= prop1
#         self.sv += (1. - prop1) * svinf

#     def get_current_general(self, V, I_out=None):
#         '''
#         Get the channel current given the voltage, according to integration
#         paradigm

#         Parameters
#         ----------
#         V : numpy.array of floats
#             Location voltage (length should be equal to `self.Ninloc`)
#         I_out : {numpy.array of floats, None}, optional
#             Array to store the output current. Defaults to None, in which case
#             a new array is created.

#         Returns
#         -------
#         numpy.array of floats
#             The channel current at each location
#         '''
#         if self.flag == 1:
#             return self.get_current_np(V, I_out=I_out)
#         else:
#             return self.get_current(V, I_out=I_out)

#     def get_current(self, V, I_out=None):
#         '''
#         Get the full channel current given the voltage.

#         Parameters
#         ----------
#         V : numpy.array of floats
#             Location voltage (length should be equal to `self.Ninloc`)
#         I_out : {numpy.array of floats, None}, optional
#             Array to store the output current. Defaults to None, in which case
#             a new array is created.

#         Returns
#         -------
#         numpy.array of floats
#             The channel current at each location
#         '''
#         if I_out == None: I_out = np.zeros(self.Ninloc)
#         if self.mode == 1:
#             I_out[self.inloc_inds] -= self.g * self.p_open \
#                                       * (V[self.inloc_inds] - self.e)
#         else:
#             I_out -= self.g * self.popen \
#                      * (V - self.e)
#         return I_out

#     def get_current_np(self, V, I_out=None):
#         '''
#         Get the non-passive channel current given the voltage.

#         Parameters
#         ----------
#         V : numpy.array of floats
#             Location voltage (length should be equal to `self.Ninloc`)
#         I_out : {numpy.array of floats, None}, optional
#             Array to store the output current. Defaults to None, in which case
#             a new array is created.

#         Returns
#         -------
#         numpy.array of floats
#             The channel current at each location
#         '''
#         if I_out == None: I_out = np.zeros(self.Ninloc)
#         if self.mode == 1:
#             I_out[self.inloc_inds] -= self.g_max \
#                                       * (self.p_open - self.p_open_eq) \
#                                       * (V[self.inloc_inds] - self.e_rev)
#         else:
#             I_out -= self.g_max \
#                      * (self.p_open - self.p_open_eq) \
#                      * (V - self.e_rev)
#         return I_out

#     def get_conductance_general(self, G_out=None, I_out=None):
#         '''
#         Let the channel current be :math:`-g (V-e)`. Returns :math:`-g` and
#         :math:`-g (E_eq-e)`. Returns the component according to integration
#         paradigm.

#         Parameters
#         ----------
#         V : numpy.array of floats
#             Location voltage (length should be equal to `self.Ninloc`)
#         G_out : {numpy.array of floats, None}, optional
#             Array to store the output :math:`-g`. Defaults to None, in
#             which case a new array is created.
#         I_out : {numpy.array of floats, None}, optional
#             Array to store the output :math:`-g (E_eq-e)`. Defaults to None, in
#             which case a new array is created.

#         Returns
#         -------
#         (numpy.array of floats, numpy.array of floats)
#             :math:`-g` at each location and :math:`-g (E_eq-e)` at each location
#         '''
#         if self.flag == 1:
#             return self.get_conductance_np(G_out=G_out, I_out=I_out)
#         else:
#             return self.get_conductance(G_out=G_out, I_out=I_out)

#     def get_conductance(self, G_out=None, I_out=None):
#         '''
#         Let the channel current be :math:`-g (V-e)`. Returns :math:`-g` and
#         :math:`-g (E_eq-e)`. Returns the full component

#         Parameters
#         ----------
#         V : numpy.array of floats
#             Location voltage (length should be equal to `self.Ninloc`)
#         G_out : {numpy.array of floats, None}, optional
#             Array to store the output :math:`-g`. Defaults to None, in
#             which case a new array is created.
#         I_out : {numpy.array of floats, None}, optional
#             Array to store the output :math:`-g (E_eq-e)`. Defaults to None, in
#             which case a new array is created.

#         Returns
#         -------
#         (numpy.array of floats, numpy.array of floats)
#             :math:`-g` at each location and :math:`-g (E_eq-e)` at each location
#         '''
#         if G_out == None: G_out = np.zeros(self.Ninloc)
#         if I_out == None: I_out = np.zeros(self.Ninloc)
#         p_open = self.p_open
#         if self.mode == 1:
#             G_out[self.inloc_inds] -= self.g_max * p_open
#             I_out[self.inloc_inds] -= self.g_max * p_open \
#                                       * (self.es_eq - self.e_rev)
#         else:
#             G_out -= self.g_max * p_open
#             I_out -= self.g_max * p_open \
#                      * (self.es_eq - self.e_rev)
#         return G_out, I_out

#     def get_conductance_np(self, G_out=None, I_out=None):
#         '''
#         Let the channel current be :math:`-g (V-e)`. Returns :math:`-g` and
#         :math:`-g (E_eq-e)`. Returns the non-passive component

#         Parameters
#         ----------
#         V : numpy.array of floats
#             Location voltage (length should be equal to `self.Ninloc`)
#         G_out : {numpy.array of floats, None}, optional
#             Array to store the output :math:`-g`. Defaults to None, in
#             which case a new array is created.
#         I_out : {numpy.array of floats, None}, optional
#             Array to store the output :math:`-g (E_eq-e)`. Defaults to None, in
#             which case a new array is created.

#         Returns
#         -------
#         (numpy.array of floats, numpy.array of floats)
#             :math:`-g` at each location and :math:`-g (E_eq-e)` at each location
#         '''
#         if G_out == None: G_out = np.zeros(self.Ninloc)
#         if I_out == None: I_out = np.zeros(self.Ninloc)
#         p_open = self.p_open - self.p_open_eq
#         if self.mode == 1:
#             G_out[self.inloc_inds] -= self.g_max * p_open
#             I_out[self.inloc_inds] -= self.g_max * p_open \
#                                       * (self.es_eq - self.e_rev)
#         else:
#             G_out -= self.g_max * p_open
#             I_out -= self.g_max * p_open \
#                      * (self.es_eq - self.e_rev)
#         return G_out, I_out
