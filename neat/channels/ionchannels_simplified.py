import sympy as sp
import numpy as np
import scipy.optimize as so

import os
import copy

CONC_DICT = {'na': 10., # mM
             'k': 54.4, # mM
             'ca': 1e-4, # 1e-4
            }

E_ION_DICT = {'na': 50.,
             'k': -85.,
             'ca': 50.,
            }


class _func(object):
    def __init__(self, eval_func_aux, eval_func_vtrap, e_trap):
        self.eval_func_aux = eval_func_aux
        self.eval_func_vtrap = eval_func_vtrap
        self.e_trap = e_trap

    def __call__(self, *args):
        vv = args[0]
        if isinstance(vv, float):
            if np.abs(vv - self.e_trap) < 0.001:
                return self.eval_func_vtrap(*args)
            else:
                return self.eval_func_aux(*args)
        else:
            fv_return = np.zeros_like(vv)
            bool_vtrap = np.abs(vv - self.e_trap) < 0.0001
            inds_vtrap = np.where(bool_vtrap)
            args_ = [a[inds_vtrap] for a in args]
            fv_return[inds_vtrap] = self.eval_func_vtrap(*args_)
            inds = np.where(np.logical_not(bool_vtrap))
            args_ = [a[inds] for a in args]
            fv_return[inds] = self.eval_func_aux(*args_)
            return fv_return


def _insert_function_prefixes(string, prefix='np',
                              functions=['exp', 'sin', 'cos', 'tan', 'pi']):
    """
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
    """
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


class IonChannelSimplified(object):
    """


            I.e. if the channel defines state
            variables 'm' and 'h', call `computePOpen(v_inp, m=m_inp, h=h_inp),
            with `m_inp` and `h_inp` being of the same type and shape as `v_inp`

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
    """

    def __init__(self):
        """
        Will give an ``AttributeError`` if initialized as is. Should only be
        initialized from its' derived classes that implement specific ion
        channel types.
        """
        # define the channel based on user specified state variables and activations
        self.define()
        if not hasattr(self, 'ion'):
            self.ion = ''
        if not hasattr(self, 'concentrations'):
            self.concentrations = []
        self.sp_c = [sp.symbols(conc) for conc in self.concentrations]
        if not hasattr(self, 'qtemp'):
            self.qtemp = 1.

        # extract the state variables
        self.statevars = list(self.p_open.free_symbols)

        if 'alpha' in self.__dict__ and 'beta' in self.__dict__:
            # convert to internal representation
            self.varinf, self.tauinf = {}, {}
            for var in self.statevars:
                self.varinf[var] = self.alpha[var] / (self.alpha[var]  + self.beta[var])
                self.tauinf[var] = (1./self.qtemp) / (self.alpha[var]  + self.beta[var])

        # set the voltage state variable
        self.sp_v = sp.symbols('v')
        # set the right hand side of the differential equation for
        # state variables
        self.fstatevar = {}
        for statevar in self.statevars:
            self.fstatevar[statevar] = (-statevar + self.varinf[statevar]) / self.tauinf[statevar]

        # set the lambda functions for efficient numpy evaluation
        self.setLambdaFuncs()

    def __getstate__(self):
        """
        remove lambdified functions from dict as they can not be pickled
        """
        d = dict(self.__dict__)

        del d['f_statevar']
        del d['f_varinf']
        del d['f_tauinf']
        del d['f_p_open']
        del d['dp_dx'], d['df_dv'], d['df_dx'], d['df_dc']

        return d

    def __setstate__(self, s):
        """
        since lambdified functions were not pickled we need to restore them
        """
        self.__dict__ = s
        self.setLambdaFuncs()

    def setLambdaFuncs(self):
        # construct lambda function for state variables
        self.f_statevar = self.lambdifyFStatevar()
        # construct lambda functions for steady state activation
        self.f_varinf = self.lambdifyVarInf()
        # construct lambda functions for state variable time scales
        self.f_tauinf = self.lambdifyTauInf()
        # construct lambda function for passive opening
        self.f_p_open = self.lambdifyPOpen()
        # construct lambda function for linear current coefficient evaluations
        self.dp_dx, self.df_dv, self.df_dx, self.df_dc = \
                        self.lambdifyDerivatives()

    def _substituteConc(self, expr):
        for sp_c, ion in zip(self.sp_c, self.concentrations):
            expr = expr.subs(sp_c, CONC_DICT[ion])
        return expr

    def lambdifyVarInf(self):
        f_varinf = {}
        for statevar in self.statevars:
            varinf = self._substituteConc(self.varinf[statevar])
            f_varinf[statevar] = sp.lambdify(self.sp_v, varinf)
        return f_varinf

    def lambdifyTauInf(self):
        f_tauinf = {}
        for statevar in self.statevars:
            tauinf = self._substituteConc(self.tauinf[statevar])
            f_tauinf[statevar] = sp.lambdify(self.sp_v, tauinf)
        return f_tauinf

    def lambdifyPOpen(self):
        # arguments for lambda function
        args = [self.sp_v] + [statevar for statevar in self.statevars]
        # return lambda function
        return sp.lambdify(args, self.p_open)

    def lambdifyFStatevar(self):
        # arguments for lambda function
        args = [self.sp_v] + [statevar for statevar in self.statevars]
        # construct dict of lambda functions for each statevar
        f_statevar = {}
        for statevar in self.statevars:
            f_statevar[statevar] = sp.lambdify(args, self.fstatevar[statevar])
        return f_statevar

    def lambdifyDerivatives(self):
        # arguments for lambda function
        args = [self.sp_v] + [statevar for statevar in self.statevars]
        # compute open probability derivatives to state vars
        dp_dx_aux = np.zeros(len(self.statevars), dtype=object)
        for ind, var in np.ndenumerate(self.statevars):
            dp_dx_aux[ind] = sp.lambdify(args,
                                     sp.diff(self.p_open, var, 1))
        # compute state variable derivatives
        df_dv_aux = np.zeros(len(self.statevars), dtype=object)
        df_dx_aux = np.zeros(len(self.statevars), dtype=object)
        df_dc_aux = [np.zeros(len(self.statevars), dtype=object) for _ in self.sp_c]
        # differentiate
        for ind, var in np.ndenumerate(self.statevars):
            f_sv = self.fstatevar[var]
            # derivatives to concentrations
            for ii, sp_c in enumerate(self.sp_c):
                df_dc_aux[ii][ind] = sp.lambdify(args,
                                        self._substituteConc(sp.diff(f_sv, sp_c, 1)))
            # derivative to voltage and state variable
            f_sv = self._substituteConc(f_sv)
            df_dv_aux[ind] = sp.lambdify(args,
                                     sp.diff(f_sv, self.sp_v, 1))
            df_dx_aux[ind] = sp.lambdify(args,
                                     sp.diff(f_sv, var, 1))

        # define convenient functions
        def dp_dx(*args):
            dp_dx_list = [[] for _ in range(len(self.statevars))]
            for ind, dp_dx_ in np.ndenumerate(dp_dx_aux):
                dp_dx_list[ind[0]].append(dp_dx_aux[ind](*args))
            return np.array(dp_dx_list)
        def df_dv(*args):
            df_dv_list = [[] for _ in range(len(self.statevars))]
            for ind, df_dv_ in np.ndenumerate(df_dv_aux):
                df_dv_list[ind[0]].append(df_dv_aux[ind](*args))
            return np.array(df_dv_list)
        def df_dx(*args):
            df_dx_list = [[] for _ in range(len(self.statevars))]
            for ind, df_dx_ in np.ndenumerate(df_dx_aux):
                df_dx_list[ind[0]].append(df_dx_aux[ind](*args))
            return np.array(df_dx_list)
        def df_dc(*args):
            df_dc_list = []
            for ic, (sp_c, df_dc__) in enumerate(zip(self.sp_c, df_dc_aux)):
                df_dc_list.append([[] for _ in range(len(self.statevars))])
                for ind, df_dc_ in np.ndenumerate(df_dc__):
                    df_dc_list[-1][ind[0]].append(df_dc__[ind](*args))
            return np.array(df_dc_list)

        return dp_dx, df_dv, df_dx, df_dc

    def expansionPointAsString(self, v, statevars=None):
        if statevars is None:
            statevars = np.zeros(self.statevars.shape, dtype=float)
            for ind, f_varinf in np.ndenumerate(self.f_varinf):
                statevars[ind] = f_varinf(v)
        rstring = 'v = %.2f'%(v) + ', sv --> '
        for ind, sv in np.ndenumerate(statevars):
            sv_name = self.varnames[ind]
            rstring += sv_name + ' = %.6f, '%(sv)
        p_open = self.computePOpen(v, statevars=statevars)
        rstring += 'p_open = %.4f'%(p_open)
        return rstringp

    def _argsAsList(self, v, **kwargs):
        """
        Converts arguments to list for lambdified functions
        """
        arg_list = [v]

        for var in self.statevars:
            try:
                arg_list.append(kwargs[var])
            except KeyError:
                # state variable is not in kwargs
                # set default value based on voltage
                arg_list.append(self.f_varinf[var](v))

        return arg_list

    def computePOpen(self, v, **kwargs):
        """
        Compute the open probability of the ion channel

        Parameters
        ----------
        v: float or `np.ndarray` of float
            The voltage at which to evaluate the open probability
        **kwargs
            Optional Values of the state variables.

        Returns
        -------
        float or `np.ndarray` of float
            The open probability
        """
        args = self._argsAsList(v, **kwargs)
        return self.f_p_open(*args)

    def computeDerivatives(self, v, **kwargs):
        """
        Compute:
        (i) the derivatives of the open probability to the state variables
        (ii) The derivatives of state functions to the voltage
        (iii) The derivatives of state functions to the state variables

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage at which to evaluate the open probability
        **kwargs
            Optional Values of the state variables.

        Returns
        -------
        tuple of three floats or three `np.ndarray`s of float
            The derivatives
        """
        args = self._argsAsList(v, **kwargs)
        return self.dp_dx(*args), self.df_dv(*args), self.df_dx(*args)

    def computeDerivativesConc(self, v, **kwargs):
        args = self._argsAsList(v, **kwargs)
        return self.df_dc(*args)

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

    def computeLinear(self, v, freqs, **kwargs):
        dp_dx_arr, df_dv_arr, df_dx_arr = self.computeDerivatives(v, **kwargs)
        lin_f = np.zeros_like(freqs)
        for ind, dp_dx_ in np.ndenumerate(dp_dx_arr):
            df_dv_ = df_dv_arr[ind] * 1e3 # convert to 1 / s
            df_dx_ = df_dx_arr[ind] * 1e3 # convert to 1 / s
            # add to the impedance contribution
            lin_f += dp_dx_ * df_dv_ / (freqs - df_dx_)
        return lin_f

    def computeLinearConc(self, v, freqs, ion, **kwargs):
        ind_c = self.concentrations.index(ion)
        dp_dx_arr, df_dv_arr, df_dx_arr = self.computeDerivatives(v, **kwargs)
        df_dc = self.computeDerivativesConc(v, **kwargs)
        lin_f = np.zeros_like(freqs)
        for ind, dp_dx_ in np.ndenumerate(dp_dx_arr):
            df_dc_ = df_dc[ind_c][ind] * 1e3 # convert to 1 / s
            df_dx_ = df_dx_arr[ind] * 1e3 # convert to 1 / s
            # add to the impedance contribution
            lin_f += dp_dx_ * df_dc_ / (freqs - df_dx_)
        return lin_f

    def computeLinSum(self, v, freqs, e_rev, **kwargs):
        return (e_rev - v) * self.computeLinear(v, freqs, **kwargs) - \
               self.computePOpen(v, **kwargs)

    def computeLinConc(self, v, freqs, e_rev, ion, **kwargs):\
        return (e_rev - v) * self.computeLinearConc(v, freqs, ion, **kwargs)


    def writeModFile(self, path, g=0., e=0.):
        """
        Writes a modfile of the ion channel for simulations with neuron
        """
        modname = 'I' + self.__class__.__name__ + '.mod'
        fname = os.path.join(path, modname)

        file = open(fname, 'w')

        file.write(': This mod file is automaticaly generated by the ' +
                    '``neat.channels.ionchannels`` module\n\n')

        file.write('NEURON {\n')
        cname =  self.__class__.__name__
        file.write('    SUFFIX I' + cname + '\n')
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

        return modname

    def writeCPPCode(self, path, e_rev):
        """
        Warning: concentration dependent ion channels get constant concentrations
        substituted for c++ simulation
        """

        fcc = open(os.path.join(path, 'Ionchannels.cc'), 'a')
        fh = open(os.path.join(path, 'Ionchannels.h'), 'a')
        # fstruct = open('cython_code/channelstruct.h', 'a')


        fh.write('class ' + self.__class__.__name__ + ': public IonChannel{' + '\n')
        fh.write('private:' + '\n')
        # fh.write('    double m_g_bar = 0.0, m_e_rev = %.8f;\n'%e_rev)
        for ind, varname in np.ndenumerate(self.statevars):
                fh.write('    double m_' + sp.printing.ccode(varname) +';\n')
        for ind, varname in np.ndenumerate(self.statevars):
                fh.write('    double m_' + sp.printing.ccode(varname) + '_inf, m_tau_' + sp.printing.ccode(varname) + ';\n')
        for ind, varname in np.ndenumerate(self.statevars):
                fh.write('    double m_v_' + sp.printing.ccode(varname) + '= 10000.;\n')
        fh.write('    double m_p_open_eq = 0.0, m_p_open = 0.0;\n')
        fh.write('public:' + '\n')
        fh.write('    void calcFunStatevar(double v) override;' + '\n')
        fh.write('    double calcPOpen() override;' + '\n')
        fh.write('    void setPOpen() override;' + '\n')
        fh.write('    void setPOpenEQ(double v) override;' + '\n')
        fh.write('    void advance(double dt) override;' + '\n')
        fh.write('    double getCond() override;' + '\n')
        fh.write('    double getCondNewton() override;' + '\n')
        fh.write('    double f(double v) override;' + '\n')
        fh.write('    double DfDv(double v) override;' + '\n')
        fh.write('    void setfNewtonConstant(double* vs, int v_size) override;' + '\n')
        fh.write('    double fNewton(double v) override;' + '\n')
        fh.write('    double DfDvNewton(double v) override;' + '\n')
        fh.write('};' + '\n')

        fcc.write('void ' + self.__class__.__name__ + '::calcFunStatevar(double v){' + '\n')
        for ind, varname in np.ndenumerate(self.statevars):
            tauinf = self.tauinf[varname]
            varinf = self.varinf[varname]
            varinf_ = self._substituteConc(varinf)
            tauinf_ = self._substituteConc(tauinf)
            fcc.write('    m_' + sp.printing.ccode(varname) + '_inf = ' + sp.printing.ccode(varinf_) + ';' + '\n')
            # if self.varinf.shape[1] == 2 and ind == (0,0):
            if str(varname) == 'm':
                # instantaneous approximation possible if statevar is activation (denoted by 'm')
                fcc.write('    if(m_instantaneous)' + '\n')
                fcc.write('        m_tau_' + sp.printing.ccode(varname) + ' = ' + sp.printing.ccode(sp.Float(1e-5))  + ';\n')
                fcc.write('    else' + '\n')
                fcc.write('        m_tau_' + sp.printing.ccode(varname) + ' = ' + sp.printing.ccode(tauinf_) + ';\n')
            else:
                fcc.write('    m_tau_' + sp.printing.ccode(varname) + ' = ' + sp.printing.ccode(tauinf_) + ';\n')
        fcc.write('}' + '\n')

        fcc.write('double ' + self.__class__.__name__ + '::calcPOpen(){' + '\n')
        expr = copy.deepcopy(self.p_open)
        for ind, varname in np.ndenumerate(self.statevars):
            symb = sp.symbols('m_' + sp.printing.ccode(varname))
            expr = expr.subs(varname, symb)
        fcc.write('    return ' + sp.printing.ccode(expr) + ';' + '\n')
        fcc.write('}' + '\n')

        fcc.write('void ' + self.__class__.__name__ + '::setPOpen(){' + '\n')
        fcc.write('    m_p_open = calcPOpen();' + '\n')
        fcc.write('}' + '\n')

        fcc.write('void ' + self.__class__.__name__ + '::setPOpenEQ(double v){' + '\n')
        fcc.write('    calcFunStatevar(v);' + '\n')
        fcc.write('')
        expr = copy.deepcopy(self.p_open)
        for ind, varname in np.ndenumerate(self.statevars):
            symb = sp.symbols('m_' + sp.printing.ccode(varname) + '_inf')
            expr = expr.subs(varname, symb)
            fcc.write('    m_' + sp.printing.ccode(varname) + ' = ' + sp.printing.ccode(symb) + ';\n')
        fcc.write('    m_p_open_eq =' + sp.printing.ccode(expr) + ';' + '\n')
        fcc.write('}' + '\n')

        fcc.write('void ' + self.__class__.__name__ + '::advance(double dt){' + '\n')
        for ind, varname in np.ndenumerate(self.statevars):
            tauinf = self.tauinf[varname]
            varinf = self.varinf[varname]
            varname_ = 'm_' + sp.printing.ccode(varname)
            varname_inf = 'm_' + sp.printing.ccode(varname) + '_inf'
            varname_tau = 'm_tau_' + sp.printing.ccode(varname)
            propname = 'p0_' + sp.printing.ccode(varname)
            # fcc.write('    ' + varname + ' += dt * (' + varname_inf + ' - ' + varname + ') / ' + varname_tau + ';' + '\n')
            fcc.write('    double ' + propname + ' = exp(-dt / ' + varname_tau + ');\n')
            fcc.write('    ' + varname_ + ' *= ' + propname + ' ;\n')
            fcc.write('    ' + varname_ + ' += (1. - ' + propname + ' ) *  ' + varname_inf + ';\n')
        fcc.write('}' + '\n')


        # self.exp_aux = np.exp(-dt/self.tauinf_aux)
        # # advance the variables
        # self.sv *= self.exp_aux
        # self.sv += (1.-self.exp_aux) * self.svinf_aux

        fcc.write('double ' + self.__class__.__name__ + '::getCond(){' + '\n')
        fcc.write('    return m_g_bar * (m_p_open - m_p_open_eq);' + '\n')
        fcc.write('}' + '\n')

        fcc.write('double ' + self.__class__.__name__ + '::getCondNewton(){' + '\n')
        fcc.write('    return m_g_bar;' + '\n')
        fcc.write('}' + '\n')

        # function for temporal integration
        fcc.write('double ' + self.__class__.__name__ + '::f(double v){' + '\n')
        fcc.write('    return (m_e_rev - v);' + '\n')
        fcc.write('}' + '\n')

        fcc.write('double ' + self.__class__.__name__ + '::DfDv(double v){' + '\n')
        fcc.write('    return -1.;' + '\n')
        fcc.write('}' + '\n')

        # set voltage values to evaluate at constant voltage during newton iteration
        fcc.write('void ' + self.__class__.__name__ + '::setfNewtonConstant(double* vs, int v_size){' + '\n')
        fcc.write('    if(v_size != %d)'%len(self.statevars) + '\n')
        fcc.write('        cerr << "input arg [vs] has incorrect size, ' + \
                  'should have same size as number of channel state variables" << endl' + ';\n')
        for ii, var in enumerate(self.statevars):
            fcc.write('    m_v_' + sp.printing.ccode(var) + ' = vs[%d]'%ii + ';\n')
        fcc.write('}' + '\n')

        # functions for solving Newton iteration
        fcc.write('double ' + self.__class__.__name__ + '::fNewton(double v){' + '\n')
        p_o = self.p_open
        for ind, varname in np.ndenumerate(self.statevars):
            v_var = sp.symbols('v_' + str(varname))
            # substitute voltage symbol in the activation
            varinf_ = self._substituteConc(self.varinf[varname]).subs(self.sp_v, v_var)
            # assign dynamic or fixed voltage to the activation
            fcc.write('    double ' + sp.printing.ccode(v_var) + ';\n')
            fcc.write('    if(m_' + sp.printing.ccode(v_var) + ' > 1000.){' + '\n')
            fcc.write('        ' + sp.printing.ccode(v_var) + ' = v' + ';\n')
            fcc.write('    } else{' + '\n')
            fcc.write('        ' + sp.printing.ccode(v_var) + ' = m_' + sp.printing.ccode(v_var) + ';\n')
            fcc.write('    }' + '\n')
            fcc.write('    double ' + sp.printing.ccode(varname) + ' = ' + sp.printing.ccode(varinf_) + ';\n')
            # p_o = p_o.subs(self.statevars[ind], varinf_)

        fcc.write('    return (m_e_rev - v) * (' + sp.printing.ccode(p_o) + ' - m_p_open_eq)' + ';\n')
        fcc.write('}' + '\n')

        fcc.write('double ' + self.__class__.__name__ + '::DfDvNewton(double v){' + '\n')
        p_o = self.p_open
        # compute partial derivatives
        dp_o = np.zeros_like(self.statevars)
        for ind, varname in np.ndenumerate(self.statevars):
            dp_o[ind] = sp.diff(p_o, varname, 1)
        # print derivatives
        for ind, varname in np.ndenumerate(self.statevars):
            v_var = sp.symbols('v_' + str(varname))
            # substitute voltage symbol in the activation
            varinf_ = self._substituteConc(self.varinf[varname]).subs(self.sp_v, v_var)
            dvarinf_dv = sp.diff(varinf_, v_var, 1)
            # compute derivative
            fcc.write('    double ' + sp.printing.ccode(v_var) + ';\n')
            fcc.write('    double d' + sp.printing.ccode(varname) + '_dv;\n')
            fcc.write('    if(m_' + sp.printing.ccode(v_var) + ' > 1000.){' + '\n')
            fcc.write('        ' + sp.printing.ccode(v_var) + ' = v' + ';\n')
            fcc.write('        d' + sp.printing.ccode(varname) + '_dv = ' + sp.printing.ccode(dvarinf_dv) + ';\n')
            fcc.write('    } else{' + '\n')
            fcc.write('        ' + sp.printing.ccode(v_var) + ' = m_' + sp.printing.ccode(v_var) + ';\n')
            fcc.write('        d' + sp.printing.ccode(varname) + '_dv = 0;\n')
            fcc.write('    }' + '\n')
            fcc.write('    double ' + sp.printing.ccode(varname) + ' = ' + sp.printing.ccode(varinf_) + ';\n')


        expr_str = '+'.join([sp.printing.ccode(dp_o_) + ' * d' + sp.printing.ccode(varname) + '_dv' for dp_o_, varname in zip(dp_o, self.statevars)])

        fcc.write('    return -1. * (' + sp.printing.ccode(p_o) + ' - m_p_open_eq) + (' + expr_str + ') * (m_e_rev - v)' + ';\n')
        fcc.write('}' + '\n')




        fh.write('\n')
        fcc.write('\n')
        # fstruct.write('    ' + self.__class__.__name__ + ' ' + self.__class__.__name__ + '_;' + '\n')

        fh.close()
        fcc.close()
        # fstruct.close()

