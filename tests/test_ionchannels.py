import numpy as np
import sympy as sp
# import sympy as cse

from neat.channels.channelcollection import channelcollection
from neat import IonChannel
from neat import IonChannelSimplified

import pytest
import pickle


class TestChannels():
    def testBasic(self):
        tcn = channelcollection.TestChannel()
        v_arr = np.linspace(-80., -10., 10)

        # state variable asymptotic values
        def varinf(v):
            aux = np.ones_like(v) if isinstance(v, np.ndarray) else 1.
            return np.array([[1. / (1. + np.exp((v - 30.) / 100.)), 1. / (1. + np.exp((-v + 30.) / 100.)), -10. * aux],
                             [2. / (1. + np.exp((v - 30.) / 100.)), 2. / (1. + np.exp((-v + 30.) / 100.)), -30. * aux]])

        # state variable functions
        def dvarinf_dv(v):
            aux = np.ones_like(v) if isinstance(v, np.ndarray) else 1.
            vi_aux = varinf(v)
            return np.array([[-vi_aux[0, 0, :] * (1 - vi_aux[0, 0, :]) / 100., vi_aux[0, 1, :] * (1 - vi_aux[0, 1, :]) / 100., 0. * aux],
                             [-vi_aux[1, 0, :] * (1 - vi_aux[1, 0, :] / 2.) / 100., vi_aux[1, 1, :] * (1 - vi_aux[1, 1, :] / 2.) / 100., 0. * aux]])

        # state variable relaxation time scale
        def taurel(v):
            aux = np.ones_like(v) if isinstance(v, np.ndarray) else 1.
            return np.array([[1. * aux, 2. * aux, 1. * aux],
                             [2. * aux, 2. * aux, 3. * aux]])

        # test whether activations are correct
        var_inf = varinf(v_arr)
        for ind, f_vi in np.ndenumerate(tcn.f_varinf):
            var_inf_ind = f_vi(v_arr)
            assert np.allclose(var_inf_ind, var_inf[ind])
        # test whether open probability is correct
        p_open = np.sum(tcn.factors[:, np.newaxis] * \
                        np.product(var_inf ** tcn.powers[:, :, np.newaxis], 1), 0)
        p_open_ = tcn.computePOpen(v_arr)
        assert np.allclose(p_open_, p_open)
        # test whether derivatives are correct
        dp_dx_arr, df_dv_arr, df_dx_arr = tcn.computeDerivatives(v_arr)
        # first: derivatives of open probability
        for ind, name in np.ndenumerate(tcn.varnames):
            dp_dx = tcn.factors[ind[0]] * tcn.powers[ind] * \
                    np.prod(var_inf[ind[0]] ** tcn.powers[ind[0]][:, np.newaxis], 0) / var_inf[ind]
            assert np.allclose(dp_dx_arr[ind], dp_dx)
        # second: derivatives of state variable functions to voltage
        df_dv = dvarinf_dv(v_arr) / taurel(v_arr)
        for ind, name in np.ndenumerate(tcn.varnames):
            assert np.allclose(df_dv[ind], df_dv_arr[ind])
        # third: derivatives of state variable functions to state variables
        df_dx = -1. / taurel(v_arr)
        for ind, name in np.ndenumerate(tcn.varnames):
            assert np.allclose(df_dx[ind], df_dx_arr[ind])
        # test whether sympy expressions are correct
        # TODO

class TestNa(IonChannel):
    def __init__(self):
        ## USER DEFINED
        # ion the ion channel current uses (don't define for unspecific)
        ion = 'na'
        # define open probability
        p_open = sp.sympify('m**3 * h')
        # define state variable activations and timescales
        spalpham = sp.sympify('   0.182 * (v + 38.) / (1. - exp(-(v + 38.) / 6.))', evaluate=False) # 1/ms
        spbetam  = sp.sympify(' - 0.124 * (v + 38.) / (1. - exp( (v + 38.) / 6.))', evaluate=False) # 1/ms
        spalphah = sp.sympify(' - 0.015 * (v + 66.) / (1. - exp( (v + 66.) / 6.))', evaluate=False) # 1/ms
        spbetah  = sp.sympify('   0.015 * (v + 66.) / (1. - exp(-(v + 66.) / 6.))', evaluate=False) # 1/ms
        m_inf = spalpham / (spalpham + spbetam)
        h_inf = spalphah / (spalphah + spbetah)
        tau_m = (1./2.95) / (spalpham + spbetam)
        tau_h = (1./2.95) / (spalphah + spbetah)


        ## INTERNAL
        # this array fixes the ordering of symbols
        # --> PROBLEM: the order of the symbols is not fixed, hence this might
        # cause confusion whenever a lambdified function is called
        self.statevars = np.array([symbol for symbol in p_open.free_symbols])
        # redundant, but current IonChannel implementation uses it
        self.varnames = np.array([str(sv) for sv in self.statevars])

        # define symbols
        for symbol in p_open.free_symbols:
            exec(str(symbol) + ' = symbol')

        # store open probability as attribute
        self.p_open = p_open

        # store v sympy symbol
        self.sp_v = [symb for symb in m_inf.free_symbols][0]
        print('---', self.sp_v)

        # define state variable function arrays
        # we have to maintain the same ordering of statevariables to be able to
        # keep track of lambda function arguments
        self.fstatevar = np.zeros_like(self.statevars)
        self.varinf = np.zeros_like(self.statevars)
        self.tauinf = np.zeros_like(self.statevars)
        for ind, var in np.ndenumerate(self.statevars):
            self.varinf[ind] = eval(str(var)+'_inf')
            self.tauinf[ind] = eval('tau_' +str(var))
            self.fstatevar[ind] = eval('('+str(var)+'_inf - '+ str(var) +') / tau_'+str(var))

        # concentrations, for if the ion channel uses it
        if not hasattr(self, 'ion'):
            self.ion = ''
        if not hasattr(self, 'concentrations'):
            self.concentrations = []
        self.sp_c = [sp.symbols(conc) for conc in self.concentrations]

        # set lambda functions
        self.setLambdaFuncs()

    def testPOpen(self):
        print('---')
        na = channelcollection.Na_Ta()

        p_o_1 = na.computePOpen(-35.)
        p_o_2 = self.computePOpen(-35.)
        assert np.allclose(p_o_1, p_o_2)

    def testLinSum(self):
        na = channelcollection.Na_Ta()

        l_s_1 = na.computeLinSum(-35., 0., 50.)
        l_s_2 = self.computeLinSum(-35., 0., 50.)
        assert np.allclose(l_s_1, l_s_2)


def test_na():
    tna = TestNa()
    tna.testPOpen()
    tna.testLinSum()




def sp_exp(x):
    return sp.exp(x, evaluate=False)


class Na_Ta_new(IonChannelSimplified):
    def define(self):
        """
        (Colbert and Pan, 2002)

        Used in (Hay, 2011)
        """
        self.ion = 'na'
        self.concentrations = []
        # define symbols used in the equations
        v, m, h = sp.symbols('v, m, h')
        # define channel open probability
        self.p_open = h * m ** 3
        # define activation functions
        self.alpha, self.beta = {}, {}
        self.alpha[m] = '0.182 * (v + 38.) / (1. - exp(-(v + 38.) / 6.))' # 1/ms
        self.beta[m] = '-0.124 * (v + 38.) / (1. - exp( (v + 38.) / 6.))' # 1/ms
        self.alpha[h] = '-0.015 * (v + 66.) / (1. - exp( (v + 66.) / 6.))' # 1/ms
        self.beta[h] =  '0.015 * (v + 66.) / (1. - exp(-(v + 66.) / 6.))' # 1/ms
        # self.alpha, self.beta = {}, {}
        # self.alpha[m] =  0.182 * (v + 38.) / (1. - sp_exp(-(v + 38.) / 6.)) # 1/ms
        # self.beta[m] = -0.124 * (v + 38.) / (1. - sp_exp( (v + 38.) / 6.)) # 1/ms
        # self.alpha[h] = -0.015 * (v + 66.) / (1. - sp_exp( (v + 66.) / 6.)) # 1/ms
        # self.beta[h] =  0.015 * (v + 66.) / (1. - sp_exp(-(v + 66.) / 6.)) # 1/ms
        # temperature factor for time-scales
        self.qtemp = 2.95

        for var, expr in self.alpha.items():
            self.alpha[var] = sp.sympify(expr, evaluate=False)
            self.beta[var] = sp.sympify(expr, evaluate=False)

        print(self.alpha[m].free_symbols)

        sp_v = sp.symbols('v')

        am = self.alpha[m]
        dam_dv = sp.diff(self.alpha[m], sp_v)

        exprs = sp.cse([dam_dv])

        print(exprs)


def test_ionchannel_simplified():
    na = channelcollection.Na_Ta()
    na_simplified = channelcollection.Na_Ta_simplified()



    p_o_1 = na.computePOpen(-35.)
    p_o_2 = na_simplified.computePOpen(-35.)
    assert np.allclose(p_o_1, p_o_2)

    l_s_1 = na.computeLinSum(-35., 0., 50.)
    l_s_2 = na_simplified.computeLinSum(-35., 0., 50.)
    assert np.allclose(l_s_1, l_s_2)


def test_pickling():
    # pickle and restore
    na_ta_channel = channelcollection.Na_Ta()
    s = pickle.dumps(na_ta_channel)
    new_na_ta_channel = pickle.loads(s)

    # multiple pickles
    s = pickle.dumps(na_ta_channel)
    s = pickle.dumps(na_ta_channel)
    new_na_ta_channel = pickle.loads(s)

    assert True  # reaching this means we didn't encounter an error


if __name__ == '__main__':
    # tcns = TestChannels()
    # tcns.testBasic()

    # tna = TestNa()
    # tna.testPOpen()
    # tna.testLinSum()

    test_ionchannel_simplified()

    # Na_Ta_new()

