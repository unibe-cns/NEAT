import numpy as np
import sympy as sp
# import sympy as cse

from neat.channels.channelcollection import channelcollection
from neat import IonChannel

import pytest
import pickle
import os, shutil


class TestChannels():
    def testBasic(self):
        tcn = channelcollection.TestChannel()
        v_arr = np.linspace(-80., -10., 10)

        factors = np.array([5.,1.])
        powers = np.array([[3,3,1],
                           [2,2,1]])

        varnames = np.array([[sp.symbols('a00'), sp.symbols('a01'), sp.symbols('a02')],
                             [sp.symbols('a10'), sp.symbols('a11'), sp.symbols('a12')]])
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
        var_inf_chan = tcn.computeVarinf(v_arr)
        for ind, varname in np.ndenumerate(varnames):
            assert np.allclose(var_inf[ind], var_inf_chan[varname])

        # test whether open probability is correct
        p_open = np.sum(factors[:, np.newaxis] * \
                        np.product(var_inf ** powers[:, :, np.newaxis], 1), 0)
        p_open_ = tcn.computePOpen(v_arr)
        assert np.allclose(p_open_, p_open)

        # test whether derivatives are correct
        dp_dx_chan, df_dv_chan, df_dx_chan = tcn.computeDerivatives(v_arr)

        # first: derivatives of open probability
        for ind, varname in np.ndenumerate(varnames):
            dp_dx = factors[ind[0]] * powers[ind] * \
                    np.prod(var_inf[ind[0]] ** powers[ind[0]][:, np.newaxis], 0) / var_inf[ind]
            assert np.allclose(dp_dx_chan[varname], dp_dx)

        # second: derivatives of state variable functions to voltage
        df_dv = dvarinf_dv(v_arr) / taurel(v_arr)
        for ind, varname in np.ndenumerate(varnames):
            assert np.allclose(df_dv[ind], df_dv_chan[varname])

        # third: derivatives of state variable functions to state variables
        df_dx = -1. / taurel(v_arr)
        for ind, varname in np.ndenumerate(varnames):
            assert np.allclose(df_dx[ind], df_dx_chan[varname])


def sp_exp(x):
    return sp.exp(x, evaluate=False)



def test_ionchannel_simplified(remove=True):
    if not os.path.exists('mech/'):
        os.mkdir('mech/')

    na = channelcollection.Na_Ta()

    p_o = na.computePOpen(-35.)
    assert np.allclose(p_o, 0.002009216860105564)

    l_s = na.computeLinSum(-35., 0., 50.)
    assert np.allclose(l_s, -0.00534261017220376)

    na.writeModFile('mech/')

    sk = channelcollection.SK()
    sk.writeModFile('mech/')

    if remove:
        shutil.rmtree('mech/')


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


def test_broadcasting():
    na_ta = channelcollection.Na_Ta()

    v = np.array([-73.234, -50.325, -25.459])
    s = np.array([0., 10., 20., 40.])*1j

    # error must be raised if arguments are not broadcastable
    with pytest.raises(ValueError):
        na_ta.computeLinSum(v, s)

    # check if broadcasting rules are applied correctly for voltage and frequency
    ll = na_ta.computeLinSum(v[:,None], s[None,:])
    l1 = na_ta.computeLinear(v[:,None], s[None,:])
    l2 = na_ta.computePOpen(v[:,None])

    assert ll.shape == (3,4)
    assert l1.shape == (3,4)
    assert l2.shape == (3,1)
    assert np.allclose(ll, (na_ta._getReversal(None) - v[:,None]) * l1 - l2)

    # check if broadcasting rules are applied correctly for state variables
    sv = {'m': .2,
          'h': .4}
    ll = na_ta.computeLinSum(v[:,None], s[None,:], **sv)
    assert ll.shape == (3,4)

    sv = {'m': np.array([0.1, 0.2, 0.3]),
          'h': np.array([0.9, 0.6, 0.3])}
    with pytest.raises(ValueError):
        ll = na_ta.computeLinSum(v[:,None], s[None,:], **sv)

    sv_ = {'m': sv['m'][:,None],
           'h': sv['h'][:,None]}
    ll = na_ta.computeLinSum(v[:,None], s[None,:], **sv_)
    assert ll.shape == (3,4)

    sv__ = {'m': sv['m'][:, None, None],
            'h': sv['h'][None, None, :]}
    l_ = na_ta.computeLinSum(v[:, None, None], s[None, :, None], **sv__)
    assert l_.shape == (3,4,3)
    for ii in range(4):
        assert np.allclose([ll[0,ii], ll[1,ii], ll[2,ii]],
                           [l_[0,ii,0], l_[1,ii,1], l_[2,ii,2]])

    # test braodcasting for piecewise channel
    pwc = channelcollection.PiecewiseChannel()
    varinf = pwc.computeVarinf(v)
    tauinf = pwc.computeTauinf(v)

    assert np.allclose(varinf['a'], np.array([0.1, 0.1, 0.9]))
    assert np.allclose(varinf['b'], np.array([0.8, 0.8, 0.2]))
    assert np.allclose(tauinf['a'], np.array([10., 10., 20.]))
    assert np.allclose(tauinf['b'], np.array([0.1, 0.1, 50.]))

if __name__ == '__main__':
    tcns = TestChannels()
    tcns.testBasic()
    test_ionchannel_simplified()
    test_broadcasting()
