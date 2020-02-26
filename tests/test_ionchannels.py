import numpy as np

from neat.channels import channelcollection


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


if __name__ == '__main__':
    tcns = TestChannels()
    tcns.testBasic()
