import numpy as np
import scipy.signal as sig
import sklearn.isotonic as ski


## histogram segmentation ################################################
class histogramSegmentator:
    def __init__(self, hist):
        """
        Class to segment a histogram as returned by the numpy histogram
        function. (Delon, 2005; 2007)

        input:
            [hist]: output of the numpy histogram function
        """
        self.hist = hist[0]
        self.xmap = hist[1]
        # basic quantities
        self.N = np.sum(hist[0])
        self.L = hist[0].size
        # make the histogram noisy
        self.hist = self.hist + 1e-10 * np.random.rand(self.L)
        # frequencies in every interval
        self.r = self.hist / self.N

    def _pool_adjacent_violators(self, i0, i1, increasing=True):
        """
        Obtains an increasing or decreasing Grenander estimator of
        the histogram over the interval [i0,i1]

        input:
            [i0, i1]: ints, (inclusive) limits of the histogram
            [increasing]: boolean, if True, increasing estimator,
                else, decreasing estimator

        output:
            [r_grenander]: numpy array, obtained Grenander estimator
        """
        X = np.arange(len(self.r[i0 : i1 + 1]))
        IR = ski.IsotonicRegression(increasing=increasing)
        IR.fit(X, self.r[i0 : i1 + 1])
        r_grenander = IR.predict(X)
        return r_grenander

    def test_unimodal_hypothesis(self, i0, i1, ip, eps=1.0, maximum=True):
        """
        Tests the unimodal hypothesis, i.e. increasing hypothesis in
        [i0,p] and decreasing hypothesis over [p,i1]

        input:
            [i0,p,i1]: ints

        output:
            boolean, whether the hypothesis is valid or not
        """
        assert i0 <= ip and ip <= i1
        if maximum:
            # construct increasing Grenander estimator on [i0,ip]
            # and decreasing Grenander astimator on [ip,i1]
            r_incr = self._pool_adjacent_violators(i0, ip, increasing=True)
            r_decr = self._pool_adjacent_violators(ip, i1, increasing=False)
            # test both the increasing and decreasing hypothesis
            b_incr = self.test_hypothesis(self.r[i0 : ip + 1], r_incr, eps=eps)
            b_decr = self.test_hypothesis(self.r[ip : i1 + 1], r_decr, eps=eps)
        else:
            # construct decreasing Grenander estimator on [i0,ip]
            # and increasing Grenander astimator on [ip,i1]
            r_decr = self._pool_adjacent_violators(i0, ip, increasing=False)
            r_incr = self._pool_adjacent_violators(ip, i1, increasing=True)
            # test both the increasing and decreasing hypothesis
            b_decr = self.test_hypothesis(self.r[i0 : ip + 1], r_decr, eps=eps)
            b_incr = self.test_hypothesis(self.r[ip : i1 + 1], r_incr, eps=eps)
        return b_incr and b_decr

    def test_hypothesis(self, r, p, eps=1.0):
        """
        Tests whether the hypothesis that the array of frequencies [r]
        is generated from an underlying probability distribution [p] is
        valid.

        input:
            [r]: numpy array, array of observed frequencies
            [p]: numpy array, array of probabilies

        output:
            boolean, whether the hypothesis is true
        """
        assert len(r) == len(p)
        N = len(p)
        # compute values for all intervals
        rvec = np.zeros(N * (N + 1) // 2)
        pvec = np.zeros(N * (N + 1) // 2)
        for j0 in range(N):
            for j1 in range(j0, N):
                k = (N - j0) * j0 + j1
                rvec[k] = np.sum(r[j0 : j1 + 1])
                pvec[k] = np.sum(p[j0 : j1 + 1])
        # stabilize the log
        rvec += 1e-12
        pvec += 1e-12
        # evaluate relative entropy for all intervals
        Hvec = rvec * np.log(rvec / pvec) + (1.0 - rvec) * np.log(
            (1.0 - rvec) / (1.0 - pvec)
        )
        # hypothesis is false if Hvec contains values larger than this
        return not np.max(Hvec > eps * np.log(N * (N + 1.0) / 2.0) / self.N)

    def get_initial_partition(self):
        """
        Partitions the histogram by computing all the minima of the array
        """
        s_inds = [0] + sig.argrelmin(self.hist)[0].tolist() + [len(self.hist) - 1]
        p_inds = sig.argrelmax(self.hist)[0].tolist()
        if self.hist[0] > self.hist[1]:
            p_inds = [0] + p_inds
        if self.hist[-2] < self.hist[-1]:
            p_inds = p_inds + [len(self.hist) - 1]
        return s_inds, p_inds

    def find_unimodal_extremum(self, i0s, i1s, eps=1.0, maximum=True):
        """
        If there is a maximum (minimum) that obeys the unimodal hypothesis, returns it
        """
        ip = i0s + np.argmax(self.r[i0s : i1s + 1])
        found = self.test_unimodal_hypothesis(i0s, i1s, ip, eps=eps, maximum=maximum)
        if found:
            return ip
        else:
            return None

    def partition_fine_to_coarse(self, fix_minima=False, eps=1.0, pprint=False):
        """
        Fine to coarse partitioning algorithm.

        output:
            [s_inds]: the minima of the partition
            [p_inds]: the maxima of the partition
        """
        # initial hypothesis
        s_inds, p_inds = self.get_initial_partition()
        # find the coarsest possible partition
        k = 0
        n_iter = 0
        while k < 10 and len(s_inds) > 2:
            i = np.random.randint(1, len(s_inds) - 1)
            i0s = s_inds[i - 1]
            i1s = s_inds[i + 1]
            ip = self.find_unimodal_extremum(i0s, i1s, eps=eps)
            if ip != None:
                del s_inds[i]
                del p_inds[i]
                p_inds[i - 1] = ip
                k = 0
            else:
                k += 1
            k += 1
            n_iter += 1
            if pprint:
                print(">>> iter: " + str(n_iter) + " <<<")
        if fix_minima:
            for k in range(len(p_inds) - 1):
                i0p = p_inds[k]
                i1p = p_inds[k + 1]
                im = i0p + np.argmin(self.r[i0p : i1p + 1])
                s_inds[k + 1] = im
        return s_inds, p_inds


## empirical interpolation method ########################################
class empiricalInterpolation:
    def __init__(self, fmat, xgrid, mutrain):
        assert fmat.shape[0] == xgrid.shape[0]
        assert fmat.shape[1] == mutrain.shape[0]
        self.mutrain = mutrain
        self.xgrid = xgrid
        self.fmat = fmat

    def run_EIM(self, eps=1e-2, Mmax=50, pprint=False):
        if pprint:
            print(">>> running EIM <<<")
        Q = np.zeros((len(self.xgrid), 0))
        I_ps = []
        I_mus = []
        E_ms = []
        # residue matrix for initialization
        R = self.fmat
        # get the first parameter index
        Rsup = various.supremum_norm(R, axis=0)
        iMu = np.argmax(Rsup)
        I_mus.append(iMu)
        # get the first residue
        res = R[:, iMu]
        # iterate untill the desired precision is reached
        M = 0
        E_m = eps + 1
        while M < Mmax and E_m > eps:
            if pprint:
                print("> iter", M, " eps =", E_m)
            M += 1
            # get the interpolation point index and add it to the list
            iP = np.argmax(np.abs(res))
            I_ps.append(iP)
            # get the new basis function and add it to the matrix
            rho_new = res / res[iP]
            Q = np.concatenate((Q, rho_new[:, np.newaxis]), 1)
            # new B matrix
            B = Q[I_ps, :]
            # new parameter functions
            G = np.linalg.solve(B, self.fmat[I_ps, :])
            # new interpolant
            P = np.dot(Q, G)
            # residue matrix for all snapshots
            R = self.fmat - P
            # find the new parameter index
            Rsup = various.supremum_norm(R, axis=0)
            iMu = np.argmax(Rsup)
            I_mus.append(iMu)
            # get the new residue
            res = R[:, iMu]
            # get the new error
            E_m = Rsup[iMu]
            E_ms.append(E_m)
        else:
            if pprint:
                print("> finished, eps =", E_m)

        self.G = G
        self.Q = Q
        self.M = M

    def get_f_EIM(self, return_der=True):
        self.f_rhos = []
        self.df_rhos = []
        self.f_gammas = []
        self.df_gammas = []

        for rho in self.Q.T:
            # spline interpolation, get the knots
            tck = intp.splrep(self.xgrid, rho, k=3)

            # construct a spline evaluation function and its derivative
            def f(x, tcktuple=tck):
                return intp.splev(x, tcktuple, der=0)

            def df(x, tcktuple=tck):
                return intp.splev(x, tcktuple, der=1)

            self.f_rhos.append(f)
            self.df_rhos.append(df)

        for gamma in self.G:
            # spline interpolation, get the knots
            tck = intp.splrep(self.mutrain, gamma, k=3)

            # construct a spline evaluation function and its derivative
            def f(x, tcktuple=tck):
                return intp.splev(x, tcktuple, der=0)

            def df(x, tcktuple=tck):
                return intp.splev(x, tcktuple, der=1)

            self.f_gammas.append(f)
            self.df_gammas.append(df)

        if return_der:
            return self.f_rhos, self.f_gammas, self.df_rhos, self.df_gammas
        else:
            return self.f_rhos, self.f_gammas

    def f_intp(self, xv, mv, N=None):
        N = N or self.M
        if N > self.M:
            N = self.M

        return np.sum(
            [
                f_rho(xv)[:, np.newaxis] * self.f_gammas[i](mv)[np.newaxis, :]
                for i, f_rho in enumerate(self.f_rhos[:N])
            ],
            0,
        )

    def df_intp(self, xv, mv, N=None):
        N = N or self.M
        if N > self.M:
            N = self.M

        return np.sum(
            [
                f_rho(xv)[:, np.newaxis] * self.f_gammas[i](mv)[np.newaxis, :]
                for i, f_rho in enumerate(self.f_rhos[:N])
            ],
            0,
        )


##########################################################################
