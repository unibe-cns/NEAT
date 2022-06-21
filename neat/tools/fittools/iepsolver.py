import numpy as np
import scipy.linalg as la
import scipy.optimize as so

import copy


class MatCollection(object):
    def __init__(self, rmat, rvec, rvar, tvec, tvar):
        self.rmat = rmat
        self.rvec = rvec
        self.rvar = rvar
        self.tmat = tvec
        self.tvec = tvar

    def calcFVar(self):
        return self.rvar

    def calcJFRow(self):
        # print 't mat ='
        # print self.tmat
        # print 't vec ='
        # print self.tvec
        return self.tvec - np.dot(self.tmat, la.solve(self.rmat, self.rvec))
        # return - np.dot(self.tmat, la.solve(self.rmat, self.rvec))
        # return self.tvec - np.dot(self.tmat, np.dot(la.inv(self.rmat), self.rvec))


class NewtonMatrices(object):
    def __init__(self):
        self.initMatrixCollection()

    def initMatrixCollection(self):
        self.matrix_collections = []

    def addCollection(self, rmat, rvec, rvar, tvec, tvar):
        mc = MatCollection(rmat, rvec, rvar, tvec, tvar)
        self.matrix_collections.append(mc)

    def getFC(self):
        return np.array([mc.calcFVar() for mc in self.matrix_collections])

    def getJFC(self):
        return np.array([mc.calcJFRow() for mc in self.matrix_collections])

    def calcResidual(self):
        # rr = self.getFC()
        # print '!! r_nn =', rr
        return la.norm(self.getFC())


class IEPSolver(object):
    def __init__(self, pencil, lambdas=None):
        assert pencil.shape[1] == pencil.shape[2]

        self.pencil = pencil
        # shape constants
        self.K = self.pencil.shape[0]
        self.M = self.pencil.shape[2]
        # matrixcollection
        self.nm = NewtonMatrices()
        # target eigenvalues
        if lambdas is not None:
            self.initLambdas(lambdas)

    def initLambdas(self, lambdas):
        self.lambdas = lambdas
        self.N = self.lambdas.shape[0]

    def evalPencil(self, cvec):
        return self.pencil[0] + np.sum([c*p_mat for c, p_mat in zip(cvec, self.pencil[1:])], axis=0)

    def _evalQRs(self, cvec):
        NN, MM, KK = self.N, self.M, self.K

        self.nm.initMatrixCollection()
        for nn, ll in enumerate(self.lambdas):
            # print '\n------------------------------'
            pp = self.evalPencil(cvec) - ll * np.eye(MM)

            qq, rr, pp = la.qr(pp, pivoting=True)
            rmat = rr[:MM-1, :MM-1]
            rvec = rr[:MM-1, -1]
            rvar = rr[-1, -1]

            ptc = self.pencil[1:] - ll * np.eye(MM)[None,:,:]
            ptc = ptc[:,:,pp]
            tmat = np.zeros((KK-1, MM-1))
            tvec = np.zeros(KK-1)
            for kk, ppc in enumerate(ptc):
            # for kk in range(self.K-1):
                # ppc_ = self.pencil[kk+1] - ll * np.eye(MM)
                # ppc_ = ppc_[:,pp]

                # print 'ppc1 =', ppc
                # print 'ppc2 =', ppc_

                tt = np.dot(qq.T, ppc)
                tmat[kk] = tt[-1,:MM-1]
                # tmat[kk] = tt[:MM-1,-1]
                tvec[kk] = tt[-1, -1]

            self.nm.addCollection(rmat, rvec, rvar, tmat, tvec)

    def updateC(self, cvec, inplace=True, pprint=False):
        self._evalQRs(cvec)
        # Jacobian and Fvec
        fvec = self.nm.getFC()
        jfmat = self.nm.getJFC()

        print('!! Jf =')
        print(jfmat)

        jfaux = np.dot(jfmat.T, jfmat)
        # compute next C vec
        delta_c = la.solve(jfaux, -np.dot(jfmat.T, fvec))
        if pprint:
            print('>> norm(Delta C) =', la.norm(delta_c))
        if inplace:
            cvec += delta_c
            return cvec
        else:
            return cvec + delta_c

    def __call__(self, c0, eps=1e-5, max_iter=20, return_residual=True, pprint=False):
        rr = 10.*eps
        kk = 0
        cc = c0
        while kk < max_iter and rr > eps:
            self.updateC(cc, pprint=pprint)
            rr = self.nm.calcResidual()
            kk += 1
            if pprint:
                print('\n---- Iter no. %d ----'%kk)
                print('>> residual =', rr)
                print('>> c =', cc)

        if pprint:
            print('\n---- Final eigenvalues ----')
            eig, _ = la.eig(self.evalPencil(cc))
            print('>> eig =', eig)

        if return_residual:
            return cc, rr
        else:
            return cc

    def minimizeResiduals(self, c0, return_residual=True, pprint=True):

        def fun_aux(cc):
            self._evalQRs(cc)
            return self.nm.calcResidual()

        # res = so.minimize(fun_aux, c0, method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-15})
        res = so.minimize(fun_aux, c0, method='TNC')
        c_sol = res['x']
        rr = res['fun']

        # print res

        if return_residual:
            return c_sol, rr
        else:
            return c_sol


class TestIEPSolver(object):
    def __init__(self, n_size=4, n_mat=6):
        self.n_size, self.n_mat = n_size, n_mat
        # define the matrix pensil
        pencil = [np.random.rand(n_size, n_size) for _ in range(n_mat)]
        pencil = np.array([p.T + p for p in pencil])
        self.ieps = IEPSolver(pencil)
        # compute eigenvalues
        self.c_orig = np.linspace(1., float(n_mat-1), n_mat-1)
        mat_orig = self.ieps.evalPencil(self.c_orig)
        lambdas_orig, _ = la.eig(mat_orig)
        self.lambdas_orig = lambdas_orig.real
        # set eigenvalues
        self.ieps.initLambdas(self.lambdas_orig[2:])

        # self.c_orig =np.array([5.,2.,7.,1.,3.])

    def testResiduals(self):
        # compute residuals and test that they are 0
        self.ieps._evalQRs(self.c_orig)
        assert np.allclose(self.ieps.nm.getFC(), np.zeros(self.ieps.N))

    def testJacobian(self, dc_over_c=0.01):
        ii = np.random.randint(self.n_mat-1)
        dc_arr = np.array([self.c_orig[ii] * dc_over_c if jj == ii else 0. for jj in range(self.n_mat-1)])
        c_1 = self.c_orig + dc_arr
        # evaluate residuals & jacobian
        self.ieps._evalQRs(self.c_orig)
        f0 = self.ieps.nm.getFC()
        jf0 = self.ieps.nm.getJFC()
        print('jf0 =\n', jf0)
        self.ieps._evalQRs(c_1)
        f1 = self.ieps.nm.getFC()
        print('\n!!!')
        print(ii)
        print(f0)
        print(f1)
        print('!!!\n')
        # evaluate approximate jacobian
        jf_col = (f1 - f0) / dc_arr[ii]
        # check if correct
        assert jf0.shape == (self.ieps.N, self.ieps.K-1)
        print(jf0[:,ii])
        print(jf_col)
        assert np.allclose(jf_col, jf0[:,ii])

    def plotContinuity(self, dc_range=np.linspace(-1.,1.,100)):
        ii = np.random.randint(self.n_mat-1)
        rvar_range = np.zeros((dc_range.shape[0], self.n_size))
        for jj, dc in enumerate(dc_range):
            dc_arr = np.array([self.c_orig[ii] * dc if kk == ii else 0. for kk in range(self.n_mat-1)])
            c_1 = self.c_orig + dc_arr

            self.ieps._evalQRs(c_1)
            rvar_range[jj] = self.ieps.nm.getFC()

        # for kk in range(self.n_size):
        #     pl.plot(dc_range, rvar_range[:,kk])
        # pl.show()

    def plotJacobian(self, dc_range=np.linspace(-1.,1.,100)):
        delta_c = dc_range[1] - dc_range[0]
        ii = np.random.randint(self.n_mat-1)
        rvar_range = np.zeros((dc_range.shape[0], self.n_size))
        drvar_range = np.zeros((dc_range.shape[0], self.n_size))
        for jj, dc in enumerate(dc_range):
            dc_arr = np.array([self.c_orig[ii] * dc if kk == ii else 0. for kk in range(self.n_mat-1)])
            c_1 = self.c_orig + dc_arr

            self.ieps._evalQRs(c_1)
            rvar_range[jj] = self.ieps.nm.getFC()
            drvar_range[jj] = self.ieps.nm.getJFC()[:,ii]

        dc_ = (dc_range[1:] + dc_range[:-1]) / 2.
        drvar_approx = (rvar_range[1:,:] - rvar_range[:-1,:]) / delta_c


        # # from datarep.matplotlibsettings import *
        # for kk in range(self.n_size):
        #     pl.plot(dc_, drvar_approx[:,kk], c=colours[kk%len(colours)])
        #     pl.plot(dc_range, drvar_range[:,kk], lw=2, ls='--', c=colours[kk%len(colours)])
        # pl.show()


    def testSolver(self, eps=1.):

        print('\nc_0 =', self.c_orig)
        p0 = self.ieps.evalPencil(self.c_orig)
        eig0, _ = la.eig(p0)
        print('eig_0 =', eig0.real)

        self.ieps._evalQRs(self.c_orig)
        r0 = self.ieps.nm.getFC()
        print('r_0 =', r0)


        c_perturb = self.c_orig + eps * np.random.randn(*self.c_orig.shape)
        print('\nc_1 =', c_perturb)

        p1 = self.ieps.evalPencil(c_perturb)
        eig1, _ = la.eig(p1)
        print('eig_1 =', eig1.real)

        self.ieps._evalQRs(c_perturb)
        r1 = self.ieps.nm.getFC()
        print('r_1 =', r1)

        # c_fit = ieps(c_perturb, pprint=True)
        c_fit, r_fit = self.ieps.minimizeResiduals(c_perturb, pprint=True)


        print('\nc_fit =', c_fit)
        p_fit = self.ieps.evalPencil(c_fit)
        eig2, _ = la.eig(p_fit)
        print('eig_fit =', eig2.real)

        self.ieps._evalQRs(c_fit)
        r2 = self.ieps.nm.getFC()
        print('r_fit =', r2)

if __name__ == '__main__':
    ts = TestIEPSolver()

    # ts.testResiduals()
    # ts.testJacobian()

    ts.testSolver()

    # ts.plotContinuity()
    # ts.plotJacobian()

