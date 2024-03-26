import numpy as np
import numpy.polynomial.polynomial as npol
import scipy.linalg as la
import scipy.optimize as op

import copy
import itertools
import types


class monicPolynomial(npol.Polynomial):
    def __init__(self, coef, coef_type='monic'):
        if len(coef) == 0:
            coef = [1.]
        elif coef_type == 'monic':
            if type(coef) == list:
                coef = coef + [1.]
            else:
                coef = np.concatenate((coef, np.array([1.])))
        elif coef_type == 'zeros':
            coef = npol.polyfromroots(coef)
        elif coef_type == 'normal':
            if np.abs(coef[-1] - 1.) > 1e-15:
                raise Exception('Coefficients don\'t define a monic polynomial')
        else:
            raise Exception('Invalid coef_type, should be \'normal\', \'monic\' or \'zeros\'.')
        super().__init__(coef)

    def f_polynomial(self):
        if len(self.coef) == 1:
            # zero'th degree polynomial is 1
            def p(x):
                if type(x) == np.ndarray:
                    return np.ones(x.shape, dtype=complex)
                else:
                    return 1.+0.*1j
        elif len(self.coef) == 2:
            # first degree polynomial, only one coefficient
            p = lambda x: x + self.coef[0]
        else:
            # higher degree polynomial
            powers = np.arange(len(self.coef))
            def p(x):
                if type(x) == np.ndarray:
                    return np.sum(self.coef[np.newaxis,:] * x[:,np.newaxis]**powers[np.newaxis,:], 1)
                else:
                    return np.sum(self.coef * x**powers)
        return p


def pf_is_left_vec(p1, p2, point):
    """
    Tests if a point is Left|On|Right of the i'th trough the points 
    in p1[i,:] and p2[i,:], for all i.

    Adapted from Maciej Kalisiak <mac@dgp.toronto.edu>, the January 
    2001 Algorithm "Area of 2D and 3D Triangles and Polygons"

    Input: 
        p1, p2: 2d numpy arrays of dimension v*2, where v is the number of
            vertices
        point: 1d numpy array of length 2, representing the point in the plane

    Return: numpy array of size v, its i'th valuas is
            >0 for point left of the line through p1[i,:] and p2[i,:]
            =0 for point on the line
            <0 for point right of the line
    """
    return (p2[:,0]-p1[:,0]) * (point[1]-p1[:,1]) - (point[0]-p1[:,0]) * (p2[:,1]-p1[:,1])


def pf_winding_number(point, poly):
    """
    Winding number test for a point in a polygon.
    (Franklin, 2000), adapted from Maciej Kalisiak <mac@dgp.toronto.edu>

    Input:  
        point: 1d numpy array of length two, representing a point in
            the plane
        poly: 2d numpy array of dimension v*2, with v the number of
            vertices
    
    Output: 
        wn: int, the winding number (=0 only if point is outside the polygon)
    """
    wn = 0
    # up intersections
    ind = np.where( np.logical_and( poly[:-1,1] <= point[1], poly[1:,1] > point[1] ) )[0]
    wn += np.where( pf_is_left_vec(poly[ind,:], poly[ind+1,:], point) > 0 )[0].shape[0]
    # down intersections
    ind = np.where( np.logical_and( poly[:-1,1] > point[1], poly[1:,1] <= point[1] ) )[0]
    wn -= np.where( pf_is_left_vec(poly[ind,:], poly[ind+1,:], point) <= 0 )[0].shape[0]

    return wn


class contour(object):
    def __init__(self, N_eval=1e3):
        self.curves =  [lambda x:               np.exp(2.*np.pi*1j*x)]
        self.dcurves = [lambda x: 2.*np.pi*1j * np.exp(2.*np.pi*1j*x)]
        self.N_eval = [N_eval]
        self.make_arrays = False

    def construct_arrays(self):
        self.ts = [np.linspace(0., 1., int(N)) for N in self.N_eval]
        self.curve_arrs = [curve(self.ts[i]) for i, curve in enumerate(self.curves)]
        self.dcurve_arrs = [dcurve(self.ts[i]) for i, dcurve in enumerate(self.dcurves)]
        self.make_arrays = True

    def store_fun_vals(self, fun):
        self.fun_arrs  = [fun(curve_arr) for curve_arr in self.curve_arrs]

    def calc_boundaries(self):
        # boundaries
        minreal = 0.; maxreal = 0.
        minimag = 0.; maximag = 0.
        for i, N in enumerate(self.N_eval):
            if self.make_arrays:
                c = self.curve_arrs[i]
            else:
                c = self.curves[i](np.linspace(0., 1., int(N)))
            mincr = np.min(c.real); maxcr = np.max(c.real)
            if mincr < minreal: minreal = mincr
            if maxcr > maxreal: maxreal = maxcr
            minci = np.min(c.imag); maxci = np.max(c.imag)
            if minci < minimag: minimag = minci
            if maxci > maximag: maximag = maxci
        self.minreal = minreal; self.maxreal = maxreal
        self.minimag = minimag; self.maximag = maximag

    def construct_polygon(self):
        """ construct polygon from curves """
        if 'polygon' not in self.__dict__:
            if self.make_arrays:
                polygon = np.concatenate(self.curve_arrs)
            else:
                polygon = np.concatenate([curve(np.linspace(0., 1., int(self.N_eval[i]))) for i, curve in enumerate(self.curves)])
            self.polygon = np.concatenate((polygon.real[:,np.newaxis], polygon.imag[:,np.newaxis]), 1)
            self.polygon[-1,:] = copy.copy(self.polygon[0,:])

    def is_inside(self, cpoint):
        self.construct_polygon()
        return pf_winding_number(np.array([cpoint.real, cpoint.imag]), self.polygon) != 0


class circularContour(contour):
    def __init__(self, radius=1., center=0.+0j, N_eval=1e3):
        super().__init__()
        self.center = center
        self.radius = radius
        self.curves  = [lambda x: center +               radius * np.exp(2.*np.pi*1j*x)]
        self.dcurves = [lambda x:          2.*np.pi*1j * radius * np.exp(2.*np.pi*1j*x)]
        self.N_eval = [N_eval]

    def is_inside(self, cpoint):
        return np.abs(cpoint - self.center) < self.radius


class trapezoidContour(contour):
    def __init__(self, t1, nparam=1e3, N_eval=1e3):
        super().__init__()
        # set of points defining the trapezoid
        self.points = [t1[1],t1[0]] + [np.conj(t1[0]), np.conj(t1[1])]
        # line segments defining the trapezoid contour
        self.curves  = [lambda x, p1=p, p2=self.points[(i+1)%len(self.points)]: p1 + x * (p2-p1) for i, p in enumerate(self.points)]
        self.dcurves = [lambda x, p1=p, p2=self.points[(i+1)%len(self.points)]:          (p2-p1) for i, p in enumerate(self.points)]
        self.N_eval = [N_eval, N_eval, N_eval, N_eval]

    def divide_real_axis(self):
        ps = self.points
        prcut = (ps[0].real + ps[1].real) / 2.
        pccut = ps[0].imag + (ps[1].imag-ps[0].imag) * (prcut - ps[0].real) / (ps[1].real-ps[0].real)

        return trapezoidContour(t1=(ps[0], prcut+1j*pccut)), trapezoidContour(t1=(prcut+1j*pccut, ps[1]))


class poleFinder:
    def __init__(self, fun=lambda x:x, dfun=lambda x:1, 
                        global_poles={}, make_arrays=False, use_known_zeros=False, **kwargs):
        # callables: the function and its derivative
        self.fun = fun
        self.dfun = dfun
        if len(global_poles) > 0:
            self.has_global_poles = True
            self.global_poles = np.array(global_poles['poles'])
            self.global_pmultiplicities = np.array(global_poles['pmultiplicities'])
        else:
            self.has_global_poles = False
        self.global_zeros = np.array([])
        self.global_zmultiplicities = np.array([])
        self.inner_contours = []; self.secondary_contours = []
        # set curves if args are given
        if len(kwargs) > 0:
            if 'poles' in list(kwargs.keys()):
                if 'pmultiplicities' in list(kwargs.keys()):
                    self.set_contour(kwargs['contour'], make_arrays=make_arrays, use_known_zeros=use_known_zeros, 
                                        poles=kwargs['poles'], pmultiplicities=kwargs['pmultiplicities'])
                else:
                    self.set_contour(kwargs['contour'], make_arrays=make_arrays, use_known_zeros=use_known_zeros,
                                        poles=kwargs['poles'])

            else:
                self.set_contour(kwargs['contour'], make_arrays=make_arrays, use_known_zeros=use_known_zeros)

    def set_contour(self, contour, make_arrays=False, use_known_zeros=False,
                        internal_contours=[],
                        poles=[], pmultiplicities=[], zeros=[], zmultiplicities=[]):
        self.make_arrays = make_arrays
        self.use_known_zeros = use_known_zeros
        # callables
        self.contour = contour
        # arrays
        if self.make_arrays:
            self.contour.construct_arrays()
            # function arrays
            self.contour.store_fun_vals(lambda x: self.dfun(x)/self.fun(x))
            # self.fun_arrs  = [ self.fun(curve_arr) for curve_arr in self.contour.curve_arrs]
            # self.dfun_arrs = [self.dfun(curve_arr) for curve_arr in self.contour.curve_arrs]
        self.contour.calc_boundaries()
        # set poles within contour and their multiplicities
        if len(poles) == 0 and self.has_global_poles:
            # add poles that are inside curve
            self.poles = np.array([(i, pole) for i, pole in enumerate(self.global_poles) if self.contour.is_inside(pole)])
            if len(self.poles) > 0:
                self.pmultiplicities = self.global_pmultiplicities[self.poles[:,0].real.astype(int)]
                self.poles = self.poles[:,1]
            else:
                self.pmultiplicities = np.array([])
                self.poles = np.array([])
        elif len(poles) > 0:
            self.poles = np.array(poles)
            if len(pmultiplicities) == 0:
                self.pmultiplicities = np.ones(len(poles))
            else:
                self.pmultiplicities = np.array(pmultiplicities)
        else:
            self.poles = []
            self.pmultiplicities = []
        # calculate the known zeros that are within the contour
        if use_known_zeros and len(self.global_zeros) > 0:
            # add zeros that are inside curve
            self.zeros = np.array([(i, zero) for i, zero in enumerate(self.global_zeros) if self.contour.is_inside(zero)])
            if len(self.zeros) > 0:
                self.zmultiplicities = self.global_zmultiplicities[self.zeros[:,0].real.astype(int)]
                self.zeros = self.zeros[:,1]
            else:
                self.zmultiplicities = np.array([])
                self.zeros = np.array([])
        else:
            self.zmultiplicities = np.array([])
            self.zeros = np.array([])

    def add_secondary_contour(self, contour):
        """
        assumed to entirely inside or entirely outside the main contour, 
        and non-overlapping with the other inner contours
        """
        if self.make_arrays:
            contour.construct_arrays()
            contour.store_fun_vals(lambda x: self.dfun(x)/self.fun(x))
        self.secondary_contours.append(contour)
        self.inner_contours.append(contour)

    def inner_prod(self, p1, p2, compute_maxpsum=False):
        # construct the integrand either as a callable or an array
        if self.make_arrays:
            integrand = [p1(curve_arr)*p2(curve_arr)*self.contour.fun_arrs[i] for i, curve_arr in enumerate(self.contour.curve_arrs)]
        else:
            integrand = lambda s: p1(s)*p2(s)*self.dfun(s)/self.fun(s)
        # construct the poles
        if len(self.poles) > 0:
            sum_poles = np.sum(self.pmultiplicities * p1(self.poles) * p2(self.poles))
        else:
            sum_poles = 0.
        # construct the known zeros
        if self.use_known_zeros and len(self.global_zeros) > 0:
            sum_zeros = - np.sum(self.zmultiplicities * p1(self.zeros) * p2(self.zeros))
            # sum_zeros = 0
        else:
            sum_zeros = 0.
        if len(self.inner_contours) > 0:
            inner_sum = 0.
            for inner_contour in self.inner_contours:
                if self.make_arrays:
                    inner_integrand = [p1(curve_arr)*p2(curve_arr)*inner_contour.fun_arrs[i] for i, curve_arr in enumerate(inner_contour.curve_arrs)]
                else:
                    inner_integrand = lambda s: p1(s)*p2(s)*self.dfun(s)/self.fun(s)
                inner_sum -= self.contour_integral(fun=inner_integrand, contour=inner_contour)
        else:
            inner_sum = 0.

        # compute the contour integral
        if compute_maxpsum:
            a, b = self.contour_integral(fun=integrand, compute_maxpsum=compute_maxpsum)
            return a + sum_poles + sum_zeros + inner_sum, b
        else:
            return self.contour_integral(fun=integrand) + sum_poles + sum_zeros + inner_sum

    def generalized_hankel_matrices(self, phis, pols):
        G  = np.zeros((len(phis), len(phis)), dtype=complex)
        G1 = np.zeros((len(phis), len(phis)), dtype=complex)
        prow = [monicPolynomial(npol.polymul(pols[1].coef, pols[j].coef), coef_type='normal').f_polynomial() for j in range(len(phis))]
        for i,j in itertools.product(list(range(len(phis))), list(range(len(phis)))):
            G [i,j] = self.inner_prod(phis[i], phis[j])
            G1[i,j] = self.inner_prod(phis[i], prow[j])
        return G, G1

    def contour_integral(self, fun=None, contour=None, compute_maxpsum=False):
        """
        Computes the contour integral over the contour formed by self.curves

        Input:
            - fun: callable or list of 1d arrays, the latter case corresponding to the function
                evaluated over the curves in self.curves. If None, self.fun is chosen as function.
            - contour: the contour over which to integrate the function. If None, the self.contour
                is chosen
            - compute_maxpsum: boolean, whether to compute the maximal partial sum

        Output:
            - psum: complex, the integral value
            - maxpsum: float, the maximum partial sum
        """
        if self.make_arrays:
            if contour == None:
                contour = self.contour
            if isinstance(fun, types.FunctionType):
                fun = [fun(curve_arr) for curve_arr in contour.curve_arrs]
            elif fun == None:
                fun = self.contour.fun_arrs
            psum = 0.;
            if compute_maxpsum: maxpsum = 0.
            for i, t in enumerate(contour.ts):
                fval = fun[i]*contour.dcurve_arrs[i]
                dt = t[1]-t[0]
                if compute_maxpsum:
                    cumpsum = psum + dt * np.cumsum(fval[:-1] + fval[1:]) / (4.*np.pi*1j)
                    psum = cumpsum[-1]
                    nmaxpsum = np.max(np.abs(cumpsum))
                    if nmaxpsum > maxpsum: maxpsum = nmaxpsum
                else:
                    psum += dt * np.sum(fval[:-1] + fval[1:]) / (4.*np.pi*1j)
        else:
            if fun == None:
                fun = lambda x: self.dfun(x) / self.fun(x)
            if contour == None:
                contour = self.contour
            psum = 0.
            if compute_maxpsum: maxpsum = 0.
            for i, curve in enumerate(contour.curves):
                dcurve = contour.dcurves[i]
                dt = 1./contour.N_eval[i]
                if isinstance(fun, types.FunctionType):
                    t = np.linspace(0., 1., int(contour.N_eval[i]))
                    fval = fun(curve(t)) * dcurve(t)
                else:
                    fval = fun[i] * dcurve(t)
                if compute_maxpsum:
                    cumpsum = psum + dt * np.cumsum(fval[:-1] + fval[1:]) / (4.*np.pi*1j)
                    psum = cumpsum[-1]
                    nmaxpsum = np.max(np.abs(cumpsum))
                    if nmaxpsum > maxpsum: maxpsum = nmaxpsum
                else:
                    psum += dt * np.sum(fval[:-1] + fval[1:]) / (4.*np.pi*1j)
        if compute_maxpsum:
            return psum, maxpsum
        else:
            return psum

    def points_within_contour(self, points):
        """
        check if all points in points lie within the contour
        """
        for point in points:
            if point.real < self.contour.minreal or point.real > self.contour.maxreal:
                return False
            elif point.imag < self.contour.minimag or point.imag > self.contour.maximag:
                return False
            # else: 
            #     if self.make_arrays:
            #         p = np.array(self.curve_arrs).flatten()
            #     else:
            #         p = np.array([curve(self.ts[i]) for i, curve in enumerate(self.curves)]).flatten()
            #     polygon = np.concatenate((p.real[:,np.newaxis], p.imag[:,np.newaxis]), 1)
            #     if pf_winding_number(np.array([point.real, point.imag]), polygon) == 0:
            #         return False
        return True


    def check_for_nonnumeric(self):
        """
        check if dfun(s) / fun(s) contains non-numeric values over the contour
        """
        if self.make_arrays:
            fun_arrs = self.contour.fun_arrs
        else:
            ts = [np.linspace(0., 1., int(N)) for N in self.countour.N_eval]
            fun_arrs = [self.dfun(t)/self.fun(t) for t in ts]

        contains_nan = [
            np.isnan(fun_arrs[ii]).any() \
            for ii in range(len(fun_arrs))
        ]
        contains_inf = [
            not np.isfinite(fun_arrs[ii]).all() \
            for ii in range(len(fun_arrs))
        ]
        # return True if the contour contains nan or infs
        return (np.any(contains_nan) or np.any(contains_inf))


    def test_contour(self, eps=1e-5, pprint=False):
        """
        Test if the contour integration on this contour is inaccurate (i.e. deviates
        from n for n in Z by more than eps). 
        """
        # function integrity check by checking if nans or infs occur on contour
        if self.check_for_nonnumeric():
            # if the contour has nans or infs, return False, the test has failed
            return False

        else:
            pol_unity = monicPolynomial([])
            p_unity = pol_unity.f_polynomial()
            residue = self.inner_prod(p_unity, p_unity)
            N = int(round(residue.real))
            accuracy = np.abs(N - residue)
            if pprint:
                print('residue =', residue)

            return accuracy < eps

    def find_zeros(self, pprint=False):
        """
        Find the zeros and poles of a complex function (C->C) inside a closed curve.

        input:
        """
        # total multiplicity of zeros (number of zeros * order)
        pol_unity = monicPolynomial([])
        p_unity = pol_unity.f_polynomial()
        residue = self.inner_prod(p_unity, p_unity)
        N = int(round(residue.real))
        accuracy = np.abs(N - residue)
        if pprint: print('N =', N, ' (accuracy =', accuracy, ')')
        # if N is zero
        if N == 0:
            n = 0
            zeros = np.array([])
        else:
            # list of FOPs
            phis = []; pols = []
            phis.append( p_unity ); pols.append( pol_unity ) # phi_0
            # compute arithmetic mean of nodes
            p_aux = monicPolynomial( [0.] ).f_polynomial()
            mu = self.inner_prod( p_unity, p_aux ) / N
            if pprint: print('mu =', mu)
            # append first polynomial
            pol = monicPolynomial( [-mu] )
            phis.append( pol.f_polynomial() ); pols.append( pol ) # phi_1
            # if the algorithm quits after the first zero, it is mu
            zeros = np.array([mu])
            n = 1
            # initialization
            r = 1; t = 0
            while r+t < N:
                if pprint: print(str(r+t))
                # naive criterion to check if phi_r+t+1 is regular
                prod_aux, maxpsum = self.inner_prod(phis[-1], phis[-1], compute_maxpsum=True)
                # compute eigenvalues of the next pencil
                G, G1 = self.generalized_hankel_matrices(phis, pols)
                eigv = la.eigvals(G1, G)
                # print eigv
                # check if these eigenvalues lie within the contour
                if self.points_within_contour(eigv+mu):
                # if np.abs(prod_aux)/maxpsum > eps_reg:
                    if pprint: print(str(r+t) + '.1')
                    # compute next FOP in the regular way
                    pol = monicPolynomial(eigv, coef_type='zeros')
                    phis.append(pol.f_polynomial()); pols.append(pol)
                    r += t+1; t = 0
                    n = r
                    zeros = eigv + mu
                else:
                    if pprint: print(str(r+t) + '.2')
                    c = npol.polyfromroots([mu])
                    pol = monicPolynomial(npol.polymul(c, pols[-1].coef), coef_type='normal')
                    pols.append(pol); phis.append(pol.f_polynomial())
                    t += 1

        # check multiplicities
        # constuct vandermonde system
        if len(zeros) > 0:
            A = np.vander(zeros).T[::-1,:]
            b = np.array([ self.inner_prod(p_unity, lambda x, k=k:x**k) for k in range(n)])
            # solve the Vandermonde system
            nu = la.solve(A, b)
            nu = np.round(nu).real.astype(int)
            sane = (np.sum(nu) == N)
        else:
            nu = np.array([])
            sane = True
        # for printing result
        if pprint:
            print('>>> Result of computation of zeros <<<')
            print('number of zeros = ', n - len(np.where(nu == 0)[0]))
            pstring = 'yes!' if sane == 1 else 'no!'
            print('sane? ' + pstring)
            for i, zero in enumerate(zeros):
                print('zero #' + str(i+1) + ' = ' + str(zero) + ' (multiplicity = ' + str(nu[i]) + ')')
            print('')
        # eliminate spurious zeros (the ones with zero multiplicity)
        inds_ = np.argsort(zeros.real)
        # inds = np.where(nu[inds_] > 0)[0]

        if self.use_known_zeros:
            self.global_zeros = np.concatenate((self.global_zeros, zeros[inds_][inds]))
            self.global_zmultiplicities = np.concatenate((self.global_zmultiplicities, nu[inds_][inds]))
            iglob = np.argsort(self.global_zeros.real)
            self.global_zeros = self.global_zeros[iglob]
            self.global_zmultiplicities = self.global_zmultiplicities[iglob]

        return zeros[inds_], (n-len(np.where(nu == 0)[0]), nu[inds_], sane)

    def find_real_zeros_recursively(self, realtol=1e-4, minradius=1., depth=0, maxdepth=10, contourparams_orig=None, pprint=False):
        """
        Searches for the zeros of a real function by using circular contours with a center on the real axis.
        If the starting contour is not accurate enough, it is split up in two equal parts.
        """
        # check if contour is circular and has real center
        assert isinstance(self.contour, circularContour)
        assert np.abs(self.contour.center.imag) < 1e-10*self.contour.radius
        # increase the depth parameter
        depth += 1
        if pprint: print('depth =', depth)
        # store the original contour parameters to restore this contour later on
        if contourparams_orig == None:
            contourparams_orig = (self.contour.center, self.contour.radius)
        # if the contour is sufficiently accurate, try to find the zeros within
        if self.test_contour(pprint=pprint):
            zeros, _ = self.find_zeros(pprint=pprint)
            realzero_inds = np.where(np.abs(zeros.imag) < realtol)[0]
            if len(realzero_inds) == 0 and len(zeros)%2 == 1:
                # if no real zeros are found, even though the number of zeros is uneven, look further
                zeros = self._continue_real_zero_recursion(depth=depth, maxdepth=maxdepth, contourparams_orig=contourparams_orig)
            else:
                zeros = zeros[realzero_inds].real
            # restore the original contour
            if contourparams_orig != None:
                contour = circularContour(center=contourparams_orig[0], radius=contourparams_orig[1])
                self.set_contour(contour, make_arrays=True)
            return zeros
        else:
            # look further for zeros
            zeros = self._continue_real_zero_recursion(depth=depth, maxdepth=maxdepth, contourparams_orig=contourparams_orig)
            # restore the original contour
            if contourparams_orig != None:
                contour = circularContour(center=contourparams_orig[0], radius=contourparams_orig[1])
                self.set_contour(contour, make_arrays=True)
            return zeros

    def _continue_real_zero_recursion(self, realtol=1e-4, minradius=1., depth=0, maxdepth=10, contourparams_orig=None, pprint=False):
        c0 = self.contour.center.real; r0 = self.contour.radius
        if r0 > minradius and depth < maxdepth:
            # first new contour for zero finding
            contour1 = circularContour(center=c0-r0/2.+0j, radius=r0/2.)
            self.set_contour(contour1, make_arrays=True)
            zeros1 = self.find_real_zeros_recursively(realtol=realtol, minradius=minradius, 
                                            depth=depth, maxdepth=maxdepth, contourparams_orig=contourparams_orig, pprint=pprint)
            # second new contour for zero finding
            contour2 = circularContour(center=c0+r0/2.+0j, radius=r0/2.)
            self.set_contour(contour2, make_arrays=True)
            zeros2 = self.find_real_zeros_recursively(realtol=realtol, minradius=minradius, 
                                            depth=depth, maxdepth=maxdepth, contourparams_orig=contourparams_orig, pprint=pprint)
            # zeros
            zeros = np.concatenate((zeros1, zeros2))
        else:
            zeros = np.array([])
        return zeros

    def find_zeros_and_poles_(self, eps_reg=1e-15, eps_stop=1e-10, P_estimate=0, pprint=False):
        """
        Find the zeros and poles of a complex function (C->C) inside a closed curve.

        input:
            -fun: callable, the function of which the poles have to be found
            -dfun: callable, the derivative of the function
            -contours: list of callables, the contours (R->C) that constitute
                a closed curve in the complex plane
            -dcontours: list of callables, the derivatives (R->C) of the
                respective contours
            -t_params: list of arrays, the parametrizations of the contours

        !!! Hard to get stopping right !!!
        """
        # total multiplicity of zeros (number of zeros * order)
        pol_unity = monicPolynomial([])
        p_unity = pol_unity.f_polynomial()
        s0 = int(round(self.inner_prod(p_unity, p_unity).real))
        N =  s0 + 2*P_estimate
        if pprint: print('N =', N)
        # list of FOPs
        phis = []; pols = []
        phis.append( p_unity ); pols.append( pol_unity ) # phi_0
        # if s0 is zero
        if s0 == 0:
            n = 0
            # arithmetic mean of nodes is zero
            mu = 0.
            pol = monicPolynomial( [0.] )
            phis.append( pol.f_polynomial() ); pols.append( pol ) # phi_1
            zeros = np.array([])
            r = 0; t = 1
        else:
            # compute arithmetic mean of nodes
            p_aux = monicPolynomial( [0.] ).f_polynomial()
            mu = self.inner_prod( p_unity, p_aux ) / N
            if pprint: print('mu =', mu)
            # append first polynomial
            pol = monicPolynomial( [-mu] )
            phis.append( pol.f_polynomial() ); pols.append( pol ) # phi_1
            # if the algorithm quits after the first zero, it is mu
            zeros = np.array([mu])
            n = 1
            # initialization
            r = 1; t = 0
        while r+t < N:
            if pprint: print('1.')
            # naive criterion to check if phi_r+t+1 is regular
            prod_aux, maxpsum = self.inner_prod( phis[-1], phis[-1], compute_maxpsum=True )
            # compute eigenvalues of the next pencil
            G, G1 = self.generalized_hankel_matrices(phis, pols)
            eigv = la.eigvals(G1, G)
            # check if these eigenvalues lie within the contour
            if self.points_within_contour(eigv+mu):
            # if np.abs(prod_aux)/maxpsum > eps_reg:
                if pprint: print('1.1')
                # compute next FOP in the regular way
                pol = monicPolynomial(eigv, coef_type='zeros')
                phis.append(pol.f_polynomial()); pols.append(pol)
                r += t+1; t = 0
                allsmall = True; tau = 0
                while allsmall and (r+tau) < N:
                    if pprint: print('1.1.1')
                    # if all further inner porducts are zero
                    taupol = npol.polyfromroots([mu for _ in range(tau)])
                    tauphi = monicPolynomial(npol.polymul(taupol, pols[-1].coef), coef_type='normal').f_polynomial()
                    ip, maxpsum = self.inner_prod(tauphi, phis[r], compute_maxpsum=True )
                    tau += 1
                    if np.abs(ip)/maxpsum > eps_stop:
                        allsmall = False
                if allsmall:
                    if pprint: print('1.1.2: STOP')
                    n = r
                    zeros = eigv + mu
                    t = N # STOP
            else:
                if pprint: print('1.2')
                c = npol.polyfromroots([mu])
                pol = monicPolynomial(npol.polymul(c, pols[-1].coef), coef_type='normal')
                pols.append(pol); phis.append(pol.f_polynomial())
                t += 1

        # check multiplicities
        # constuct vandermonde system
        A = np.vander(zeros).T[::-1,:]
        b = np.array([ self.inner_prod(p_unity, lambda x, k=k:x**k) for k in range(n)])
        # solve the Vandermonde system
        nu = la.solve(A, b)
        nu = np.round(nu).real.astype(int)
        # for printing result
        if pprint:
            print('>>> Result of computation of zeros <<<')
            print('number of zeros = ', n - len(np.where(nu == 0)[0]))
            for i, zero in enumerate(zeros):
                print('zero #' + str(i+1) + ' = ' + str(zero) + ' (multiplicity = ' + str(nu[i]) + ')')
            print('')
        # eliminate spurious zeros (the ones with zero multiplicity)
        inds = np.where(nu > 0)[0]
        return zeros[inds], n, nu[inds]

    def find_real_zeros(self, xtol=1e-9, vmin=None, vmax=None):
        if vmin == None:
            vmin = 0.
        if vmax == None:
            vmax = self.global_poles[-1]
        inds = np.where(np.logical_and(
                            np.logical_and(self.global_poles > vmin, self.global_poles < vmax),
                                        self.global_poles.imag < 1e-1) )[0]

        f_aux = lambda x: self.fun(x+0j).real
        df_aux = lambda x: self.dfun(x+0j).real

        # find the zeros
        zeros = []
        plist = np.concatenate(([vmin], self.global_poles[inds], [vmax]))
        for i in range(len(plist)-1):
            xeps = (plist[i+1]-plist[i])*xtol
            xmin = plist[i] + xeps; xmax = plist[i+1] - xeps
            # check if interval between poles is too small
            if xmin < xmax:
                # find zeros, maximal detection is 2 per interval
                sf0 = np.sign(f_aux(xmin)); sf1 = np.sign(f_aux(xmax))
                sdf0 = np.sign(df_aux(xmin)); sdf1 = np.sign(df_aux(xmax))
                xs = []
                if sf0 != sf1:
                    x0 = op.brentq(f_aux, xmin, xmax)
                    xs = [x0]
                elif sdf0 != sdf1:
                    dx0 = op.brentq(df_aux, xmin, xmax)
                    sfex = np.sign(f_aux(dx0))
                    if sfex != sf0:
                        x0 = op.brentq(f_aux, xmin, dx0)
                        # try:
                        x1 = op.brentq(f_aux, dx0, xmax)
                        # except ValueError as e:
                        #     print e
                        #     print self.fun(xmin), self.fun(xmax)
                        #     print np.sign(self.fun(xmin)), np.sign(self.fun(xmax))
                        #     print dx0
                        #     print self.fun(dx0)
                        #     print np.sign(self.fun(dx0))
                        xs = [x0, x1]
                zeros.extend(xs)
            # STILL TO BE IMPLEMTED: strategy to check if there are zeros
            # when interval is too small

        return np.sort(zeros), np.ones(len(zeros))


def find_zeros_on_segment(zeros, zmultiplicities, xmin, xmax, fun, dfun, poles, pmultiplicities, xtree='', pprint=False):
    """
    Auxiliary recursive function to find the zeros on a segment with sufficient accuracy. Decrease the contour radius 
    untill sufficient accuracy is reached
    """
    # print(f">>> xmin, xmax: {xmin}, {xmax}")
    cc = circularContour(radius=(xmax-xmin)/2., center=(xmax+xmin)/2.+0j, N_eval=1e2)
    PF = poleFinder(fun=fun, dfun=dfun, global_poles={'poles': poles, 'pmultiplicities': pmultiplicities},
                            make_arrays=True, use_known_zeros=False, contour=cc)

    if pprint:
        # print ''
        # print '>>> <<<'
        print(xtree)
        print('xmin = ', xmin, ', xmax = ', xmax)
        print('radius = ', (xmax-xmin)/2., ', center = ', (xmax+xmin)/2.)
        print(PF.test_contour(pprint=pprint))

    if PF.test_contour():
        # find the zero
        p1, info = PF.find_zeros(pprint=False)
        zeros.extend(p1.tolist()); zmultiplicities.extend(info[1].tolist())
    else:
        xmin0 = xmin; xmax0 = (xmax+xmin)/2.
        xmin1 = xmax0; xmax1 = xmax

        find_zeros_on_segment(zeros, zmultiplicities, xmin0, xmax0, fun, dfun, poles, pmultiplicities, xtree=xtree+'0', pprint=pprint)
        find_zeros_on_segment(zeros, zmultiplicities, xmin1, xmax1, fun, dfun, poles, pmultiplicities, xtree=xtree+'1', pprint=pprint)

# make sure a variable is complex
def _to_complex(x):
    if isinstance(x, np.ndarray):
        if x.dtype == float:
            x = copy.copy(x)
            x = x + 0j
    elif type(x) == float:
        x = copy.copy(x)
        x = x + 0j
    return x
