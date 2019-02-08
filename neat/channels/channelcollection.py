import numpy as np
import sympy as sp

from ionchannels import IonChannel


def sp_exp(x):
    return sp.exp(x, evaluate=False)

def sp_pow(x, y):
    return sp.Pow(x, y, evaluate=False)

# dictionary with suggested reversal potential for each channel
E_REV_DICT = {
                'TestChannel': -23.,
                'TestChannel2': -23.,
                'h': -43.,
                # Hay et al., 2011 channels
                'h_HAY': -45.,
                'Na_Ta': 50.,
                'Na_p': 50.,
                'Kv3_1': -85.,
                'Kpst': -85.,
                'Ktst': -85.,
                'm': -85.,
                'SK': -85.,
                'Ca_LVA': 50.,
                'Ca_HVA': 50.,
                'L': -75.,
                # Miyasho et al., 2001 channels
                'CaE': 50.,
                'CaP': 50.,
                'CaP2': 50.,
                'CaT': 50.,
                'NaP': 50.,
                'NaF': 50.,
                'K23': -85.,
                'KC3': -85.,
                'KA': -85.,
                'KD': -85.,
                'KM': -85.,
                'Kh': -85.,
                'Khh': -85.,
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


class TestChannel(IonChannel):
    '''
    Simple channel to test basic functionality
    '''
    def __init__(self):
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # basic factors to construct channel open probability
        self.factors = np.array([5.,1.])
        self.powers = np.array([[3,3,1],
                                [2,2,1]])
        self.varnames = np.array([['a00', 'a01', 'a02'],
                                  ['a10', 'a11', 'a12']])
        # asomptotic state variable functions
        self.varinf = np.array([[1./(1.+sp_exp((self.sp_v-30.)/100.)), 1./(1.+sp_exp((-self.sp_v+30.)/100.)), -10.],
                                [2./(1.+sp_exp((self.sp_v-30.)/100.)), 2./(1.+sp_exp((-self.sp_v+30.)/100.)), -30.]])
        # state variable relaxation time scale
        self.tauinf = np.array([[1., 2., 1.],
                                [2., 2., 3.]])
        # base class instructor
        super(TestChannel, self).__init__()


class TestChannel2(IonChannel):
    '''
    Simple channel to test basic functionality
    '''
    def __init__(self):
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # basic factors to construct channel open probability
        self.factors = np.array([.9,.1])
        self.powers = np.array([[3,2],
                                [2,1]])
        self.varnames = np.array([['a00', 'a01'],
                                  ['a10', 'a11']])
        # asomptotic state variable functions
        self.varinf = np.array([[sp.Float(.3), sp.Float(.5)],
                                [sp.Float(.4), sp.Float(.6)]])
        # state variable relaxation time scale
        self.tauinf = np.array([[1., 2.],
                                [2., 2.]])
        # base class instructor
        super(TestChannel2, self).__init__()


class h(IonChannel):
    def __init__(self, ratio=0.2):
        '''
        Hcn channel from (Bal and Oertel, 2000)
        '''
        self.ion = ''
        self.concentrations = []
        self.ratio = ratio
        self.tauf = 40. # ms
        self.taus = 300. # ms
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # basic factors to construct channel open probability
        self.varnames = np.array([['hf'],
                                  ['hs']])
        self.powers = np.array([[1],
                                [1]], dtype=int)
        self.factors = np.array([1.-self.ratio, self.ratio])
        # asomptotic state variable functions
        self.varinf = np.array([[1./(1.+sp_exp((self.sp_v+82.)/7.))],
                                [1./(1.+sp_exp((self.sp_v+82.)/7.))]])
        # state variable relaxation time scales
        self.tauinf = np.array([[sp.Float(self.tauf)],
                                [sp.Float(self.taus)]])
        # base class constructor
        super(h, self).__init__()


class h_HAY(IonChannel):
    def __init__(self):
        '''
        Hcn channel from (Kole, Hallermann and Stuart, 2006)

        Used in (Hay, 2011)
        '''
        self.ion = ''
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 0.001 * 6.43 * (self.sp_v + 154.9) / (sp_exp((self.sp_v + 154.9) / 11.9) - 1.)
        spbetam  = 0.001 * 193. * sp_exp(self.sp_v / 33.1)
        self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam)]])
        # base class constructor
        super(h_HAY, self).__init__()


class Na_Ta(IonChannel):
    def __init__(self):
        '''
        (Colbert and Pan, 2002)

        Used in (Hay, 2011)
        '''
        self.ion = 'na'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham =   0.182 * (self.sp_v + 38.) / (1. - sp_exp(-(self.sp_v + 38.) / 6.)) # 1/ms
        spbetam  = - 0.124 * (self.sp_v + 38.) / (1. - sp_exp( (self.sp_v + 38.) / 6.)) # 1/ms
        spalphah = - 0.015 * (self.sp_v + 66.) / (1. - sp_exp( (self.sp_v + 66.) / 6.)) # 1/ms
        spbetah  =   0.015 * (self.sp_v + 66.) / (1. - sp_exp(-(self.sp_v + 66.) / 6.)) # 1/ms
        self.varinf = np.array([[spalpham / (spalpham + spbetam),
                                 spalphah / (spalphah + spbetah) ]])
        self.tauinf = np.array([[(1./2.95) / (spalpham + spbetam),
                                 (1./2.95) / (spalphah + spbetah)]]) # 1/ms
        # base class constructor
        super(Na_Ta, self).__init__()


class Na_p(IonChannel):
    def __init__(self):
        '''
        Derived by (Hay, 2011) from (Magistretti and Alonso, 1999)

        Used in (Hay, 2011)

        !!! Does not work !!!
        '''
        self.ion = 'na'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham =   0.182   * (self.sp_v + 38. ) / (1. - sp_exp(-(self.sp_v + 38. ) / 6.  ))  #1/ms
        spbetam  = - 0.124   * (self.sp_v + 38. ) / (1. - sp_exp( (self.sp_v + 38. ) / 6.  ))  #1/ms
        spalphah = - 2.88e-6 * (self.sp_v + 17. ) / (1. - sp_exp( (self.sp_v + 17. ) / 4.63))   #1/ms
        spbetah  =   6.94e-6 * (self.sp_v + 64.4) / (1. - sp_exp(-(self.sp_v + 64.4) / 2.63))   #1/ms
        self.varinf = np.array([[   1. / (1. + sp_exp(-(self.sp_v + 52.6) / 4.6)) ,
                                    1. / (1. + sp_exp( (self.sp_v + 48.8) / 10.)) ]])
        self.tauinf = np.array([[(6./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(Na_p, self).__init__()

# mInf = 1.0/(1+exp((v- -52.6)/-4.6))
# mAlpha = (0.182 * (v- -38))/(1-(exp(-(v- -38)/6)))
#         mBeta  = (0.124 * (-v -38))/(1-(exp(-(-v -38)/6)))
#         mTau = 6*(1/(mAlpha + mBeta))/qt

#         hInf = 1.0/(1+exp((v- -48.8)/10))
#     hAlpha = -2.88e-6 * (v + 17) / (1 - exp((v + 17)/4.63))
#     hBeta = 6.94e-6 * (v + 64.4) / (1 - exp(-(v + 64.4)/2.63))
#         hTau = (1/(hAlpha + hBeta))/qt



class Kv3_1(IonChannel):
    def __init__(self):
        '''
        Shaw-related potassium channel (Rettig et al., 1992)

        Used in (Hay et al., 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v - 18.7) / 9.7))]])
        self.tauinf = np.array([[4. / (1. + sp_exp(-(self.sp_v + 46.56) / 44.14))]]) # ms
        # base class constructor
        super(Kv3_1, self).__init__()


class Kpst(IonChannel):
    def __init__(self):
        '''
        Persistent Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v + 11.) / 12.)) ,
                                 1. / (1. + sp_exp( (self.sp_v + 64.) / 11.)) ]])
        self.tauinf = np.array([[(3.04 + 17.3 * sp_exp(-((self.sp_v + 60.) / 15.9)**2) + 25.2 * sp_exp(-((self.sp_v + 60.) / 57.4)**2)) / 2.95, \
                                 (360. + (1010. + 24. * (self.sp_v + 65.)) * sp_exp(-((self.sp_v + 85.) / 48.)**2)) / 2.95]]) # ms
        # base class constructor
        super(Kpst, self).__init__()


class Ktst(IonChannel):
    def __init__(self):
        '''
        Transient Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v + 10.) / 19.)) ,
                                 1. / (1. + sp_exp( (self.sp_v + 76.) / 10.)) ]])
        self.tauinf = np.array([[(0.34 + 0.92 * sp_exp(-((self.sp_v + 81.) / 59.)**2)) / 2.95 ,
                                 (8.   + 49.  * sp_exp(-((self.sp_v + 83.) / 23.)**2)) / 2.95]]) # ms
        # base class constructor
        super(Ktst, self).__init__()


class m(IonChannel):
    def __init__(self):
        '''
        M-type potassium current (Adams, 1982)

        Used in (Hay, 2011)

        !!! does not work when e > V0 !!!
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 3.3e-3 * sp_exp( 2.5 * 0.04 * (self.sp_v + 35.))
        spbetam = 3.3e-3 * sp_exp(-2.5 * 0.04 * (self.sp_v + 35.))
        self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
        self.tauinf = np.array([[(1. / (spalpham + spbetam)) / 2.95]])
        # base class constructor
        super(m, self).__init__()


class SK(IonChannel):
    def __init__(self):
        '''
        SK-type calcium-activated potassium current (Kohler et al., 1996)

        used in (Hay et al., 2011)
        '''
        self.ion = 'k'
        self.concentrations = ['ca']
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        self.sp_c = [sp.symbols(conc) for conc in self.concentrations]
        # state variables
        self.varnames = np.array([['z']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        zinf = 1. / (1. + sp_pow(0.00043 / self.sp_c[0], 4.8))
        ztau = sp.Float(1.)
        self.varinf = np.array([[zinf]])
        self.tauinf = np.array([[ztau]]) # ms
        # base class constructor
        super(SK, self).__init__()


class Ca_LVA(IonChannel):
    def __init__(self):
        '''
        LVA calcium channel (Avery and Johnston, 1996; tau from Randall, 1997)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v + 40.)/6.)), \
                                 1. / (1. + sp_exp((self.sp_v + 90.)/6.4))]])
        self.tauinf = np.array([[5. + 20./(1. + sp_exp((self.sp_v  + 35.)/5.))/2.95,
                                 20. + 50./(1. + sp_exp((self.sp_v + 50.)/7.))/2.95]]) # ms
        # base class constructor
        super(Ca_LVA, self).__init__()


class Ca_HVA(IonChannel):
    def __init__(self):
        '''
        High voltage-activated calcium channel (Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = -0.055 * (27. + self.sp_v) / (sp_exp(-(27. + self.sp_v)/3.8) - 1.)  #1/ms
        spbetam = 0.94 * sp_exp(-(75. + self.sp_v)/17.)  #1/ms
        spalphah = 0.000457 * sp_exp(-(13. + self.sp_v)/50.)   #1/ms
        spbetah = 0.0065 / (sp_exp(-(self.sp_v + 15.)/28.) + 1.)   #1/ms
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(Ca_HVA, self).__init__()


class CaE(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[1,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 2.6/(1+sp_exp((self.sp_v+7)/(-8)))
        spbetam =  0.18/(1+sp_exp((self.sp_v+26)/4))
        spalphah = 0.0025/(1+sp_exp((self.sp_v+32)/8))
        spbetah = 0.19/(1+sp_exp((self.sp_v+42)/(-10)))
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(CaE, self).__init__()


class CaP(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[1,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 8.5 / (1. + sp_exp(-(self.sp_v - 8.) / 12.5))  #1/ms
        spbetam = 35. / (1. + sp_exp((self.sp_v + 74.) / 14.5))  #1/ms
        spalphah = 0.0015 / (1. + sp_exp((self.sp_v + 29.) / 8.))  #1/ms
        spbetah = 0.0055 / (1. + sp_exp(-(self.sp_v + 23.) / 8.))  #1/ms
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(CaP, self).__init__()


class CaP2(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 8.5/(1+sp_exp((self.sp_v-8)/(-12.5)))
        spbetam =  35/(1+sp_exp((self.sp_v+74)/14.5))
        self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam)]]) # ms
        # base class constructor
        super(CaP2, self).__init__()


class CaT(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[1,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 2.6/(1+sp_exp((self.sp_v+21)/(-8)))
        spbetam =  0.18/(1+sp_exp((self.sp_v+40)/4))
        spalphah = 0.0025/(1+sp_exp((self.sp_v+40)/8))
        spbetah = 0.19/(1+sp_exp((self.sp_v+50)/(-10)))
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(CaT, self).__init__()


class NaF(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'na'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 35/sp_exp((self.sp_v+5)/(-10))
        spbetam =  7/sp_exp((self.sp_v+65)/20)
        spalphah = 0.225/(1+sp_exp((self.sp_v+80)/10))
        spbetah = 7.5/sp_exp((self.sp_v-3)/(-18))
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(NaF, self).__init__()


class NaP(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'na'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[3]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 200./(1+sp_exp((self.sp_v-18)/(-16)))
        spbetam = 25/(1+sp_exp((self.sp_v+58)/8))
        self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam)]]) # ms
        # base class constructor
        super(NaP, self).__init__()


class K23(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'k'
        self.concentrations = ['ca']
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        self.sp_c = [sp.symbols(conc) for conc in self.concentrations]
        # state variables
        self.varnames = np.array([['m', 'z']])
        self.powers = np.array([[1,2]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        zinf = 1. / (1. + 20. / (self.sp_c[0]*1000))
        ztau = sp.Float(10.)
        maux = 0.075 / sp_exp((self.sp_v + 5.) / 10.)
        minf = 25. / (25. + maux)
        mtau = 1. / (25. + maux)
        self.varinf = np.array([[minf, zinf]])
        self.tauinf = np.array([[mtau, ztau]]) # ms
        # base class constructor
        super(K23, self).__init__()


class KC3(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'k'
        self.concentrations = ['ca']
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        self.sp_c = [sp.symbols(conc) for conc in self.concentrations]
        # state variables
        self.varnames = np.array([['m', 'z']])
        self.powers = np.array([[1,2]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        zinf = 1. / (1. + 400. / (self.sp_c[0]*1000))
        ztau = sp.Float(10.)
        maux = 0.11 / sp_exp((self.sp_v - 35.) / 14.9)
        minf = 7.5 / (7.5 + maux)
        mtau = 1. / (7.5 + maux)
        self.varinf = np.array([[minf, zinf]])
        self.tauinf = np.array([[mtau, ztau]]) # ms
        # base class constructor
        super(KC3, self).__init__()


class KA(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[4, 1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 1.4/(1+sp_exp((self.sp_v+27)/(-12)))
        spbetam =  0.49/(1+sp_exp((self.sp_v+30)/4))
        spalphah = 0.0175/(1+sp_exp((self.sp_v+50)/8))
        spbetah = 1.3/(1+sp_exp((self.sp_v+13)/(-10)))
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(KA, self).__init__()


class KD(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[1, 1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 8.5/(1+sp_exp((self.sp_v+17)/(-12.5)))
        spbetam = 35/(1+sp_exp((self.sp_v+99)/14.5))
        spalphah = 0.0015/(1+sp_exp((self.sp_v+89)/8))
        spbetah = 0.0055/(1+sp_exp((self.sp_v+83)/(-8)))
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(KD, self).__init__()


class KM(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1.0 / (1+sp_exp(-(self.sp_v+35)/10))]])
        self.tauinf = np.array([[1. / (3.3*(sp_exp((self.sp_v+35)/40)+sp_exp(-(self.sp_v+35)/20))/200)]]) # ms
        # base class constructor
        super(KM, self).__init__()


class Kh(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m'], ['n']])
        self.powers = np.array([[1], [1]], dtype=int)
        self.factors = np.array([.8, .2])
        # activation functions
        auxinf = 1/(1+sp_exp((self.sp_v+78)/7))
        # asomptotic state variable functions
        self.varinf = np.array([[auxinf],
                                [auxinf]])
        # state variable relaxation time scales
        self.tauinf = np.array([[sp.Float(38.)],
                                [sp.Float(319.)]])
        # base class constructor
        super(Kh, self).__init__()


class Khh(IonChannel):
    def __init__(self):
        '''
        Purkinje Cell (Miyasho et al., 2001)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['n']])
        self.powers = np.array([[4]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalphan = .01 * -(self.sp_v + 55.) / (sp_exp(-(self.sp_v + 55.) / 10.) - 1.)
        spbetan = .125 * sp_exp(-(self.sp_v + 65.) / 80.)
        self.varinf = np.array([[spalphan / (spalphan + spbetan)]])
        self.tauinf = np.array([[1. / (spalphan + spbetan)]]) # ms
        # vtrap activation functions
        spalphan_vtrap = .01 / (0.1 - (self.sp_v + 55.) / 200.)
        self.varinf_vtrap = np.array([[spalphan_vtrap / (spalphan_vtrap + spbetan)]])
        self.tauinf_vtrap = np.array([[1. / (spalphan_vtrap + spbetan)]]) # ms
        # base class constructor
        super(Khh, self).__init__()

    def lambdifyVarInf(self):
        f_varinf_vtrap = np.zeros(self.varnames.shape, dtype=object)
        for ind, varinf in np.ndenumerate(self.varinf_vtrap):
            varinf = self._substituteConc(varinf)
            f_varinf_vtrap[ind] = sp.lambdify(self.sp_v, varinf)

        f_varinf_aux = super(Khh, self).lambdifyVarInf()

        def f_varinf(vv):
            if isinstance(vv, float):
                if np.abs(vv + 55.) < 0.001:
                    return f_varinf_vtrap[0,0](vv)
                else:
                    return f_varinf_aux[0,0](vv)
            else:
                fv_return = np.zeros_like(vv)
                bool_vtrap = np.abs(vv + 55) < 0.0001
                inds_vtrap = np.where(bool_vtrap)
                fv_return[inds_vtrap] = f_varinf_vtrap[0,0](vv[inds_vtrap])
                inds = np.where(np.logical_not(bool_vtrap))
                fv_return[inds] = f_varinf_aux[0,0](vv[inds])
                return fv_return

        return np.array([[f_varinf]])

    def lambdifyTauInf(self):
        f_tauinf_vtrap = np.zeros(self.varnames.shape, dtype=object)
        for ind, tauinf in np.ndenumerate(self.tauinf_vtrap):
            tauinf = self._substituteConc(tauinf)
            f_tauinf_vtrap[ind] = sp.lambdify(self.sp_v, tauinf)

        f_tauinf_aux = super(Khh, self).lambdifyTauInf()

        def f_tauinf(vv):
            if isinstance(vv, float):
                if np.abs(vv + 55.) < 0.001:
                    return f_tauinf_vtrap[0,0](vv)
                else:
                    return f_tauinf_aux[0,0](vv)
            else:
                fv_return = np.zeros_like(vv)
                bool_vtrap = np.abs(vv + 55) < 0.0001
                inds_vtrap = np.where(bool_vtrap)
                fv_return[inds_vtrap] = f_tauinf_vtrap[0,0](vv[inds_vtrap])
                inds = np.where(np.logical_not(bool_vtrap))
                fv_return[inds] = f_tauinf_aux[0,0](vv[inds])
                return fv_return

        return np.array([[f_tauinf]])

    def lambdifyDerivatives(self):
        fstatevar_vtrap = (self.varinf_vtrap - self.statevars) / self.tauinf_vtrap
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
            f_sv = self._substituteConc(fstatevar_vtrap[ind])
            df_dv_aux[ind] = sp.lambdify(args,
                                     sp.diff(f_sv, self.sp_v, 1))
            df_dx_aux[ind] = sp.lambdify(args,
                                     sp.diff(f_sv, var, 1))
        # define convenient functions
        def dp_dx_vtrap(*args):
            dp_dx_list = [[] for _ in range(self.statevars.shape[0])]
            for ind, dp_dx_ in np.ndenumerate(dp_dx_aux):
                dp_dx_list[ind[0]].append(dp_dx_aux[ind](*args))
            return np.array(dp_dx_list)
        def df_dv_vtrap(*args):
            df_dv_list = [[] for _ in range(self.statevars.shape[0])]
            for ind, df_dv_ in np.ndenumerate(df_dv_aux):
                df_dv_list[ind[0]].append(df_dv_aux[ind](*args))
            return np.array(df_dv_list)
        def df_dx_vtrap(*args):
            df_dx_list = [[] for _ in range(self.statevars.shape[0])]
            for ind, df_dx_ in np.ndenumerate(df_dx_aux):
                df_dx_list[ind[0]].append(df_dx_aux[ind](*args))
            return np.array(df_dx_list)

        dp_dx_aux_, df_dv_aux_, df_dx_aux_ = super(Khh, self).lambdifyDerivatives()

        dp_dx = _func(dp_dx_aux_, dp_dx_vtrap, -55.)
        df_dv = _func(df_dv_aux_, df_dv_vtrap, -55.)
        df_dx = _func(df_dx_aux_, df_dx_vtrap, -55.)

        return dp_dx, df_dv, df_dx

    # def lambdifyTauInf(self):
    #     f_tauinf = np.zeros(self.varnames.shape, dtype=object)
    #     for ind, tauinf in np.ndenumerate(self.tauinf):
    #         tauinf = self._substituteConc(tauinf)
    #         f_tauinf[ind] = sp.lambdify(self.sp_v, tauinf)
    #     return f_tauinf


# class Khh(IonChannel):
#     def __init__(self):
#         '''
#         Purkinje Cell (Miyasho et al., 2001)
#         '''
#         self.ion = 'k'
#         self.concentrations = []
#         # always include this line, to define a sympy voltage symbol
#         self.sp_v = sp.symbols('v')
#         # state variables
#         self.varnames = np.array([['n']])
#         self.powers = np.array([[4]], dtype=int)
#         self.factors = np.array([1.])
#         # activation functions
#         spalphan = .01 * -(self.sp_v + 55.001) / (sp_exp(-(self.sp_v + 55.001) / 10.) - 1.)
#         spbetan = .125 * sp_exp(-(self.sp_v + 65.) / 80.)
#         self.varinf = np.array([[spalphan / (spalphan + spbetan)]])
#         self.tauinf = np.array([[1. / (spalphan + spbetan)]]) # ms
#         # base class constructor
#         super(Khh, self).__init__()





