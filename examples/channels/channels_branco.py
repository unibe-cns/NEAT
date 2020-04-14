import sympy as sp

from neat import IonChannel


class Na(IonChannel):
    def __init__(self):
        '''
        L2/3 Pyr (Branco et al., 2010)
        '''
        self.ion = 'na'
        self.p_open = m**3 * h
        v = sp.symbols('v')
        # activation functions
        def vfun(v, th, r, q): return r * (v - th) / (1. - sp.exp(-(v - th) / q, evaluate=False))
        self.alpha['m'] = vfun(v, -35.013, 0.182, 9.) # 1/ms
        self.beta['m']  = vfun(-v, 35.013, 0.124, 9.) # 1/ms
        self.alpha['h'] = vfun(v, -50.013, 0.024, 5.) # 1/ms
        self.beta['h']  = vfun(-v, 75.013, 0.0091, 5.) # 1/ms
        self.q10 = 3.21


class Na_shift(IonChannel):
    def __init__(self):
        '''
        L2/3 Pyr (Branco et al., 2010)
        '''
        self.ion = 'na'
        self.p_open = m**3 * h
        v = sp.symbols('v')
        # activation functions
        def vfun(v, th, r, q): return r * (v - th) / (1. - sp.exp(-(v - th) / q, evaluate=False))
        self.alpha['m'] = vfun(v-15., -35.013, 0.182, 9.) # 1/ms
        self.beta['m']  = vfun(-v+15., 35.013, 0.124, 9.) # 1/ms
        self.alpha['h'] = vfun(v-15., -50.013, 0.024, 5.) # 1/ms
        self.beta['h']  = vfun(-v+15., 75.013, 0.0091, 5.) # 1/ms
        self.q10 = 3.21


class K_v(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def __init__(self):
        self.ion = 'k'
        self.p_open = 'n'
        # activation functions
        self.alpha = {'n': '  0.02 * (v - 25.) / (1. - exp(-(v - 25.) / 9.))'}
        self.beta  = {'n': '-0.002 * (v - 25.) / (1. - exp( (v - 25.) / 9.))'}
        self.q10 = 3.21

class K_v_shift(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def __init__(self):
        self.ion = 'k'
        self.p_open = 'n'
        # activation functions
        self.varinf = {'n': '1. / (1. + exp(-(v - 20. - 18.7)  / 9.7  ))'}
        self.tauinf = {'n': '4. / (1. + exp(-(v - 20. + 46.56) / 44.14))'} # ms
        # base class constructor
        super(K_v_shift, self).__init__()


class K_m(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def __init__(self):
        self.ion = 'k'
        self.p_open = 'n'
        # activation functions
        self.alpha = {'n': ' 0.001 * (v + 30.) / (1. - exp(-(v + 30.) / 9.))'}
        self.beta  = {'n': '-0.001 * (v + 30.) / (1. - exp( (v + 30.) / 9.))'}
        self.q10 = 3.21


class K_ca(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def __init__(self):
        self.ion = 'k'
        self.concentrations = {'ca'}
        self.p_open = 'n'
        # activation functions
        self.alpha = {'n': '0.01 * ca'}
        self.beta  = {'n': '0.02'}
        self.q10 = 3.21

class K_ir(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def __init__(self):
        self.ion = 'k'
        self.p_open = 'm'
        # asomptotic state variable functions
        self.varinf = {'m': '1. / (1. + exp((v + 82.) / 13.))'}
        self.tauinf = {'m': '6.*2.'}


class H_distal(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def __init__(self):
        self.p_open = 'l'
        # state variables
        self.varinf = {'l': '1. / 1. + exp((v + 81.) / 8.)'}
        self.tauinf = {'l': '(exp(0.0378 * 2.2 * .4 * (v + 75.))) / (1.82 * 0.011 * (1. + (exp(0.0378 * 2.2 * (v + 75.)))))'}


class Ca_T(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def __init__(self):
        self.ion = 'ca'
        self.p_open = 'm**2 * h'
        # activation functions
        self.varinf = {'m': '1. / (1. + exp(-(v + 50.) / 7.4))',
                       'h': '1. / (1. + exp( (v + 78.) / 5.0))'}
        self.tauinf = {'m': '3. + 1. / (exp((v + 25.) / 20.) + exp(-(v + 100.) / 15.))',
                       'h': '85.+ 1. / (exp((v + 46.) / 4. ) + exp(-(v + 405.) / 50.))'} # 1/ms


class Ca_H(IonChannel):
    '''
    High voltage-activated calcium channel (Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993)
    Used in (Branco, 2011)
    '''
    def __init__(self):
        self.ion = 'ca'
        self.p_open = 'm**2 * h'
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha['m'] = '-0.055 * (27. + v) / (exp(-(27. + v) / 3.8) - 1.)'  #1/ms
        self.beta['m']  = '0.94 * exp(-(75. + v) / 17.)'  #1/ms
        self.alpha['h'] = '0.000457 * exp(-(13. + v) / 50.)'   #1/ms
        self.beta['h']  = '0.0065 / (exp(-(v + 15.)/28.) + 1.)'   #1/ms
        self.q10 = 3.21

class Ca_R(IonChannel):
    '''
    R-type calcium channel (slow)
    Used in (Poirazi et al, 2003)
    '''
    def __init__(self):
        self.ion = 'ca'
        self.p_open = 'm**3 * h'
        # activation functions
        self.varinf = {'m': '1. / (1. + exp(-(v + 60.) / 3.))',
                       'h': '1. / (1. + exp(v + 62.))'}
        self.tauinf = {'m': '100.',
                       'h': '5.'} # ms


class h_u(IonChannel):
    '''
    hcn channel
    Used in (Mengual et al., 2019)
    '''
    def __init__(self):
        self.p_open = 'q'
        # activation functions
        self.alpha = {'q': '0.001*6.43* (v+154.9) / (exp((v+154.9) / 11.9) - 1.)'}
        self.beta  = {'q': '0.001*193 * exp(v/33.1)'}

