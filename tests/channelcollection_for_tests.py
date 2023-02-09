import sympy as sp

from neat.channels.ionchannels import IonChannel


def vtrap(x, y):
    """
    Function to stabelize limit cases of 0/0 that occur in certain channel
    model rate equations
    """
    return sp.Piecewise(
        (y - x / 2., sp.Abs(x/y) < 1e-6),
        (x / (sp.exp(x/y) - 1), True)
    )


class NaTa_t(IonChannel):
    """
    Colbert and Pan 2002
    """
    def define(self):
        self.ion = 'na'
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = 'm ** 3 * h'
        # define activation functions
        self.alpha , self.beta, = {}, {}
        # NOTE : in the mod file they trapped the case where denominator is 0
        v = sp.symbols("v")
        self.alpha['m'] = 0.182 * vtrap(-(v + 38.), 6.)
        self.beta['m'] = 0.124 * vtrap((v + 38.), 6.)
        self.alpha['h'] = 0.015 * vtrap(v + 66., 6.)
        self.beta['h'] = 0.015 * vtrap(-(v + 66.), 6.)

        # temperature factor for time-scale
        self.q10 = '2.3**((34-21)/10)'


class SKv3_1(IonChannel):
    '''
    Characterization of a Shaw-related potassium channel family in rat brain,
    The EMBO Journal, vol.11, no.7,2473-2486 (1992)
    '''
    def define(self):
        self.ion = 'k'
        self.p_open = 'z'
        # activation functions
        self.varinf = {'z': ' 1/(1+exp(((v -(18.700))/(-9.700))))'}
        self.tauinf = {'z': '0.2*20.000/(1+exp(((v -(-46.560))/(-44.140))))'}


class Ca_HVA(IonChannel):
    """
    Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993
    """
    def define(self):
        self.ion = 'ca'
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = 'm ** 2 * h '
        # define activation functions
        self.alpha , self.beta, = {}, {}
        self.alpha['m'] = '(0.055*(-27-v))/(exp((-27-v)/3.8) - 1)'
        self.beta['m'] = '(0.94*exp((-75.-v)/17))'
        self.alpha['h'] = '(0.000457*exp((-13-v)/50))'
        self.beta['h'] = '(0.0065/(exp((-v-15)/28)+1))'


class Ca_LVAst(IonChannel):
    """
    Comment: LVA ca channel. Note: mtau is an approximation from the plots
    Reference: Avery and Johnston 1996, tau from Randall 1997
    Comment: shifted by -10 mv to correct for junction potential
    Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21
    """
    # NOTE: this is not exactly equal to the BBP due to the vshirft
    def define(self):
        self.ion = 'ca'
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = 'm ** 2 * h'
        # define activation functions
        self.tauinf , self.varinf, = {}, {}
        self.varinf['m'] = '1.0000/(1+ exp((v + 40.000)/-6))'
        self.tauinf['m'] = '5.0000 + 20.0000/(1+exp((v + 35.000)/5))'
        self.varinf['h'] = '1.0000/(1+ exp((v + 90.000)/6.4))'
        self.tauinf['h'] = '20.0000 + 50.0000/(1+exp((v + 50.000)/7))'
        # temperature dependence
        self.q10 = '2.3**((34-21)/10)'


class SK_E2(IonChannel):
    '''
    SK-type calcium-activated potassium current (Kohler et al., 1996)
    used in (Hay et al., 2011)
    '''
    def define(self):
        self.ion = 'k'
        self.conc = ['ca']
        self.p_open = 'z'
        # activation functions
        self.varinf = {'z': '1. / (1. + (0.00043 / ca)**4.8)'}
        # ca = sp.symbols("ca")
        # self.varinf = {'z': sp.Piecewise(
        #                         (1. / (1. + (0.00043 / ca)**4.8), ca > 1e-7),
        #                         (1. / (1. + (0.00043 / (ca + 1e-7))**4.8), True)
        #                     )}
        self.tauinf = {'z': '1.'} # ms
