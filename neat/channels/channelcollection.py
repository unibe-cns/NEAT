import numpy as np
import sympy as sp

from ionchannels import IonChannel


def sp_exp(x):
    return sp.exp(x, evaluate=False)


class TestChannel(IonChannel):
    '''
    Simple channel to test basic functionality
    '''
    def __init__(self):
        # always include this line, to define a sympy voltage symbol
        self.spV = sp.symbols('V')
        # basic factors to construct channel open probability
        self.factors = np.array([5.,1.])
        self.powers = np.array([[3,3,1],
                                [2,2,1]])
        self.varnames = np.array([['a00', 'a01', 'a02'],
                                  ['a10', 'a11', 'a12']])
        # asomptotic state variable functions
        self.varinf = np.array([[1./(1.+sp_exp(self.spV-30.)), 1./(1.+sp_exp(-self.spV+30.)), -10.],
                                [2./(1.+sp_exp(self.spV-30.)), 2./(1.+sp_exp(-self.spV+30.)), -30.]])
        # state variable relaxation time scale
        self.tauinf = np.array([[sp_exp(self.spV-30.), sp_exp(-self.spV+30.), 1.],
                                [2.*sp_exp(self.spV-30.), 2.*sp_exp(-self.spV+30.), 3.]])
        # base class instructor
        super(TestChannel, self).__init__()
