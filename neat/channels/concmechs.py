import sympy as sp
import numpy as np


class ConcMech(object):
    def __init__(self, ion):
        self.ion = ion

    def iteritems(self):
        yield None, None

    def __repr__(self):
        return f"ConcMech(ion={self.ion})"


class ExpConcMech(ConcMech):
    def __init__(self, ion, tau, gamma):
        self.tau = tau # ms
        self.gamma = gamma # mM / (nA*ms)
        super().__init__(ion)

    def __str__(self):
        return f"ExpConcMech(ion={self.ion}, gamma={self.gamma}, tau={self.tau})"

    # def __getitem__(self, key):
    #     if key in ['tau', 'gamma']:
    #         return self.__dict__['key']
    #     else:
    #         raise KeyError(
    #             "ExpConcMech only has \'tau\' or \'gamma\' as valid keys"
    #         )

    # def __setitem__(self, key, value):
    #     if key in ['tau', 'gamma']:
    #         self.__dict__[key] = value
    #     else:
    #         raise KeyError(
    #             "ExpConcMech only has \'tau\' or \'gamma\' as valid keys"
    #         )

    def iteritems(self):
        yield 'gamma', self.gamma
        yield 'tau', self.tau

    def items(self):
        return [('gamma', self.gamma), ('tau', self.tau)]

    def computeLinear(self, freqs):
        return -self.gamma * 1e3 / (freqs + 1e3 / self.tau)

    def computeLin(self, freqs):
        return - 1e3 / (freqs + 1e3 / self.tau)

    def computeLinTauFit(self, freqs):
        return -self.gamma * 1e3 * self.tau, 1e3

    # def __str__(self):
    #     return 'tau: %.2f ms, gamma: %.6f (ms/nA)'%(self.tau, self.gamma)

    def __repr__(self):
        return f"ExpConcMech(ion={self.ion}, gamma={self.gamma:1.6g}, tau={self.tau:1.6g})"
