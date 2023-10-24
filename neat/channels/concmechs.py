from ..factorydefaults import DefaultPhysiology

import sympy as sp
import numpy as np


CFG = DefaultPhysiology()


class ConcMech(object):
    def __init__(self, ion):
        self.ion = ion

    def iteritems(self):
        yield None, None

    def __repr__(self):
        return f"ConcMech(ion={self.ion})"


class ExpConcMech(ConcMech):
    def __init__(self, ion, tau, gamma, inf=None):
        self.tau = tau # ms
        self.gamma = gamma # mM / (nA*ms)

        if inf is None:
            self.inf = CFG.conc[ion]
        else:
            self.inf = inf

        super().__init__(ion)

    def __str__(self):
        return f"ExpConcMech(ion={self.ion}, gamma={self.gamma}, tau={self.tau})"

    def iteritems(self):
        yield 'gamma', self.gamma
        yield 'tau', self.tau
        yield 'inf', self.inf

    def items(self):
        return [('gamma', self.gamma), ('tau', self.tau), ('inf', self.inf)]

    def computeLinear(self, freqs):
        return -self.gamma * 1e3 / (freqs + 1e3 / self.tau)

    def computeLin(self, freqs):
        return - 1e3 / (freqs + 1e3 / self.tau)

    def computeLinTauFit(self, freqs):
        return -self.gamma * 1e3 * self.tau, 1e3

    # def __str__(self):
    #     return 'tau: %.2f ms, gamma: %.6f (ms/nA)'%(self.tau, self.gamma)

    def __repr__(self):
        return f"ExpConcMech(ion={self.ion}, gamma={self.gamma:1.6g}, tau={self.tau:1.6g}, inf={self.inf:1.6g})"
