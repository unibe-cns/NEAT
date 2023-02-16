import sympy as sp
import numpy as np


class ConcMech(object):
    def __init__(self, ion):
        self.ion = ion

    def iteritems(self):
        yield None, None


class ExpConcMech(ConcMech):
    def __init__(self, ion, tau, gamma):
        self.tau = tau # ms
        self.gamma = gamma # mM / (nA*ms)
        super().__init__(ion)

    def __str__(self):
        return f"ExpConcMech(ion={self.ion}, gamma={self.gamma}, tau={self.tau})"

    def iteritems(self):
        yield 'gamma', self.gamma
        yield 'tau', self.tau

    def items(self):
        return [('gamma', self.gamma), ('tau', self.tau)]

    def computeLinear(self, freqs):
        return -self.gamma * 1e3 / (freqs + 1e3 / self.tau)

    def computeLin(self, freqs):
        return - 1e3 / (freqs + 1e3 / self.tau)

    def __str__(self):
        return 'tau: %.2f ms, gamma: %.6f (ms/nA)'%(self.tau, self.gamma)
