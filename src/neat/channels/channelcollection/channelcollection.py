import sympy as sp

from neat.channels.ionchannels import IonChannel


class test_channel(IonChannel):
    """
    Simple channel to test basic functionality
    """

    def define(self):
        # open probability
        self.p_open = "5 * a00**3 * a01**3 * a02 + a10**2 * a11**2 * a12"

        # state variables activation
        self.varinf = {}
        self.varinf["a00"] = "1. / (1. + exp((v-30.) / 100.))"
        self.varinf["a01"] = "1. / (1. + exp((-v+30.)/ 100.))"
        self.varinf["a02"] = "-10."
        self.varinf["a10"] = "2. / (1. + exp((v-30.) / 100.))"
        self.varinf["a11"] = "2. / (1. + exp((-v+30.)/ 100.))"
        self.varinf["a12"] = "-30."
        # state variable time scales
        self.tauinf = {}
        self.tauinf["a00"] = "1."
        self.tauinf["a01"] = "2."
        self.tauinf["a02"] = "1."
        self.tauinf["a10"] = "2."
        self.tauinf["a11"] = "2."
        self.tauinf["a12"] = "3."
        # default reversal
        self.e = -23.0


class test_channel2(IonChannel):
    """
    Simple channel to test basic functionality
    """

    def define(self):
        # open probability
        self.p_open = ".9 * a00**3 * a01**2 + .1 * a10**2 * a11"
        # state variables
        self.varinf = {"a00": ".3", "a01": ".5", "a10": ".4", "a11": ".6"}
        self.tauinf = {"a00": "1.", "a01": "2.", "a10": "2.", "a11": "2."}
        # default reversal
        self.e = -23.0


class PiecewiseChannel(IonChannel):
    def define(self):
        # open probability
        self.p_open = "a+b"
        # state variables
        v = sp.symbols("v")
        self.varinf = {
            "a": sp.Piecewise((0.1, v < -50.0), (0.9, True)),
            "b": sp.Piecewise((0.8, v < -50.0), (0.2, True)),
        }
        self.tauinf = {
            "a": sp.Piecewise((10.0, v < -50.0), (20.0, True)),
            "b": sp.Piecewise((0.1, v < -50.0), (50.0, True)),
        }
        # default reversal
        self.e = -28.0


class Na_Ta(IonChannel):
    def define(self):
        """
        (Colbert and Pan, 2002)

        Used in (Hay, 2011)
        """
        self.ion = "na"
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = "h * m ** 3"
        # define activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "0.182 * (v + 38.) / (1. - exp(-(v + 38.) / 6.))"  # 1/ms
        self.beta["m"] = "-0.124 * (v + 38.) / (1. - exp( (v + 38.) / 6.))"  # 1/ms
        self.alpha["h"] = "-0.015 * (v + 66.) / (1. - exp( (v + 66.) / 6.))"  # 1/ms
        self.beta["h"] = "0.015 * (v + 66.) / (1. - exp(-(v + 66.) / 6.))"  # 1/ms
        # temperature factor for time-scale
        self.q10 = 2.95


class Kv3_1(IonChannel):
    def define(self):
        self.ion = "k"
        # define channel open probability
        self.p_open = "m"
        # define activation functions
        self.varinf = {"m": "1. / (1. + exp(-(v - 18.70) /  9.70))"}
        self.tauinf = {"m": "4. / (1. + exp(-(v + 46.56) / 44.14))"}  # ms


class SK(IonChannel):
    def define(self):
        """
        SK-type calcium-activated potassium current (Kohler et al., 1996)
        used in (Hay et al., 2011)
        """
        self.ion = "k"
        self.conc = ["ca"]
        # define channel open probability
        self.p_open = "z"
        # activation functions
        self.varinf = {"z": "1. / (1. + (0.00043 / ca)**4.8)"}
        self.tauinf = {"z": "1."}  # ms


class h(IonChannel):
    def define(self):
        """
        Hcn channel from (Bal and Oertel, 2000)
        """
        # define channel open probability
        self.p_open = ".8 * hf + .2 * hs"
        # define activation functions
        self.varinf, self.tauinf = {}, {}
        self.varinf["hf"] = "1. / (1. + exp((v + 82.) / 7.))"
        self.varinf["hs"] = "1. / (1. + exp((v + 82.) / 7.))"
        self.tauinf["hf"] = "40."
        self.tauinf["hs"] = "300."
        # default reversal
        self.e = -43.0
