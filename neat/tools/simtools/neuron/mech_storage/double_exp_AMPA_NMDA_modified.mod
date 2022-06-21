TITLE   double Exp NMDA synapse

NEURON {
    POINT_PROCESS double_exp_AMPA_NMDA_modified
    RANGE tau1, tau2, tau1_NMDA, tau2_NMDA, i, e, v_eq, NMDA_ratio, f_v, f_t
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER {
    tau1 = .2 (ms)
    tau2 = 3. (ms)
    tau1_NMDA = .2 (ms)
    tau2_NMDA = 43 (ms)
    e  = 0    (mV)      : reversal potential
    NMDA_ratio = 0.0
    v_eq = -75. (mV)

    v_thr0 = -12.0397 (mV) : threshold of nmda sigmoid
    w_v = 10. (mV)      : width of nmda sigmoid

    f_t = 0. (mV)       : threshold factor
    f_v = 1.            : sigmoid voltage resc
}

ASSIGNED {
    v (mV)              : postsynaptic voltage
    i (nA)              : nonspecific current = g*(v - Erev)

    f_NMDA              : voltage dependendent magnesium blockade

    factor_AMPA
    factor_NMDA

    g_tot : total conductance

    v_m : auxiliary voltage variable
    v_thr : threshold
}


STATE {
    A_AMPA (uS)
    B_AMPA (uS)
    A_NMDA (uS)
    B_NMDA (uS)
}

INITIAL {
    LOCAL tp_AMPA, tp_NMDA

    A_AMPA = 0.
    B_AMPA = 0.
    A_NMDA = 0.
    B_NMDA = 0.

    tp_AMPA = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor_AMPA = -exp(-tp_AMPA/tau1) + exp(-tp_AMPA/tau2)
    factor_AMPA = 1./factor_AMPA

    tp_NMDA = (tau1_NMDA*tau2_NMDA)/(tau2_NMDA - tau1_NMDA) * log(tau2_NMDA/tau1_NMDA)
    factor_NMDA = -exp(-tp_NMDA/tau1_NMDA) + exp(-tp_NMDA/tau2_NMDA)
    factor_NMDA = 1./factor_NMDA

    v_thr = v_thr0 - w_v * f_t
}

BREAKPOINT {
    SOLVE betadyn METHOD cnexp
    mgblock(v)
    g_tot = (A_AMPA + B_AMPA) + f_NMDA * (A_NMDA + B_NMDA)
    i = g_tot * (v - e)
}

DERIVATIVE betadyn {
    A_AMPA' = -A_AMPA/tau1
    B_AMPA' = -B_AMPA/tau2
    A_NMDA' = -A_NMDA/tau1_NMDA
    B_NMDA' = -B_NMDA/tau2_NMDA
}

NET_RECEIVE(weight (uS)) {
    A_AMPA = A_AMPA - weight*factor_AMPA
    B_AMPA = B_AMPA + weight*factor_AMPA
    A_NMDA = A_NMDA - weight*NMDA_ratio*factor_NMDA
    B_NMDA = B_NMDA + weight*NMDA_ratio*factor_NMDA
}


PROCEDURE mgblock(v (mV)) {
    v_m = f_v * (v - v_eq) + v_eq
    f_NMDA = 1. / (1. + exp(-(v_m - v_thr) / w_v))
}



COMMENT
Author Johan Hake (c) spring 2004
:     Summate input from many presynaptic sources and saturate
:     each one of them during heavy presynaptic firing

: [1] Destexhe, A., Z. F. Mainen and T. J. Sejnowski (1998)
:     Kinetic models of synaptic transmission
:     In C. Koch and I. Segev (Eds.), Methods in Neuronal Modeling

: [2] Rotter, S. and M. Diesmann (1999) Biol. Cybern. 81, 381-402
:     Exact digital simulation of time-invariant linear systems with application
:     to neural modeling

Mainen ZF, Malinow R, Svoboda K (1999) Nature. 399, 151-155.
Synaptic calcium transients in single spines indicate that NMDA
receptors are not saturated.

Chapman DE, Keefe KA, Wilcox KS (2003) J Neurophys. 89: 69-80.
Evidence for functionally distinct synaptic nmda receptors in ventromedial
vs. dorsolateral striatum.

Dalby, N. O., and Mody, I. (2003). Activation of NMDA receptors in rat
dentate gyrus granule cells by spontaneous and evoked transmitter
release. J Neurophysiol 90, 786-797.

Jahr CE, Stevens CF. (1990) Voltage dependence of NMDA activated
macroscopic conductances predicted by single channel kinetics. J
Neurosci 10: 3178, 1990.

Gutfreund H, Kinetics for the Life Sciences, Cambridge University Press, 1995, pg 234.
(suggested by Ted Carnevale)
ENDCOMMENT
