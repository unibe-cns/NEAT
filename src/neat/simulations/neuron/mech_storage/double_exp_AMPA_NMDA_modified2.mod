TITLE   double Exp NMDA synapse

NEURON {
    POINT_PROCESS double_exp_AMPA_NMDA_modified2
    RANGE tau1, tau2, tau1_NMDA, tau2_NMDA, i, e, v_eq, NMDA_ratio, f_v, z_l
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

    z_l = 0. (MOhm)     : rescale impedance
    f_v = 1.            : sigmoid voltage resc
}

ASSIGNED {
    v (mV)              : postsynaptic voltage
    i (nA)              : nonspecific current = g*(v - Erev)

    f_NMDA              : voltage dependendent magnesium blockade

    factor_AMPA
    factor_NMDA

    g_AMPA
    g_NMDA

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
}

BREAKPOINT {
    SOLVE betadyn METHOD cnexp
    g_AMPA = A_AMPA + B_AMPA
    g_NMDA = A_NMDA + B_NMDA
    mgblock(v)
    i = (g_AMPA / (1. + z_l*g_AMPA) + g_NMDA * f_NMDA / (1. + z_l*g_NMDA)) * (v - e)
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
    v_thr = v_thr0 - w_v * log(1. + z_l * g_NMDA)
    f_NMDA = 1. / (1. + exp(-(v_m - v_thr) / w_v))
}

