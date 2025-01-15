COMMENT
Noise current characterized by gaussian distribution
with mean mean and standerd deviation stdev.

Borrows from NetStim's code so it can be linked with an external instance
of the Random class in order to generate output that is independent of
other instances of InGau.

User specifies the time at which the noise starts,
and the duration of the noise.
Since a new value is drawn at each time step,
should be used only with fixed time step integration.
ENDCOMMENT

NEURON {
    POINT_PROCESS OUReversal
    NONSPECIFIC_CURRENT i
    RANGE mean, stdev, tau
    RANGE g
    RANGE dt_usr
    RANGE delay, dur, seed_usr
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    delay = 0.      (ms) : delay until noise starts
    dur = 0.        (ms) <0, 1e9> : duration of noise
    g = 0.          (uS)
    tau = 100.      (ms)
    mean = 0        (mV)
    stdev = 1       (mV)
    seed_usr = 42   (1)
    dt_usr = .1     (ms)
    noc = 0
}

ASSIGNED {
    :dt (ms)
    v (mV)          : postsynaptic voltage
    on
    per (ms)
    gvar (mV)
    evar (mV)
    i (nA)
    flag1
    exp_decay
    amp_gauss       (nA)
    donotuse
}

INITIAL {
    on = 0
    evar = mean
    gvar = 0
    i = 0
    flag1 = 0
    exp_decay = exp(-dt_usr/tau)
    amp_gauss = stdev * sqrt(1. - exp(-2.*dt_usr/tau))
    seed(seed_usr)
}

PROCEDURE seed(x) {
    set_seed(x)
}

COMMENT
BEFORE BREAKPOINT {
    i = gvar * (v - evar)
}
ENDCOMMENT

BREAKPOINT {
    SOLVE oup
    i = gvar * (v - evar)
}

PROCEDURE oup() {
    if (t < delay) {
        gvar = 0.
        evar = mean
    }
    else {
        if (flag1 == 0) {
            flag1 = 1
            gvar = g
            evar = mean
        }
        if (t < delay+dur) {
            evar = mean + exp_decay * (evar-mean) + amp_gauss * normrand(0,1)
        }
        else {
            gvar = 0.
            evar = mean
        }
    }

}
