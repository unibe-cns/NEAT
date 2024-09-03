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
    POINT_PROCESS OUClamp
    NONSPECIFIC_CURRENT i
    RANGE mean, stdev, tau
    RANGE dt_usr
    RANGE delay, dur, seed_usr
}

UNITS {
    (nA) = (nanoamp)
}

PARAMETER {
    delay (ms) : delay until noise starts
    dur (ms) <0, 1e9> : duration of noise
    tau = 100.(ms)
    mean = 0 (nA)
    stdev = 1 (nA)
    seed_usr = 42 (1)
    dt_usr = .1 (ms)
    noc = 0 
}

ASSIGNED {
    :dt (ms)
    on
    per (ms)
    ival (nA)
    ivar (nA)
    i (nA)
    flag1
    exp_decay
    amp_gauss       (nA)
    donotuse
}

INITIAL {
    :VERBATIM
    :  printf("dt = %.2f\n", dt);
    :  printf("dt_usr = %.2f\n", dt_usr);
    :ENDVERBATIM
    :per = dt
    on = 0
    ivar = 0
    i = 0
    flag1 = 0
    exp_decay = exp(-dt_usr/tau) : exp(-dt/tau)
    amp_gauss = stdev * sqrt(1. - exp(-2.*dt_usr/tau)) : stdev * sqrt(1. - exp(-2.*dt/tau))
    :VERBATIM
    :  printf("std = %.10f\n", amp_gauss);
    :  printf("std_ = %.10f\n", stdev * sqrt(1. - exp(-2.*.1/tau)));
    :ENDVERBATIM
    seed(seed_usr)
}

PROCEDURE seed(x) {
    set_seed(x)
}

COMMENT
BEFORE BREAKPOINT {
    i = -ivar
}
ENDCOMMENT

BREAKPOINT {
    SOLVE oup
    i = - ivar
}

PROCEDURE oup() {
    noc = noc + 1
    :if (t < 2.) {
    :    VERBATIM
    :        printf(">>> <<<\n");
    :        printf("noc = %.2f\n", noc);
    :        printf("t  = %.2f\n", t);
    :        printf("dt = %.2f\n", dt);
    :    ENDVERBATIM
    :}
    if (t < delay) {
        ivar = 0.
    }
    else { 
        if (flag1 == 0) {
            flag1 = 1
            ivar = mean
        }
        if (t < delay+dur) {
            ivar = mean + exp_decay * (ivar-mean) + amp_gauss * normrand(0,1)
            : ivar = ivar + (mean - ivar) * dt / tau + stdev * sqrt(2*dt/tau) * normrand(0,1)
            : ivar = stdev*ival
        }
        else {  
            ivar = 0.
        }
    }
    
}
