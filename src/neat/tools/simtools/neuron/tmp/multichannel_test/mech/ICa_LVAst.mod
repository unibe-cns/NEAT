: This mod file is automaticaly generated by the ``neat.channels.ionchannels`` module

NEURON {
    SUFFIX ICa_LVAst
    USEION ca WRITE ica
    RANGE  g, e
    GLOBAL h_inf, m_inf, tau_h, tau_m
    THREADSAFE
}

PARAMETER {
    g = 0.0 (S/cm2)
    e = 50.0 (mV)
    celsius (degC)
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (mM) = (milli/liter)
}

ASSIGNED {
    ica (mA/cm2)
    h_inf      
    tau_h (ms) 
    m_inf      
    tau_m (ms) 
    v (mV)
    temp (degC)
}

STATE {
    h
    m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ica = g * (h*pow(m, 2)) * (v - e)
}

INITIAL {
    rates(v)
    h = h_inf
    m = m_inf
}

DERIVATIVE states {
    rates(v)
    h' = (h_inf - h) /  tau_h 
    m' = (m_inf - m) /  tau_m 
}

PROCEDURE rates(v) {
    temp = celsius
    h_inf = 1.0/(1280165.5967642837*exp(0.15625*v) + 1)
    tau_h = (8568.1537495805551*exp((1.0/7.0)*v) + 23.705649191166216)/(1265.03762380433*exp((1.0/7.0)*v) + 1)
    m_inf = 1.0/(1 + 0.001272633801339809*exp(-1.0/6.0*v))
    tau_m = (1856.8857817932599*exp((1.0/5.0)*v) + 8.4663032825593625)/(1096.6331584284585*exp((1.0/5.0)*v) + 1)
}

