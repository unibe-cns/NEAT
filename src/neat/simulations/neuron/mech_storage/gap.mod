: Gap junction point process

NEURON {
    POINT_PROCESS Gap
    POINTER vgap
    RANGE i, g
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER { 
    g = 1e-3 (uS) 
}

ASSIGNED {
    v (mV)
    vgap (mV)
    i (nA)
}

BREAKPOINT { 
    i = (v - vgap) * g
}