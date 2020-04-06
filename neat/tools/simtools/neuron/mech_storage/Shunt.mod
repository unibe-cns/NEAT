: A shunt current

NEURON {
	POINT_PROCESS Shunt
	NONSPECIFIC_CURRENT i
	RANGE i, e, g
}

UNITS {
	(uS) = (microsiemens)
	(mV) = (millivolt)
	(nA) = (nanoamp)
}
	
PARAMETER {
	g = 1 (uS) 
	e = 0 (mV)
}

ASSIGNED {
	i (nA)
	v (mV)
}

BREAKPOINT { 
	i = g*(v-e)
}