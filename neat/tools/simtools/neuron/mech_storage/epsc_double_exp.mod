COMMENT
Current based synapse with double exponential profile

Author:	Willem Wybo
Date: 	29/09/2014
ENDCOMMENT

NEURON {
	POINT_PROCESS epsc_double_exp
	RANGE tau1, tau2, i
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
}

PARAMETER {
	tau1 = 0.2 (ms)
	tau2 = 3.0 (ms)
}

ASSIGNED {
	i		(nA)
	tp 		(ms)
	factor 	(1)
}

STATE {
	A (nA)
	B (nA)
}

INITIAL {
	tp 		= (tau1*tau2) / (tau2-tau1) * log(tau2/tau1)
    factor 	= 1./(-exp(-tp/tau1) + exp(-tp/tau2))
}

BREAKPOINT {
	SOLVE dyn METHOD cnexp
	i = A - B	
}

DERIVATIVE dyn {
	A' = -A / tau1
	B' = -B / tau2
}

NET_RECEIVE(weight (nA)) {
    A = A + weight*factor
    B = B + weight*factor
}
