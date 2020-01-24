TITLE   NMDA synapse for nucleus accumbens model
: see comments below

NEURON {
	POINT_PROCESS exp_AMPA_NMDA
	RANGE tau, tau_NMDA, mg, i, e, NMDA_ratio
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
}

PARAMETER {
	tau = 10 (ms)
    tau_NMDA = 200 (ms)
	e  = 0    (mV)   : reversal potential, Dalby 2003
	mg = 1      (mM)    : external magnesium concentration
    NMDA_ratio = 0.0
}

ASSIGNED {
	v (mV)   		: postsynaptic voltage
	i (nA)   		: nonspecific current = g*(v - Erev)

	B				: voltage dependendent magnesium blockade
}


STATE { 
	g (uS)
    g_NMDA (uS)
}

INITIAL {
    g = 0
}

BREAKPOINT {
	SOLVE betadyn METHOD cnexp
  	mgblock(v)
	i = (g + g_NMDA * B) * (v - e)	
}

DERIVATIVE betadyn {
    g' = -g/tau
    g_NMDA' = -g_NMDA/tau_NMDA
}

NET_RECEIVE(weight (uS)) {
    g = g + weight
    g_NMDA = g_NMDA + weight * NMDA_ratio
}


PROCEDURE mgblock( v(mV) ) {
	: from Jahr & Stevens

	TABLE B DEPEND mg
		FROM -100 TO 100 WITH 201

	B = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
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
