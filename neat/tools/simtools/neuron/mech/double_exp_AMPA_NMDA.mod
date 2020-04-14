TITLE   double Exp NMDA synapse

NEURON {
	POINT_PROCESS double_exp_AMPA_NMDA
	RANGE tau1, tau2, tau1_NMDA, tau2_NMDA, mg, i, e, NMDA_ratio
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
	e  = 0    (mV)   : reversal potential, Dalby 2003
	mg = 1      (mM)    : external magnesium concentration
    NMDA_ratio = 0.0
}

ASSIGNED {
	v (mV)   		: postsynaptic voltage
	i (nA)   		: nonspecific current = g*(v - Erev)

	f_NMDA				: voltage dependendent magnesium blockade

	factor_AMPA
	factor_NMDA
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
  	mgblock(v)
	i = ((A_AMPA + B_AMPA) + f_NMDA * (A_NMDA + B_NMDA)) * (v - e)	
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


PROCEDURE mgblock( v(mV) ) {
	: from Jahr & Stevens

	TABLE f_NMDA DEPEND mg
		FROM -100 TO 100 WITH 201

	:f_NMDA = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
	f_NMDA = 1/(1 + 0.3*exp(-0.1 (/mV) *(v)))
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
