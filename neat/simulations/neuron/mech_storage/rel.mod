TITLE transmitter release

COMMENT
-----------------------------------------------------------------------------

 
   References:

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
   excitable membranes, synaptic transmission and neuromodulation using a 
   common kinetic formalism, Journal of Computational Neuroscience 1: 
   195-230, 1994.

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of 
   synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition; 
   edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1996.

  Written by Bjoern Kampa, 2004

  Modified by Willem Wybo, 2014

-----------------------------------------------------------------------------
ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS rel
	RANGE T, tau, amp
}

UNITS {
	(mM) = (milli/liter)
}

PARAMETER {
	tau (ms)	<0,1e9>
	amp (mM)
}

STATE { 
	T (mM)
}


INITIAL {
	T = 0
}

NET_RECEIVE(weight (uS)) {
	T = T + amp
}

BREAKPOINT {
	SOLVE states METHOD cnexp
}

DERIVATIVE states {
	T' = -T / tau
}


