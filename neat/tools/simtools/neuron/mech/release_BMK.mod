TITLE transmitter release

COMMENT
-----------------------------------------------------------------------------


   References:

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of modelays for
   excitable membranes, synaptic transmission and neuromodulation using a
   common kinetic formalism, Journal of Computational Neuroscience 1:
   195-230, 1994.

   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic modelays of
   synaptic transmission.  In: Methods in Neuronal Modelaying (2nd edition;
   edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1996.

  Written by Bjoern Kampa, 2004

-----------------------------------------------------------------------------
ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX release_BMK
	RANGE T, delay, dur, amp
}

UNITS {
	(mM) = (milli/liter)
}

PARAMETER {
	delay (ms)
	dur (ms)	<0,1e9>
	amp (mM)
}

:ASSIGNED { T (mM)
:}


STATE {
    T (mM)
}

INITIAL {
	T = 0
}

BREAKPOINT {
	at_time(delay)
	at_time(delay+dur)

	if (t < delay + dur && t > delay) {
		T = amp
	}else{
		T = 0
	}
}


