: Dynamics that track inside calcium concentration
: modified from Destexhe et al. 1994

NEURON	{
	SUFFIX conc_ca__
	USEION ca READ ica WRITE cai
	RANGE decay, gamma, minCai, depth
}

UNITS	{
	(mV) = (millivolt)
	(mA) = (milliamp)
	FARADAY = (faraday) (coulombs)
	(molar) = (1/liter)
	(mM) = (millimolar)
	(um)	= (micron)
}

PARAMETER	{
	gamma = 0.05 : percent of free calcium (not buffered)
	decay = 80 (ms) : rate of removal of calcium
	depth = 0.1 (um) : depth of shell
	minCai = 1e-4 (mM)
}

ASSIGNED	{ica (mA/cm2)}

STATE	{
	cai (mM)
	var
	}

INITIAL {
    var = gamma/(2*FARADAY*depth)
}

BREAKPOINT	{
	SOLVE states METHOD cnexp 
}

DERIVATIVE states	{
	rates()
	cai' = -(10000)*(ica*gamma/(2*FARADAY*depth)) - (cai - minCai)/decay
}

PROCEDURE rates(){
	var = gamma/(2*FARADAY*depth)
}