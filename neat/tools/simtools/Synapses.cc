#include "Synapses.h"


// exponential conductance window //////////////////////////////////////////////
void ExpCond::init(double dt){
	m_dt = dt;
	m_p = exp(-dt / m_tau);
	m_g = 0.;
};

void ExpCond::setParams(double tau){
	m_tau = tau;
};

void ExpCond::feedSpike(double g_max, int n_spike){
	m_g += g_max * double(n_spike);
};

double ExpCond::advance(double dt){
	if(abs(dt - m_dt) < 1.0e-9){
		m_p = exp(-dt / m_tau);
	}
	m_g *= m_p;
	return m_g;
};

double ExpCond::getSurface(){
	return m_tau;
};
////////////////////////////////////////////////////////////////////////////////


// double exponential conductance window ///////////////////////////////////////
void Exp2Cond::init(double dt){
	m_dt = dt;
	m_p_r = exp(-dt / m_tau_r); m_p_d = exp(-dt / m_tau_d);
	m_g_r = 0.; m_g_d = 0.;
	m_g = 0.;
};

void Exp2Cond::setParams(double tau_r, double tau_d){
	m_tau_r = tau_r; m_tau_d = tau_d;
	// set the normalization
	double tp = (m_tau_r * m_tau_d) / (m_tau_d - m_tau_r) * log(m_tau_d / m_tau_r);
	m_norm = 1. / (-exp(-tp / m_tau_r) + exp(-tp / m_tau_d));
};

void Exp2Cond::feedSpike(double g_max, int n_spike){
	m_g_r -= m_norm * g_max * double(n_spike);
	m_g_d += m_norm * g_max * double(n_spike);
	m_g = m_g_r + m_g_d;
};

double Exp2Cond::advance(double dt){
	if(abs(dt - m_dt) < 1.0e-9){
		m_p_r = exp(-dt / m_tau_r); m_p_d = exp(-dt / m_tau_d);
	}
	m_g_r *= m_p_r; m_g_d *= m_p_d;
	m_g = m_g_r + m_g_d;
	return m_g;
};

double Exp2Cond::getSurface(){
	return m_norm * (m_tau_r + m_tau_d);
};
////////////////////////////////////////////////////////////////////////////////


// driving force ///////////////////////////////////////////////////////////////
double DrivingForce::f(double v){
	return m_e_r - v;
};

double DrivingForce::DfDv(double v){
	return -1.;
};
////////////////////////////////////////////////////////////////////////////////


// NMDA synapse factor /////////////////////////////////////////////////////////
double NMDA::f(double v){
	return (m_e_r - v) / (1. + 0.3 * exp(-.1 * v));
};

double NMDA::DfDv(double v){
	return 0.03 * (m_e_r - v) * exp(-0.1 * v) / pow(0.3 * exp(-0.1*v) + 1.0, 2) 
			- 1. / (0.3 * exp(-0.1*v) + 1.0);
};
////////////////////////////////////////////////////////////////////////////////