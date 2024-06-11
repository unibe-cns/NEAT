#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <complex>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <time.h>


// conductance windows /////////////////////////////////////////////////////////
class ConductanceWindow{
protected:
	double m_dt = 0.0;
	// conductance g
	double m_g = 0.0;


public:
	void init(){};
	virtual void setParams(){};
	virtual void setParams(double tau){};
	virtual void setParams(double tau_r, double tau_d){};
	virtual void reset(){};
	virtual void feedSpike(double g_max, int n_spike){};
	virtual double advance(double dt){return 0.0;};
	virtual double getSurface(){return 0.0;};
	double getCond(){return m_g;};
};

class ExpCond: public ConductanceWindow{
private:	// time scale window
	double m_tau = 3.;
	// propagator
	double m_p = 0.;

public:
	void init(double dt);
	void setParams(double tau) override;
	void reset() override {m_g = 0.0;};
	void feedSpike(double g_max, int n_spike) override;
	double advance(double dt) override;
	double getSurface() override;
};

class Exp2Cond: public ConductanceWindow{
	// conductance g
	double m_g_r = 0.0, m_g_d = 0.0;
	// time scales window
	double m_tau_r = .2, m_tau_d = 3.;
	double m_norm;
	// propagators
	double m_p_r = 0.0, m_p_d = 0.0;

public:
	void init(double dt);
	void setParams(double tau_r, double tau_d) override;
	void reset() override {m_g_r = 0.0; m_g_d = 0.0;};
	void feedSpike(double g_max, int n_spike) override;
	double advance(double dt) override;
	double getSurface() override;
};
////////////////////////////////////////////////////////////////////////////////


// voltage dependent factors////////////////////////////////////////////////////
class VoltageDependence{
protected:
	double m_e_r; // reversal potential

public:
	// contructors
	VoltageDependence(){m_e_r = 0.0;};
	VoltageDependence(double e_r){m_e_r = e_r;};
	// functions
	double getEr(){return m_e_r;};
	virtual double f(double v){return 1.0;};
	virtual double DfDv(double v){return 0.0;};
};

class DrivingForce: public VoltageDependence{
public:
	DrivingForce(double e_r) : VoltageDependence(e_r){};
	double f(double v) override;
	double DfDv(double v) override;
};

class NMDA: public VoltageDependence{
public:
	NMDA(double e_r) : VoltageDependence(e_r){};
	double f(double v) override;
	double DfDv(double v) override;

};
////////////////////////////////////////////////////////////////////////////////