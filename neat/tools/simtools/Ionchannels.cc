#include "Ionchannels.h"

void CaP::calcFunStatevar(double v){
    m_m_inf = 8.5/((35.0/(exp(0.068965517241379309*v + 5.1034482758620685) + 1.0) + 8.5/(exp(-0.080000000000000002*v + 0.64000000000000001) + 1.0))*(exp(-0.080000000000000002*v + 0.64000000000000001) + 1.0));
    m_tau_m = 1.0/(35.0/(exp(0.068965517241379309*v + 5.1034482758620685) + 1.0) + 8.5/(exp(-0.080000000000000002*v + 0.64000000000000001) + 1.0));
    m_h_inf = 0.0015/((0.0015/(exp(0.125*v + 3.625) + 1.0) + 0.0054999999999999997/(exp(-0.125*v - 2.875) + 1.0))*(exp(0.125*v + 3.625) + 1.0));
    m_tau_h = 1.0/(0.0015/(exp(0.125*v + 3.625) + 1.0) + 0.0054999999999999997/(exp(-0.125*v - 2.875) + 1.0));
}
double CaP::calcPOpen(){
    return 1.0*m_h*m_m;
}
void CaP::setPOpen(){
    m_p_open = calcPOpen();
}
void CaP::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*m_m_inf;
}
void CaP::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double CaP::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double CaP::f(double v){
    return (m_e_rev - v);
}
double CaP::DfDv(double v){
    return -1.;
}

void TestChannel::calcFunStatevar(double v){
    m_a00_inf = 1.0/(exp(0.01*v - 0.29999999999999999) + 1.0);
    m_tau_a00 = 1.0;
    m_a01_inf = 1.0/(exp(-0.01*v + 0.29999999999999999) + 1.0);
    m_tau_a01 = 2.0;
    m_a02_inf = -10.0;
    m_tau_a02 = 1.0;
    m_a10_inf = 2.0/(exp(0.01*v - 0.29999999999999999) + 1.0);
    m_tau_a10 = 2.0;
    m_a11_inf = 2.0/(exp(-0.01*v + 0.29999999999999999) + 1.0);
    m_tau_a11 = 2.0;
    m_a12_inf = -30.0;
    m_tau_a12 = 3.0;
}
double TestChannel::calcPOpen(){
    return 5.0*pow(m_a00, 3)*pow(m_a01, 3)*m_a02 + 1.0*pow(m_a10, 2)*pow(m_a11, 2)*m_a12;
}
void TestChannel::setPOpen(){
    m_p_open = calcPOpen();
}
void TestChannel::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_a00 = m_a00_inf;
    m_a01 = m_a01_inf;
    m_a02 = m_a02_inf;
    m_a10 = m_a10_inf;
    m_a11 = m_a11_inf;
    m_a12 = m_a12_inf;
    m_p_open_eq =5.0*pow(m_a00_inf, 3)*pow(m_a01_inf, 3)*m_a02_inf + 1.0*pow(m_a10_inf, 2)*pow(m_a11_inf, 2)*m_a12_inf;
}
void TestChannel::advance(double dt){
    double p0_a00 = exp(-dt / m_tau_a00);
    m_a00 *= p0_a00 ;
    m_a00 += (1. - p0_a00 ) *  m_a00_inf;
    double p0_a01 = exp(-dt / m_tau_a01);
    m_a01 *= p0_a01 ;
    m_a01 += (1. - p0_a01 ) *  m_a01_inf;
    double p0_a02 = exp(-dt / m_tau_a02);
    m_a02 *= p0_a02 ;
    m_a02 += (1. - p0_a02 ) *  m_a02_inf;
    double p0_a10 = exp(-dt / m_tau_a10);
    m_a10 *= p0_a10 ;
    m_a10 += (1. - p0_a10 ) *  m_a10_inf;
    double p0_a11 = exp(-dt / m_tau_a11);
    m_a11 *= p0_a11 ;
    m_a11 += (1. - p0_a11 ) *  m_a11_inf;
    double p0_a12 = exp(-dt / m_tau_a12);
    m_a12 *= p0_a12 ;
    m_a12 += (1. - p0_a12 ) *  m_a12_inf;
}
double TestChannel::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double TestChannel::f(double v){
    return (m_e_rev - v);
}
double TestChannel::DfDv(double v){
    return -1.;
}

void CaP2::calcFunStatevar(double v){
    m_m_inf = 8.5/((35/(exp(0.068965517241379309*v + 5.1034482758620685) + 1) + 8.5/(exp(-0.080000000000000002*v + 0.64000000000000001) + 1))*(exp(-0.080000000000000002*v + 0.64000000000000001) + 1));
    m_tau_m = 1.0/(35/(exp(0.068965517241379309*v + 5.1034482758620685) + 1) + 8.5/(exp(-0.080000000000000002*v + 0.64000000000000001) + 1));
}
double CaP2::calcPOpen(){
    return 1.0*m_m;
}
void CaP2::setPOpen(){
    m_p_open = calcPOpen();
}
void CaP2::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_p_open_eq =1.0*m_m_inf;
}
void CaP2::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
}
double CaP2::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double CaP2::f(double v){
    return (m_e_rev - v);
}
double CaP2::DfDv(double v){
    return -1.;
}

void Ktst::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.052631578947368418*v - 0.52631578947368418) + 1.0);
    m_tau_m = 0.11525423728813559 + 0.31186440677966104*exp(-pow(0.016949152542372881*v + 1.3728813559322033, 2));
    m_h_inf = 1.0/(exp(0.10000000000000001*v + 7.6000000000000005) + 1.0);
    m_tau_h = 2.7118644067796609 + 16.610169491525422*exp(-pow(0.043478260869565216*v + 3.6086956521739131, 2));
}
double Ktst::calcPOpen(){
    return 1.0*m_h*pow(m_m, 2);
}
void Ktst::setPOpen(){
    m_p_open = calcPOpen();
}
void Ktst::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 2);
}
void Ktst::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Ktst::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Ktst::f(double v){
    return (m_e_rev - v);
}
double Ktst::DfDv(double v){
    return -1.;
}

void KC3::calcFunStatevar(double v){
    m_m_inf = 7.5/(7.5 + 1.1522521003029376*exp(-0.067114093959731544*v));
    m_tau_m = 1.0/(7.5 + 1.1522521003029376*exp(-0.067114093959731544*v));
    m_z_inf = 0.00024993751562109475;
    m_tau_z = 10.0;
}
double KC3::calcPOpen(){
    return 1.0*m_m*pow(m_z, 2);
}
void KC3::setPOpen(){
    m_p_open = calcPOpen();
}
void KC3::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_z = m_z_inf;
    m_p_open_eq =1.0*m_m_inf*pow(m_z_inf, 2);
}
void KC3::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_z = exp(-dt / m_tau_z);
    m_z *= p0_z ;
    m_z += (1. - p0_z ) *  m_z_inf;
}
double KC3::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double KC3::f(double v){
    return (m_e_rev - v);
}
double KC3::DfDv(double v){
    return -1.;
}

void Ca_LVA::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.16666666666666666*v - 6.6666666666666661) + 1.0);
    m_tau_m = 5.0 + 6.7796610169491522/(exp(0.20000000000000001*v + 7.0) + 1.0);
    m_h_inf = 1.0/(exp(0.15625*v + 14.0625) + 1.0);
    m_tau_h = 20.0 + 16.949152542372879/(exp(0.14285714285714285*v + 7.1428571428571423) + 1.0);
}
double Ca_LVA::calcPOpen(){
    return 1.0*m_h*pow(m_m, 2);
}
void Ca_LVA::setPOpen(){
    m_p_open = calcPOpen();
}
void Ca_LVA::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 2);
}
void Ca_LVA::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Ca_LVA::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Ca_LVA::f(double v){
    return (m_e_rev - v);
}
double Ca_LVA::DfDv(double v){
    return -1.;
}

void NaF::calcFunStatevar(double v){
    m_m_inf = 35*exp((1.0/10.0)*v + 1.0/2.0)/(7*exp(-1.0/20.0*v - 13.0/4.0) + 35*exp((1.0/10.0)*v + 1.0/2.0));
    m_tau_m = 1.0/(7*exp(-1.0/20.0*v - 13.0/4.0) + 35*exp((1.0/10.0)*v + 1.0/2.0));
    m_h_inf = 0.22500000000000001/((7.5*exp((1.0/18.0)*v - 1.0/6.0) + 0.22500000000000001/(exp((1.0/10.0)*v + 8) + 1))*(exp((1.0/10.0)*v + 8) + 1));
    m_tau_h = 1.0/(7.5*exp((1.0/18.0)*v - 1.0/6.0) + 0.22500000000000001/(exp((1.0/10.0)*v + 8) + 1));
}
double NaF::calcPOpen(){
    return 1.0*m_h*pow(m_m, 3);
}
void NaF::setPOpen(){
    m_p_open = calcPOpen();
}
void NaF::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 3);
}
void NaF::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double NaF::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double NaF::f(double v){
    return (m_e_rev - v);
}
double NaF::DfDv(double v){
    return -1.;
}

void Na_Ta::calcFunStatevar(double v){
    m_m_inf = (0.182*v + 6.9159999999999995)/((1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v))*((-0.124*v - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v) + 1.0) + (0.182*v + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v))));
    m_tau_m = 0.33898305084745761/((-0.124*v - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v) + 1.0) + (0.182*v + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v)));
    m_h_inf = (-0.014999999999999999*v - 0.98999999999999999)/((-59874.141715197817*exp(0.16666666666666666*v) + 1.0)*((-0.014999999999999999*v - 0.98999999999999999)/(-59874.141715197817*exp(0.16666666666666666*v) + 1.0) + (0.014999999999999999*v + 0.98999999999999999)/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v))));
    m_tau_h = 0.33898305084745761/((-0.014999999999999999*v - 0.98999999999999999)/(-59874.141715197817*exp(0.16666666666666666*v) + 1.0) + (0.014999999999999999*v + 0.98999999999999999)/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v)));
}
double Na_Ta::calcPOpen(){
    return 1.0*m_h*pow(m_m, 3);
}
void Na_Ta::setPOpen(){
    m_p_open = calcPOpen();
}
void Na_Ta::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 3);
}
void Na_Ta::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Na_Ta::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Na_Ta::f(double v){
    return (m_e_rev - v);
}
double Na_Ta::DfDv(double v){
    return -1.;
}

void Kpst::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.083333333333333329*v - 0.91666666666666663) + 1.0);
    m_tau_m = 1.0305084745762711 + 5.8644067796610173*exp(-pow(0.062893081761006289*v + 3.7735849056603774, 2)) + 8.5423728813559308*exp(-pow(0.017421602787456445*v + 1.0452961672473866, 2));
    m_h_inf = 1.0/(exp(0.090909090909090912*v + 5.8181818181818183) + 1.0);
    m_tau_h = 0.33898305084745761*(24.0*v + 2570.0)*exp(-pow(0.020833333333333332*v + 1.7708333333333333, 2)) + 122.03389830508473;
}
double Kpst::calcPOpen(){
    return 1.0*m_h*pow(m_m, 2);
}
void Kpst::setPOpen(){
    m_p_open = calcPOpen();
}
void Kpst::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 2);
}
void Kpst::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Kpst::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Kpst::f(double v){
    return (m_e_rev - v);
}
double Kpst::DfDv(double v){
    return -1.;
}

void NaP::calcFunStatevar(double v){
    m_m_inf = 200.0/((25/(exp((1.0/8.0)*v + 29.0/4.0) + 1) + 200.0/(exp(-1.0/16.0*v + 9.0/8.0) + 1))*(exp(-1.0/16.0*v + 9.0/8.0) + 1));
    m_tau_m = 1.0/(25/(exp((1.0/8.0)*v + 29.0/4.0) + 1) + 200.0/(exp(-1.0/16.0*v + 9.0/8.0) + 1));
}
double NaP::calcPOpen(){
    return 1.0*pow(m_m, 3);
}
void NaP::setPOpen(){
    m_p_open = calcPOpen();
}
void NaP::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_p_open_eq =1.0*pow(m_m_inf, 3);
}
void NaP::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
}
double NaP::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double NaP::f(double v){
    return (m_e_rev - v);
}
double NaP::DfDv(double v){
    return -1.;
}

void Kv3_1::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.10309278350515465*v + 1.9278350515463918) + 1.0);
    m_tau_m = 4.0/(exp(-0.022655188038060714*v - 1.0548255550521068) + 1.0);
}
double Kv3_1::calcPOpen(){
    return 1.0*m_m;
}
void Kv3_1::setPOpen(){
    m_p_open = calcPOpen();
}
void Kv3_1::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_p_open_eq =1.0*m_m_inf;
}
void Kv3_1::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
}
double Kv3_1::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Kv3_1::f(double v){
    return (m_e_rev - v);
}
double Kv3_1::DfDv(double v){
    return -1.;
}

void K23::calcFunStatevar(double v){
    m_m_inf = 25.0/(25.0 + 0.045489799478447508*exp(-0.10000000000000001*v));
    m_tau_m = 1.0/(25.0 + 0.045489799478447508*exp(-0.10000000000000001*v));
    m_z_inf = 0.0049751243781094526;
    m_tau_z = 10.0;
}
double K23::calcPOpen(){
    return 1.0*m_m*pow(m_z, 2);
}
void K23::setPOpen(){
    m_p_open = calcPOpen();
}
void K23::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_z = m_z_inf;
    m_p_open_eq =1.0*m_m_inf*pow(m_z_inf, 2);
}
void K23::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_z = exp(-dt / m_tau_z);
    m_z *= p0_z ;
    m_z += (1. - p0_z ) *  m_z_inf;
}
double K23::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double K23::f(double v){
    return (m_e_rev - v);
}
double K23::DfDv(double v){
    return -1.;
}

void Kh::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp((1.0/7.0)*v + 78.0/7.0) + 1);
    m_tau_m = 38.0;
    m_n_inf = 1.0/(exp((1.0/7.0)*v + 78.0/7.0) + 1);
    m_tau_n = 319.0;
}
double Kh::calcPOpen(){
    return 0.80000000000000004*m_m + 0.20000000000000001*m_n;
}
void Kh::setPOpen(){
    m_p_open = calcPOpen();
}
void Kh::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_n = m_n_inf;
    m_p_open_eq =0.80000000000000004*m_m_inf + 0.20000000000000001*m_n_inf;
}
void Kh::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_n = exp(-dt / m_tau_n);
    m_n *= p0_n ;
    m_n += (1. - p0_n ) *  m_n_inf;
}
double Kh::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Kh::f(double v){
    return (m_e_rev - v);
}
double Kh::DfDv(double v){
    return -1.;
}

void Na_p::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.21739130434782611*v - 11.434782608695654) + 1.0);
    m_tau_m = 2.0338983050847457/((-0.124*v - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v) + 1.0) + (0.182*v + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v)));
    m_h_inf = 1.0/(exp(0.10000000000000001*v + 4.8799999999999999) + 1.0);
    m_tau_h = 0.33898305084745761/((-2.88e-6*v - 4.8959999999999999e-5)/(-39.318937124774365*exp(0.21598272138228941*v) + 1.0) + (6.9399999999999996e-6*v + 0.00044693599999999999)/(1.0 - 2.320410263420138e-11*exp(-0.38022813688212931*v)));
}
double Na_p::calcPOpen(){
    return 1.0*m_h*pow(m_m, 3);
}
void Na_p::setPOpen(){
    m_p_open = calcPOpen();
}
void Na_p::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 3);
}
void Na_p::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Na_p::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Na_p::f(double v){
    return (m_e_rev - v);
}
double Na_p::DfDv(double v){
    return -1.;
}

void KA::calcFunStatevar(double v){
    m_m_inf = 1.3999999999999999/((0.48999999999999999/(exp((1.0/4.0)*v + 15.0/2.0) + 1) + 1.3999999999999999/(exp(-1.0/12.0*v - 9.0/4.0) + 1))*(exp(-1.0/12.0*v - 9.0/4.0) + 1));
    m_tau_m = 1.0/(0.48999999999999999/(exp((1.0/4.0)*v + 15.0/2.0) + 1) + 1.3999999999999999/(exp(-1.0/12.0*v - 9.0/4.0) + 1));
    m_h_inf = 0.017500000000000002/((0.017500000000000002/(exp((1.0/8.0)*v + 25.0/4.0) + 1) + 1.3/(exp(-1.0/10.0*v - 13.0/10.0) + 1))*(exp((1.0/8.0)*v + 25.0/4.0) + 1));
    m_tau_h = 1.0/(0.017500000000000002/(exp((1.0/8.0)*v + 25.0/4.0) + 1) + 1.3/(exp(-1.0/10.0*v - 13.0/10.0) + 1));
}
double KA::calcPOpen(){
    return 1.0*m_h*pow(m_m, 4);
}
void KA::setPOpen(){
    m_p_open = calcPOpen();
}
void KA::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 4);
}
void KA::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double KA::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double KA::f(double v){
    return (m_e_rev - v);
}
double KA::DfDv(double v){
    return -1.;
}

void Khh::calcFunStatevar(double v){
    m_n_inf = (-0.01*v - 0.55000000000000004)/(((-0.01*v - 0.55000000000000004)/(exp(-0.10000000000000001*v - 5.5) - 1.0) + 0.125*0.44374731008107987*exp(-0.012500000000000001*v))*(exp(-0.10000000000000001*v - 5.5) - 1.0));
    m_tau_n = 1.0/((-0.01*v - 0.55000000000000004)/(exp(-0.10000000000000001*v - 5.5) - 1.0) + 0.125*0.44374731008107987*exp(-0.012500000000000001*v));
}
double Khh::calcPOpen(){
    return 1.0*pow(m_n, 4);
}
void Khh::setPOpen(){
    m_p_open = calcPOpen();
}
void Khh::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_n = m_n_inf;
    m_p_open_eq =1.0*pow(m_n_inf, 4);
}
void Khh::advance(double dt){
    double p0_n = exp(-dt / m_tau_n);
    m_n *= p0_n ;
    m_n += (1. - p0_n ) *  m_n_inf;
}
double Khh::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Khh::f(double v){
    return (m_e_rev - v);
}
double Khh::DfDv(double v){
    return -1.;
}

void Ca_HVA::calcFunStatevar(double v){
    m_m_inf = (-0.055*v - 1.4850000000000001)/(((-0.055*v - 1.4850000000000001)/(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0) + 0.012133746930834877*0.93999999999999995*exp(-0.058823529411764705*v))*(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0));
    m_tau_m = 1.0/((-0.055*v - 1.4850000000000001)/(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0) + 0.012133746930834877*0.93999999999999995*exp(-0.058823529411764705*v));
    m_h_inf = 0.00035237057471222975*exp(-0.02*v)/(0.000457*0.77105158580356625*exp(-0.02*v) + 0.0064999999999999997/(exp(-0.035714285714285712*v - 0.5357142857142857) + 1.0));
    m_tau_h = 1.0/(0.000457*0.77105158580356625*exp(-0.02*v) + 0.0064999999999999997/(exp(-0.035714285714285712*v - 0.5357142857142857) + 1.0));
}
double Ca_HVA::calcPOpen(){
    return 1.0*m_h*pow(m_m, 2);
}
void Ca_HVA::setPOpen(){
    m_p_open = calcPOpen();
}
void Ca_HVA::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 2);
}
void Ca_HVA::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Ca_HVA::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Ca_HVA::f(double v){
    return (m_e_rev - v);
}
double Ca_HVA::DfDv(double v){
    return -1.;
}

void KD::calcFunStatevar(double v){
    m_m_inf = 8.5/((35/(exp(0.068965517241379309*v + 6.8275862068965516) + 1) + 8.5/(exp(-0.080000000000000002*v - 1.3600000000000001) + 1))*(exp(-0.080000000000000002*v - 1.3600000000000001) + 1));
    m_tau_m = 1.0/(35/(exp(0.068965517241379309*v + 6.8275862068965516) + 1) + 8.5/(exp(-0.080000000000000002*v - 1.3600000000000001) + 1));
    m_h_inf = 0.0015/((0.0015/(exp((1.0/8.0)*v + 89.0/8.0) + 1) + 0.0054999999999999997/(exp(-1.0/8.0*v - 83.0/8.0) + 1))*(exp((1.0/8.0)*v + 89.0/8.0) + 1));
    m_tau_h = 1.0/(0.0015/(exp((1.0/8.0)*v + 89.0/8.0) + 1) + 0.0054999999999999997/(exp(-1.0/8.0*v - 83.0/8.0) + 1));
}
double KD::calcPOpen(){
    return 1.0*m_h*m_m;
}
void KD::setPOpen(){
    m_p_open = calcPOpen();
}
void KD::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*m_m_inf;
}
void KD::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double KD::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double KD::f(double v){
    return (m_e_rev - v);
}
double KD::DfDv(double v){
    return -1.;
}

void h::calcFunStatevar(double v){
    m_hf_inf = 1.0/(exp(0.14285714285714285*v + 11.714285714285714) + 1.0);
    m_tau_hf = 40.0;
    m_hs_inf = 1.0/(exp(0.14285714285714285*v + 11.714285714285714) + 1.0);
    m_tau_hs = 300.0;
}
double h::calcPOpen(){
    return 0.80000000000000004*m_hf + 0.20000000000000001*m_hs;
}
void h::setPOpen(){
    m_p_open = calcPOpen();
}
void h::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_hf = m_hf_inf;
    m_hs = m_hs_inf;
    m_p_open_eq =0.80000000000000004*m_hf_inf + 0.20000000000000001*m_hs_inf;
}
void h::advance(double dt){
    double p0_hf = exp(-dt / m_tau_hf);
    m_hf *= p0_hf ;
    m_hf += (1. - p0_hf ) *  m_hf_inf;
    double p0_hs = exp(-dt / m_tau_hs);
    m_hs *= p0_hs ;
    m_hs += (1. - p0_hs ) *  m_hs_inf;
}
double h::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double h::f(double v){
    return (m_e_rev - v);
}
double h::DfDv(double v){
    return -1.;
}

void m::calcFunStatevar(double v){
    m_m_inf = 0.10928099146368463*exp(0.10000000000000001*v)/(0.0033*33.115451958692312*exp(0.10000000000000001*v) + 0.0033*0.030197383422318501*exp(-0.10000000000000001*v));
    m_tau_m = 0.33898305084745761/(0.0033*33.115451958692312*exp(0.10000000000000001*v) + 0.0033*0.030197383422318501*exp(-0.10000000000000001*v));
}
double m::calcPOpen(){
    return 1.0*m_m;
}
void m::setPOpen(){
    m_p_open = calcPOpen();
}
void m::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_p_open_eq =1.0*m_m_inf;
}
void m::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
}
double m::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double m::f(double v){
    return (m_e_rev - v);
}
double m::DfDv(double v){
    return -1.;
}

void KM::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-1.0/10.0*v - 7.0/2.0) + 1);
    m_tau_m = 1.0/(0.016500000000000001*exp(-1.0/20.0*v - 7.0/4.0) + 0.016500000000000001*exp((1.0/40.0)*v + 7.0/8.0));
}
double KM::calcPOpen(){
    return 1.0*m_m;
}
void KM::setPOpen(){
    m_p_open = calcPOpen();
}
void KM::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_p_open_eq =1.0*m_m_inf;
}
void KM::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
}
double KM::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double KM::f(double v){
    return (m_e_rev - v);
}
double KM::DfDv(double v){
    return -1.;
}

void CaT::calcFunStatevar(double v){
    m_m_inf = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v + 10) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v - 21.0/8.0) + 1))*(exp(-1.0/8.0*v - 21.0/8.0) + 1));
    m_tau_m = 1.0/(0.17999999999999999/(exp((1.0/4.0)*v + 10) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v - 21.0/8.0) + 1));
    m_h_inf = 0.0025000000000000001/((0.0025000000000000001/(exp((1.0/8.0)*v + 5) + 1) + 0.19/(exp(-1.0/10.0*v - 5) + 1))*(exp((1.0/8.0)*v + 5) + 1));
    m_tau_h = 1.0/(0.0025000000000000001/(exp((1.0/8.0)*v + 5) + 1) + 0.19/(exp(-1.0/10.0*v - 5) + 1));
}
double CaT::calcPOpen(){
    return 1.0*m_h*m_m;
}
void CaT::setPOpen(){
    m_p_open = calcPOpen();
}
void CaT::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*m_m_inf;
}
void CaT::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double CaT::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double CaT::f(double v){
    return (m_e_rev - v);
}
double CaT::DfDv(double v){
    return -1.;
}

void SK::calcFunStatevar(double v){
    m_z_inf = 0.0009098213063332165;
    m_tau_z = 1.0;
}
double SK::calcPOpen(){
    return 1.0*m_z;
}
void SK::setPOpen(){
    m_p_open = calcPOpen();
}
void SK::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_z = m_z_inf;
    m_p_open_eq =1.0*m_z_inf;
}
void SK::advance(double dt){
    double p0_z = exp(-dt / m_tau_z);
    m_z *= p0_z ;
    m_z += (1. - p0_z ) *  m_z_inf;
}
double SK::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double SK::f(double v){
    return (m_e_rev - v);
}
double SK::DfDv(double v){
    return -1.;
}

void CaE::calcFunStatevar(double v){
    m_m_inf = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v + 13.0/2.0) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v - 7.0/8.0) + 1))*(exp(-1.0/8.0*v - 7.0/8.0) + 1));
    m_tau_m = 1.0/(0.17999999999999999/(exp((1.0/4.0)*v + 13.0/2.0) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v - 7.0/8.0) + 1));
    m_h_inf = 0.0025000000000000001/((0.0025000000000000001/(exp((1.0/8.0)*v + 4) + 1) + 0.19/(exp(-1.0/10.0*v - 21.0/5.0) + 1))*(exp((1.0/8.0)*v + 4) + 1));
    m_tau_h = 1.0/(0.0025000000000000001/(exp((1.0/8.0)*v + 4) + 1) + 0.19/(exp(-1.0/10.0*v - 21.0/5.0) + 1));
}
double CaE::calcPOpen(){
    return 1.0*m_h*m_m;
}
void CaE::setPOpen(){
    m_p_open = calcPOpen();
}
void CaE::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*m_m_inf;
}
void CaE::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double CaE::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double CaE::f(double v){
    return (m_e_rev - v);
}
double CaE::DfDv(double v){
    return -1.;
}

void h_HAY::calcFunStatevar(double v){
    m_m_inf = (0.00643*v + 0.99600699999999998)/(((0.00643*v + 0.99600699999999998)/(exp(0.084033613445378144*v + 13.016806722689076) - 1.0) + 0.193*exp(0.030211480362537763*v))*(exp(0.084033613445378144*v + 13.016806722689076) - 1.0));
    m_tau_m = 1.0/((0.00643*v + 0.99600699999999998)/(exp(0.084033613445378144*v + 13.016806722689076) - 1.0) + 0.193*exp(0.030211480362537763*v));
}
double h_HAY::calcPOpen(){
    return 1.0*m_m;
}
void h_HAY::setPOpen(){
    m_p_open = calcPOpen();
}
void h_HAY::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_p_open_eq =1.0*m_m_inf;
}
void h_HAY::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
}
double h_HAY::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double h_HAY::f(double v){
    return (m_e_rev - v);
}
double h_HAY::DfDv(double v){
    return -1.;
}

