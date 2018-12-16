#include "Ionchannels.h"

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

