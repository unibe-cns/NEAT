#include "Ionchannels.h"

void TestChannel::calcFunStatevar(double v){
    m_a01_inf = 1.0/(exp((30.0 - v)/100.0) + 1.0);
    m_tau_a01 = 2.0;
    m_a12_inf = -30.0;
    m_tau_a12 = 3.0;
    m_a11_inf = 2.0/(exp((30.0 - v)/100.0) + 1.0);
    m_tau_a11 = 2.0;
    m_a10_inf = 2.0/(exp((v - 30.0)/100.0) + 1.0);
    m_tau_a10 = 2.0;
    m_a02_inf = -10.0;
    m_tau_a02 = 1.0;
    m_a00_inf = 1.0/(exp((v - 30.0)/100.0) + 1.0);
    m_tau_a00 = 1.0;
}
double TestChannel::calcPOpen(){
    return 5*pow(m_a00, 3)*pow(m_a01, 3)*m_a02 + pow(m_a10, 2)*pow(m_a11, 2)*m_a12;
}
void TestChannel::setPOpen(){
    m_p_open = calcPOpen();
}
void TestChannel::setPOpenEQ(double v){
    calcFunStatevar(v);

    m_a01 = m_a01_inf;
    m_a12 = m_a12_inf;
    m_a11 = m_a11_inf;
    m_a10 = m_a10_inf;
    m_a02 = m_a02_inf;
    m_a00 = m_a00_inf;
    m_p_open_eq = 5*pow(m_a00_inf, 3)*pow(m_a01_inf, 3)*m_a02_inf + pow(m_a10_inf, 2)*pow(m_a11_inf, 2)*m_a12_inf;
}
void TestChannel::advance(double dt){
    double p0_a01 = exp(-dt / m_tau_a01);
    m_a01 *= p0_a01 ;
    m_a01 += (1. - p0_a01) *  m_a01_inf;
    double p0_a12 = exp(-dt / m_tau_a12);
    m_a12 *= p0_a12 ;
    m_a12 += (1. - p0_a12) *  m_a12_inf;
    double p0_a11 = exp(-dt / m_tau_a11);
    m_a11 *= p0_a11 ;
    m_a11 += (1. - p0_a11) *  m_a11_inf;
    double p0_a10 = exp(-dt / m_tau_a10);
    m_a10 *= p0_a10 ;
    m_a10 += (1. - p0_a10) *  m_a10_inf;
    double p0_a02 = exp(-dt / m_tau_a02);
    m_a02 *= p0_a02 ;
    m_a02 += (1. - p0_a02) *  m_a02_inf;
    double p0_a00 = exp(-dt / m_tau_a00);
    m_a00 *= p0_a00 ;
    m_a00 += (1. - p0_a00) *  m_a00_inf;
}
double TestChannel::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double TestChannel::getCondNewton(){
    return m_g_bar;
}
double TestChannel::f(double v){
    return (m_e_rev - v);
}
double TestChannel::DfDv(double v){
    return -1.;
}
void TestChannel::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 6)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_a00 = vs[0];
    m_v_a01 = vs[1];
    m_v_a02 = vs[2];
    m_v_a10 = vs[3];
    m_v_a11 = vs[4];
    m_v_a12 = vs[5];
}
double TestChannel::fNewton(double v){
    double v_a01;
    if(m_v_a01 > 1000.){
        v_a01 = v;
    } else{
        v_a01 = m_v_a01;
    }
    double a01 = 1.0/(exp((30.0 - v_a01)/100.0) + 1.0);
    double v_a12;
    if(m_v_a12 > 1000.){
        v_a12 = v;
    } else{
        v_a12 = m_v_a12;
    }
    double a12 = -30.0;
    double v_a11;
    if(m_v_a11 > 1000.){
        v_a11 = v;
    } else{
        v_a11 = m_v_a11;
    }
    double a11 = 2.0/(exp((30.0 - v_a11)/100.0) + 1.0);
    double v_a10;
    if(m_v_a10 > 1000.){
        v_a10 = v;
    } else{
        v_a10 = m_v_a10;
    }
    double a10 = 2.0/(exp((v_a10 - 30.0)/100.0) + 1.0);
    double v_a02;
    if(m_v_a02 > 1000.){
        v_a02 = v;
    } else{
        v_a02 = m_v_a02;
    }
    double a02 = -10.0;
    double v_a00;
    if(m_v_a00 > 1000.){
        v_a00 = v;
    } else{
        v_a00 = m_v_a00;
    }
    double a00 = 1.0/(exp((v_a00 - 30.0)/100.0) + 1.0);
    return (m_e_rev - v) * (5*pow(a00, 3)*pow(a01, 3)*a02 + pow(a10, 2)*pow(a11, 2)*a12 - m_p_open_eq);
}
double TestChannel::DfDvNewton(double v){
    double v_a01;
    double da01_dv;
    if(m_v_a01 > 1000.){
        v_a01 = v;
        da01_dv = 0.01*exp((30.0 - v_a01)/100.0)/pow(exp((30.0 - v_a01)/100.0) + 1.0, 2);
    } else{
        v_a01 = m_v_a01;
        da01_dv = 0;
    }
    double a01 = 1.0/(exp((30.0 - v_a01)/100.0) + 1.0);
    double v_a12;
    double da12_dv;
    if(m_v_a12 > 1000.){
        v_a12 = v;
        da12_dv = 0;
    } else{
        v_a12 = m_v_a12;
        da12_dv = 0;
    }
    double a12 = -30.0;
    double v_a11;
    double da11_dv;
    if(m_v_a11 > 1000.){
        v_a11 = v;
        da11_dv = 0.02*exp((30.0 - v_a11)/100.0)/pow(exp((30.0 - v_a11)/100.0) + 1.0, 2);
    } else{
        v_a11 = m_v_a11;
        da11_dv = 0;
    }
    double a11 = 2.0/(exp((30.0 - v_a11)/100.0) + 1.0);
    double v_a10;
    double da10_dv;
    if(m_v_a10 > 1000.){
        v_a10 = v;
        da10_dv = -0.02*exp((v_a10 - 30.0)/100.0)/pow(exp((v_a10 - 30.0)/100.0) + 1.0, 2);
    } else{
        v_a10 = m_v_a10;
        da10_dv = 0;
    }
    double a10 = 2.0/(exp((v_a10 - 30.0)/100.0) + 1.0);
    double v_a02;
    double da02_dv;
    if(m_v_a02 > 1000.){
        v_a02 = v;
        da02_dv = 0;
    } else{
        v_a02 = m_v_a02;
        da02_dv = 0;
    }
    double a02 = -10.0;
    double v_a00;
    double da00_dv;
    if(m_v_a00 > 1000.){
        v_a00 = v;
        da00_dv = -0.01*exp((v_a00 - 30.0)/100.0)/pow(exp((v_a00 - 30.0)/100.0) + 1.0, 2);
    } else{
        v_a00 = m_v_a00;
        da00_dv = 0;
    }
    double a00 = 1.0/(exp((v_a00 - 30.0)/100.0) + 1.0);
    return -1. * (5*pow(a00, 3)*pow(a01, 3)*a02 + pow(a10, 2)*pow(a11, 2)*a12 - m_p_open_eq) + (15*pow(a00, 3)*pow(a01, 2)*a02 * da01_dv + pow(a10, 2)*pow(a11, 2) * da12_dv + 2*pow(a10, 2)*a11*a12 * da11_dv + 2*a10*pow(a11, 2)*a12 * da10_dv + 5*pow(a00, 3)*pow(a01, 3) * da02_dv + 15*pow(a00, 2)*pow(a01, 3)*a02 * da00_dv) * (m_e_rev - v);
}

void TestChannel2::calcFunStatevar(double v){
    m_a11_inf = 0.59999999999999998;
    m_tau_a11 = 2.0;
    m_a10_inf = 0.40000000000000002;
    m_tau_a10 = 2.0;
    m_a01_inf = 0.5;
    m_tau_a01 = 2.0;
    m_a00_inf = 0.29999999999999999;
    m_tau_a00 = 1.0;
}
double TestChannel2::calcPOpen(){
    return 0.90000000000000002*pow(m_a00, 3)*pow(m_a01, 2) + 0.10000000000000001*pow(m_a10, 2)*m_a11;
}
void TestChannel2::setPOpen(){
    m_p_open = calcPOpen();
}
void TestChannel2::setPOpenEQ(double v){
    calcFunStatevar(v);

    m_a11 = m_a11_inf;
    m_a10 = m_a10_inf;
    m_a01 = m_a01_inf;
    m_a00 = m_a00_inf;
    m_p_open_eq = 0.90000000000000002*pow(m_a00_inf, 3)*pow(m_a01_inf, 2) + 0.10000000000000001*pow(m_a10_inf, 2)*m_a11_inf;
}
void TestChannel2::advance(double dt){
    double p0_a11 = exp(-dt / m_tau_a11);
    m_a11 *= p0_a11 ;
    m_a11 += (1. - p0_a11) *  m_a11_inf;
    double p0_a10 = exp(-dt / m_tau_a10);
    m_a10 *= p0_a10 ;
    m_a10 += (1. - p0_a10) *  m_a10_inf;
    double p0_a01 = exp(-dt / m_tau_a01);
    m_a01 *= p0_a01 ;
    m_a01 += (1. - p0_a01) *  m_a01_inf;
    double p0_a00 = exp(-dt / m_tau_a00);
    m_a00 *= p0_a00 ;
    m_a00 += (1. - p0_a00) *  m_a00_inf;
}
double TestChannel2::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double TestChannel2::getCondNewton(){
    return m_g_bar;
}
double TestChannel2::f(double v){
    return (m_e_rev - v);
}
double TestChannel2::DfDv(double v){
    return -1.;
}
void TestChannel2::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 4)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_a00 = vs[0];
    m_v_a01 = vs[1];
    m_v_a10 = vs[2];
    m_v_a11 = vs[3];
}
double TestChannel2::fNewton(double v){
    double v_a11;
    if(m_v_a11 > 1000.){
        v_a11 = v;
    } else{
        v_a11 = m_v_a11;
    }
    double a11 = 0.59999999999999998;
    double v_a10;
    if(m_v_a10 > 1000.){
        v_a10 = v;
    } else{
        v_a10 = m_v_a10;
    }
    double a10 = 0.40000000000000002;
    double v_a01;
    if(m_v_a01 > 1000.){
        v_a01 = v;
    } else{
        v_a01 = m_v_a01;
    }
    double a01 = 0.5;
    double v_a00;
    if(m_v_a00 > 1000.){
        v_a00 = v;
    } else{
        v_a00 = m_v_a00;
    }
    double a00 = 0.29999999999999999;
    return (m_e_rev - v) * (0.90000000000000002*pow(a00, 3)*pow(a01, 2) + 0.10000000000000001*pow(a10, 2)*a11 - m_p_open_eq);
}
double TestChannel2::DfDvNewton(double v){
    double v_a11;
    double da11_dv;
    if(m_v_a11 > 1000.){
        v_a11 = v;
        da11_dv = 0;
    } else{
        v_a11 = m_v_a11;
        da11_dv = 0;
    }
    double a11 = 0.59999999999999998;
    double v_a10;
    double da10_dv;
    if(m_v_a10 > 1000.){
        v_a10 = v;
        da10_dv = 0;
    } else{
        v_a10 = m_v_a10;
        da10_dv = 0;
    }
    double a10 = 0.40000000000000002;
    double v_a01;
    double da01_dv;
    if(m_v_a01 > 1000.){
        v_a01 = v;
        da01_dv = 0;
    } else{
        v_a01 = m_v_a01;
        da01_dv = 0;
    }
    double a01 = 0.5;
    double v_a00;
    double da00_dv;
    if(m_v_a00 > 1000.){
        v_a00 = v;
        da00_dv = 0;
    } else{
        v_a00 = m_v_a00;
        da00_dv = 0;
    }
    double a00 = 0.29999999999999999;
    return -1. * (0.90000000000000002*pow(a00, 3)*pow(a01, 2) + 0.10000000000000001*pow(a10, 2)*a11 - m_p_open_eq) + (0.10000000000000001*pow(a10, 2) * da11_dv + 0.20000000000000001*a10*a11 * da10_dv + 1.8*pow(a00, 3)*a01 * da01_dv + 2.7000000000000002*pow(a00, 2)*pow(a01, 2) * da00_dv) * (m_e_rev - v);
}

void Na_Ta::calcFunStatevar(double v){
    m_m_inf = 0.182*(v + 38.0)/((1.0 - exp((-v - 38.0)/6.0))*((-0.124)*(v + 38.0)/(1.0 - exp((v + 38.0)/6.0)) + 0.182*(v + 38.0)/(1.0 - exp((-v - 38.0)/6.0))));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
        m_tau_m = 0.33898305084745761/((-0.124)*(v + 38.0)/(1.0 - exp((v + 38.0)/6.0)) + 0.182*(v + 38.0)/(1.0 - exp((-v - 38.0)/6.0)));
    m_h_inf = -0.014999999999999999*(v + 66.0)/((1.0 - exp((v + 66.0)/6.0))*((-0.014999999999999999)*(v + 66.0)/(1.0 - exp((v + 66.0)/6.0)) + 0.014999999999999999*(v + 66.0)/(1.0 - exp((-v - 66.0)/6.0))));
    m_tau_h = 0.33898305084745761/((-0.014999999999999999)*(v + 66.0)/(1.0 - exp((v + 66.0)/6.0)) + 0.014999999999999999*(v + 66.0)/(1.0 - exp((-v - 66.0)/6.0)));
}
double Na_Ta::calcPOpen(){
    return m_h*pow(m_m, 3);
}
void Na_Ta::setPOpen(){
    m_p_open = calcPOpen();
}
void Na_Ta::setPOpenEQ(double v){
    calcFunStatevar(v);

    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq = m_h_inf*pow(m_m_inf, 3);
}
void Na_Ta::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h) *  m_h_inf;
}
double Na_Ta::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Na_Ta::getCondNewton(){
    return m_g_bar;
}
double Na_Ta::f(double v){
    return (m_e_rev - v);
}
double Na_Ta::DfDv(double v){
    return -1.;
}
void Na_Ta::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_h = vs[0];
    m_v_m = vs[1];
}
double Na_Ta::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 0.182*(v_m + 38.0)/((1.0 - exp((-v_m - 38.0)/6.0))*((-0.124)*(v_m + 38.0)/(1.0 - exp((v_m + 38.0)/6.0)) + 0.182*(v_m + 38.0)/(1.0 - exp((-v_m - 38.0)/6.0))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = -0.014999999999999999*(v_h + 66.0)/((1.0 - exp((v_h + 66.0)/6.0))*((-0.014999999999999999)*(v_h + 66.0)/(1.0 - exp((v_h + 66.0)/6.0)) + 0.014999999999999999*(v_h + 66.0)/(1.0 - exp((-v_h - 66.0)/6.0))));
    return (m_e_rev - v) * (h*pow(m, 3) - m_p_open_eq);
}
double Na_Ta::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.182*(v_m + 38.0)*(0.124/(1.0 - exp((v_m + 38.0)/6.0)) - 0.16666666666666666*(-0.124*v_m - 4.7119999999999997)*exp((v_m + 38.0)/6.0)/pow(1.0 - exp((v_m + 38.0)/6.0), 2) - 0.182/(1.0 - exp((-v_m - 38.0)/6.0)) + 0.16666666666666666*(0.182*v_m + 6.9159999999999995)*exp((-v_m - 38.0)/6.0)/pow(1.0 - exp((-v_m - 38.0)/6.0), 2))/((1.0 - exp((-v_m - 38.0)/6.0))*pow((-0.124)*(v_m + 38.0)/(1.0 - exp((v_m + 38.0)/6.0)) + 0.182*(v_m + 38.0)/(1.0 - exp((-v_m - 38.0)/6.0)), 2)) + 0.182/((1.0 - exp((-v_m - 38.0)/6.0))*((-0.124)*(v_m + 38.0)/(1.0 - exp((v_m + 38.0)/6.0)) + 0.182*(v_m + 38.0)/(1.0 - exp((-v_m - 38.0)/6.0)))) - 0.03033333333333333*(v_m + 38.0)*exp((-v_m - 38.0)/6.0)/(pow(1.0 - exp((-v_m - 38.0)/6.0), 2)*((-0.124)*(v_m + 38.0)/(1.0 - exp((v_m + 38.0)/6.0)) + 0.182*(v_m + 38.0)/(1.0 - exp((-v_m - 38.0)/6.0))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 0.182*(v_m + 38.0)/((1.0 - exp((-v_m - 38.0)/6.0))*((-0.124)*(v_m + 38.0)/(1.0 - exp((v_m + 38.0)/6.0)) + 0.182*(v_m + 38.0)/(1.0 - exp((-v_m - 38.0)/6.0))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -0.014999999999999999*(v_h + 66.0)*(0.014999999999999999/(1.0 - exp((v_h + 66.0)/6.0)) - 0.16666666666666666*(-0.014999999999999999*v_h - 0.98999999999999999)*exp((v_h + 66.0)/6.0)/pow(1.0 - exp((v_h + 66.0)/6.0), 2) - 0.014999999999999999/(1.0 - exp((-v_h - 66.0)/6.0)) + 0.16666666666666666*(0.014999999999999999*v_h + 0.98999999999999999)*exp((-v_h - 66.0)/6.0)/pow(1.0 - exp((-v_h - 66.0)/6.0), 2))/((1.0 - exp((v_h + 66.0)/6.0))*pow((-0.014999999999999999)*(v_h + 66.0)/(1.0 - exp((v_h + 66.0)/6.0)) + 0.014999999999999999*(v_h + 66.0)/(1.0 - exp((-v_h - 66.0)/6.0)), 2)) - 0.014999999999999999/((1.0 - exp((v_h + 66.0)/6.0))*((-0.014999999999999999)*(v_h + 66.0)/(1.0 - exp((v_h + 66.0)/6.0)) + 0.014999999999999999*(v_h + 66.0)/(1.0 - exp((-v_h - 66.0)/6.0)))) - 0.0024999999999999996*(v_h + 66.0)*exp((v_h + 66.0)/6.0)/(pow(1.0 - exp((v_h + 66.0)/6.0), 2)*((-0.014999999999999999)*(v_h + 66.0)/(1.0 - exp((v_h + 66.0)/6.0)) + 0.014999999999999999*(v_h + 66.0)/(1.0 - exp((-v_h - 66.0)/6.0))));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = -0.014999999999999999*(v_h + 66.0)/((1.0 - exp((v_h + 66.0)/6.0))*((-0.014999999999999999)*(v_h + 66.0)/(1.0 - exp((v_h + 66.0)/6.0)) + 0.014999999999999999*(v_h + 66.0)/(1.0 - exp((-v_h - 66.0)/6.0))));
    return -1. * (h*pow(m, 3) - m_p_open_eq) + (3*h*pow(m, 2) * dm_dv + pow(m, 3) * dh_dv) * (m_e_rev - v);
}

void Kv3_1::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp((18.699999999999999 - v)/9.6999999999999993) + 1.0);
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
        m_tau_m = 4.0/(exp((-v - 46.560000000000002)/44.140000000000001) + 1.0);
}
double Kv3_1::calcPOpen(){
    return m_m;
}
void Kv3_1::setPOpen(){
    m_p_open = calcPOpen();
}
void Kv3_1::setPOpenEQ(double v){
    calcFunStatevar(v);

    m_m = m_m_inf;
    m_p_open_eq = m_m_inf;
}
void Kv3_1::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m) *  m_m_inf;
}
double Kv3_1::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Kv3_1::getCondNewton(){
    return m_g_bar;
}
double Kv3_1::f(double v){
    return (m_e_rev - v);
}
double Kv3_1::DfDv(double v){
    return -1.;
}
void Kv3_1::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
}
double Kv3_1::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(exp((18.699999999999999 - v_m)/9.6999999999999993) + 1.0);
    return (m_e_rev - v) * (m - m_p_open_eq);
}
double Kv3_1::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.10309278350515465*exp((18.699999999999999 - v_m)/9.6999999999999993)/pow(exp((18.699999999999999 - v_m)/9.6999999999999993) + 1.0, 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(exp((18.699999999999999 - v_m)/9.6999999999999993) + 1.0);
    return -1. * (m - m_p_open_eq) + (1 * dm_dv) * (m_e_rev - v);
}

void SK::calcFunStatevar(double v){
    m_z_inf = 1.0/(pow(0.00042999999999999999/m_ca, 4.7999999999999998) + 1.0);
    m_tau_z = 1.0;
}
double SK::calcPOpen(){
    return m_z;
}
void SK::setPOpen(){
    m_p_open = calcPOpen();
}
void SK::setPOpenEQ(double v){
    calcFunStatevar(v);

    m_z = m_z_inf;
    m_p_open_eq = m_z_inf;
}
void SK::advance(double dt){
    double p0_z = exp(-dt / m_tau_z);
    m_z *= p0_z ;
    m_z += (1. - p0_z) *  m_z_inf;
}
double SK::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double SK::getCondNewton(){
    return m_g_bar;
}
double SK::f(double v){
    return (m_e_rev - v);
}
double SK::DfDv(double v){
    return -1.;
}
void SK::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_z = vs[0];
}
double SK::fNewton(double v){
    double v_z;
    if(m_v_z > 1000.){
        v_z = v;
    } else{
        v_z = m_v_z;
    }
    double z = 1.0/(pow(0.00042999999999999999/m_ca, 4.7999999999999998) + 1.0);
    return (m_e_rev - v) * (z - m_p_open_eq);
}
double SK::DfDvNewton(double v){
    double v_z;
    double dz_dv;
    if(m_v_z > 1000.){
        v_z = v;
        dz_dv = 0;
    } else{
        v_z = m_v_z;
        dz_dv = 0;
    }
    double z = 1.0/(pow(0.00042999999999999999/m_ca, 4.7999999999999998) + 1.0);
    return -1. * (z - m_p_open_eq) + (1 * dz_dv) * (m_e_rev - v);
}

void h::calcFunStatevar(double v){
    m_hf_inf = 1.0/(exp((v + 82.0)/7.0) + 1.0);
    m_tau_hf = 40.0;
    m_hs_inf = 1.0/(exp((v + 82.0)/7.0) + 1.0);
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
    m_p_open_eq = 0.80000000000000004*m_hf_inf + 0.20000000000000001*m_hs_inf;
}
void h::advance(double dt){
    double p0_hf = exp(-dt / m_tau_hf);
    m_hf *= p0_hf ;
    m_hf += (1. - p0_hf) *  m_hf_inf;
    double p0_hs = exp(-dt / m_tau_hs);
    m_hs *= p0_hs ;
    m_hs += (1. - p0_hs) *  m_hs_inf;
}
double h::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double h::getCondNewton(){
    return m_g_bar;
}
double h::f(double v){
    return (m_e_rev - v);
}
double h::DfDv(double v){
    return -1.;
}
void h::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_hf = vs[0];
    m_v_hs = vs[1];
}
double h::fNewton(double v){
    double v_hf;
    if(m_v_hf > 1000.){
        v_hf = v;
    } else{
        v_hf = m_v_hf;
    }
    double hf = 1.0/(exp((v_hf + 82.0)/7.0) + 1.0);
    double v_hs;
    if(m_v_hs > 1000.){
        v_hs = v;
    } else{
        v_hs = m_v_hs;
    }
    double hs = 1.0/(exp((v_hs + 82.0)/7.0) + 1.0);
    return (m_e_rev - v) * (0.80000000000000004*hf + 0.20000000000000001*hs - m_p_open_eq);
}
double h::DfDvNewton(double v){
    double v_hf;
    double dhf_dv;
    if(m_v_hf > 1000.){
        v_hf = v;
        dhf_dv = -0.14285714285714285*exp((v_hf + 82.0)/7.0)/pow(exp((v_hf + 82.0)/7.0) + 1.0, 2);
    } else{
        v_hf = m_v_hf;
        dhf_dv = 0;
    }
    double hf = 1.0/(exp((v_hf + 82.0)/7.0) + 1.0);
    double v_hs;
    double dhs_dv;
    if(m_v_hs > 1000.){
        v_hs = v;
        dhs_dv = -0.14285714285714285*exp((v_hs + 82.0)/7.0)/pow(exp((v_hs + 82.0)/7.0) + 1.0, 2);
    } else{
        v_hs = m_v_hs;
        dhs_dv = 0;
    }
    double hs = 1.0/(exp((v_hs + 82.0)/7.0) + 1.0);
    return -1. * (0.80000000000000004*hf + 0.20000000000000001*hs - m_p_open_eq) + (0.80000000000000004 * dhf_dv + 0.20000000000000001 * dhs_dv) * (m_e_rev - v);
}

