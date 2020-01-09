#include "Ionchannels.h"

void CaP::calcFunStatevar(double v){
    m_m_inf = 8.5/((35.0/(exp(0.068965517241379309*v + 5.1034482758620685) + 1.0) + 8.5/(exp(-0.080000000000000002*v + 0.64000000000000001) + 1.0))*(exp(-0.080000000000000002*v + 0.64000000000000001) + 1.0));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double CaP::getCondNewton(){
    return m_g_bar;
}
double CaP::f(double v){
    return (m_e_rev - v);
}
double CaP::DfDv(double v){
    return -1.;
}
void CaP::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double CaP::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 8.5/((1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m))*(35.0/(164.58847636550695*exp(0.068965517241379309*v_m) + 1.0) + 8.5/(1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.0015/((0.0015/(37.524723159600995*exp(0.125*v_h) + 1.0) + 0.0054999999999999997/(1.0 + 0.05641613950377735*exp(-0.125*v_h)))*(37.524723159600995*exp(0.125*v_h) + 1.0));
    return (m_e_rev - v) * (1.0*h*m - m_p_open_eq);
}
double CaP::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 8.5*(397.28252915812021*exp(0.068965517241379309*v_m)/pow(164.58847636550695*exp(0.068965517241379309*v_m) + 1.0, 2) - 1.289606997927367*exp(-0.080000000000000002*v_m)/pow(1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m), 2))/((1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m))*pow(35.0/(164.58847636550695*exp(0.068965517241379309*v_m) + 1.0) + 8.5/(1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m)), 2)) + 1.289606997927367*exp(-0.080000000000000002*v_m)/(pow(1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m), 2)*(35.0/(164.58847636550695*exp(0.068965517241379309*v_m) + 1.0) + 8.5/(1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 8.5/((1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m))*(35.0/(164.58847636550695*exp(0.068965517241379309*v_m) + 1.0) + 8.5/(1.0 + 1.8964808793049515*exp(-0.080000000000000002*v_m))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.0015*(0.0070358855924251866*exp(0.125*v_h)/pow(37.524723159600995*exp(0.125*v_h) + 1.0, 2) - 3.8786095908846929e-5*exp(-0.125*v_h)/pow(1.0 + 0.05641613950377735*exp(-0.125*v_h), 2))/(pow(0.0015/(37.524723159600995*exp(0.125*v_h) + 1.0) + 0.0054999999999999997/(1.0 + 0.05641613950377735*exp(-0.125*v_h)), 2)*(37.524723159600995*exp(0.125*v_h) + 1.0)) - 0.0070358855924251866*exp(0.125*v_h)/((0.0015/(37.524723159600995*exp(0.125*v_h) + 1.0) + 0.0054999999999999997/(1.0 + 0.05641613950377735*exp(-0.125*v_h)))*pow(37.524723159600995*exp(0.125*v_h) + 1.0, 2));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.0015/((0.0015/(37.524723159600995*exp(0.125*v_h) + 1.0) + 0.0054999999999999997/(1.0 + 0.05641613950377735*exp(-0.125*v_h)))*(37.524723159600995*exp(0.125*v_h) + 1.0));
    return -1. * (1.0*h*m - m_p_open_eq) + (1.0*h * dm_dv+1.0*m * dh_dv) * (m_e_rev - v);
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
    double v_a00;
    if(m_v_a00 > 1000.){
        v_a00 = v;
    } else{
        v_a00 = m_v_a00;
    }
    double a00 = 1.0/(0.74081822068171788*exp(0.01*v_a00) + 1.0);
    double v_a01;
    if(m_v_a01 > 1000.){
        v_a01 = v;
    } else{
        v_a01 = m_v_a01;
    }
    double a01 = 1.0/(1.0 + 1.3498588075760032*exp(-0.01*v_a01));
    double v_a02;
    if(m_v_a02 > 1000.){
        v_a02 = v;
    } else{
        v_a02 = m_v_a02;
    }
    double a02 = -10.0;
    double v_a10;
    if(m_v_a10 > 1000.){
        v_a10 = v;
    } else{
        v_a10 = m_v_a10;
    }
    double a10 = 2.0/(0.74081822068171788*exp(0.01*v_a10) + 1.0);
    double v_a11;
    if(m_v_a11 > 1000.){
        v_a11 = v;
    } else{
        v_a11 = m_v_a11;
    }
    double a11 = 2.0/(1.0 + 1.3498588075760032*exp(-0.01*v_a11));
    double v_a12;
    if(m_v_a12 > 1000.){
        v_a12 = v;
    } else{
        v_a12 = m_v_a12;
    }
    double a12 = -30.0;
    return (m_e_rev - v) * (5.0*pow(a00, 3)*pow(a01, 3)*a02 + 1.0*pow(a10, 2)*pow(a11, 2)*a12 - m_p_open_eq);
}
double TestChannel::DfDvNewton(double v){
    double v_a00;
    double da00_dv;
    if(m_v_a00 > 1000.){
        v_a00 = v;
        da00_dv = -0.007408182206817179*exp(0.01*v_a00)/pow(0.74081822068171788*exp(0.01*v_a00) + 1.0, 2);
    } else{
        v_a00 = m_v_a00;
        da00_dv = 0;
    }
    double a00 = 1.0/(0.74081822068171788*exp(0.01*v_a00) + 1.0);
    double v_a01;
    double da01_dv;
    if(m_v_a01 > 1000.){
        v_a01 = v;
        da01_dv = 0.013498588075760033*exp(-0.01*v_a01)/pow(1.0 + 1.3498588075760032*exp(-0.01*v_a01), 2);
    } else{
        v_a01 = m_v_a01;
        da01_dv = 0;
    }
    double a01 = 1.0/(1.0 + 1.3498588075760032*exp(-0.01*v_a01));
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
    double v_a10;
    double da10_dv;
    if(m_v_a10 > 1000.){
        v_a10 = v;
        da10_dv = -0.014816364413634358*exp(0.01*v_a10)/pow(0.74081822068171788*exp(0.01*v_a10) + 1.0, 2);
    } else{
        v_a10 = m_v_a10;
        da10_dv = 0;
    }
    double a10 = 2.0/(0.74081822068171788*exp(0.01*v_a10) + 1.0);
    double v_a11;
    double da11_dv;
    if(m_v_a11 > 1000.){
        v_a11 = v;
        da11_dv = 0.026997176151520065*exp(-0.01*v_a11)/pow(1.0 + 1.3498588075760032*exp(-0.01*v_a11), 2);
    } else{
        v_a11 = m_v_a11;
        da11_dv = 0;
    }
    double a11 = 2.0/(1.0 + 1.3498588075760032*exp(-0.01*v_a11));
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
    return -1. * (5.0*pow(a00, 3)*pow(a01, 3)*a02 + 1.0*pow(a10, 2)*pow(a11, 2)*a12 - m_p_open_eq) + (15.0*pow(a00, 2)*pow(a01, 3)*a02 * da00_dv+15.0*pow(a00, 3)*pow(a01, 2)*a02 * da01_dv+5.0*pow(a00, 3)*pow(a01, 3) * da02_dv+2.0*a10*pow(a11, 2)*a12 * da10_dv+2.0*pow(a10, 2)*a11*a12 * da11_dv+1.0*pow(a10, 2)*pow(a11, 2) * da12_dv) * (m_e_rev - v);
}

void Na::calcFunStatevar(double v){
    m_m_inf = (0.182*v + 6.3723659999999995)/((1.0 - 0.020438532058318047*exp(-0.1111111111111111*v))*((-0.124*v - 4.3416119999999996)/(-48.927192870146527*exp(0.1111111111111111*v) + 1.0) + (0.182*v + 6.3723659999999995)/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v))));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
        m_tau_m = 0.3115264797507788/((-0.124*v - 4.3416119999999996)/(-48.927192870146527*exp(0.1111111111111111*v) + 1.0) + (0.182*v + 6.3723659999999995)/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v)));
    m_h_inf = 1.0/(exp(0.16129032258064516*v + 10.483870967741936) + 1.0);
    m_tau_h = 0.3115264797507788/((-0.0091000000000000004*v - 0.68261830000000012)/(-3277527.8765015295*exp(0.20000000000000001*v) + 1.0) + (0.024*v + 1.200312)/(1.0 - 4.5282043263959816e-5*exp(-0.20000000000000001*v)));
}
double Na::calcPOpen(){
    return 1.0*m_h*pow(m_m, 3);
}
void Na::setPOpen(){
    m_p_open = calcPOpen();
}
void Na::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 3);
}
void Na::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Na::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Na::getCondNewton(){
    return m_g_bar;
}
double Na::f(double v){
    return (m_e_rev - v);
}
double Na::DfDv(double v){
    return -1.;
}
void Na::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Na::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = (0.182*v_m + 6.3723659999999995)/((1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m))*((-0.124*v_m - 4.3416119999999996)/(-48.927192870146527*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 6.3723659999999995)/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(35734.467126792646*exp(0.16129032258064516*v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 3) - m_p_open_eq);
}
double Na::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = (0.182*v_m + 6.3723659999999995)*(-5.4363547633496134*(-0.124*v_m - 4.3416119999999996)*exp(0.1111111111111111*v_m)/pow(-48.927192870146527*exp(0.1111111111111111*v_m) + 1.0, 2) + 0.124/(-48.927192870146527*exp(0.1111111111111111*v_m) + 1.0) - 0.182/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m)) + 0.0022709480064797829*(0.182*v_m + 6.3723659999999995)*exp(-0.1111111111111111*v_m)/pow(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m), 2))/((1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m))*pow((-0.124*v_m - 4.3416119999999996)/(-48.927192870146527*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 6.3723659999999995)/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m)), 2)) + 0.182/((1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m))*((-0.124*v_m - 4.3416119999999996)/(-48.927192870146527*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 6.3723659999999995)/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m)))) - 0.0022709480064797829*(0.182*v_m + 6.3723659999999995)*exp(-0.1111111111111111*v_m)/(pow(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m), 2)*((-0.124*v_m - 4.3416119999999996)/(-48.927192870146527*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 6.3723659999999995)/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = (0.182*v_m + 6.3723659999999995)/((1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m))*((-0.124*v_m - 4.3416119999999996)/(-48.927192870146527*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 6.3723659999999995)/(1.0 - 0.020438532058318047*exp(-0.1111111111111111*v_m))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -5763.6237301278461*exp(0.16129032258064516*v_h)/pow(35734.467126792646*exp(0.16129032258064516*v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(35734.467126792646*exp(0.16129032258064516*v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 3) - m_p_open_eq) + (3.0*h*pow(m, 2) * dm_dv+1.0*pow(m, 3) * dh_dv) * (m_e_rev - v);
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
double CaP2::getCondNewton(){
    return m_g_bar;
}
double CaP2::f(double v){
    return (m_e_rev - v);
}
double CaP2::DfDv(double v){
    return -1.;
}
void CaP2::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
}
double CaP2::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 8.5/((1 + 1.8964808793049515*exp(-0.080000000000000002*v_m))*(35/(164.58847636550695*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 1.8964808793049515*exp(-0.080000000000000002*v_m))));
    return (m_e_rev - v) * (1.0*m - m_p_open_eq);
}
double CaP2::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 8.5*(397.28252915812021*exp(0.068965517241379309*v_m)/pow(164.58847636550695*exp(0.068965517241379309*v_m) + 1, 2) - 1.289606997927367*exp(-0.080000000000000002*v_m)/pow(1 + 1.8964808793049515*exp(-0.080000000000000002*v_m), 2))/((1 + 1.8964808793049515*exp(-0.080000000000000002*v_m))*pow(35/(164.58847636550695*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 1.8964808793049515*exp(-0.080000000000000002*v_m)), 2)) + 1.289606997927367*exp(-0.080000000000000002*v_m)/(pow(1 + 1.8964808793049515*exp(-0.080000000000000002*v_m), 2)*(35/(164.58847636550695*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 1.8964808793049515*exp(-0.080000000000000002*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 8.5/((1 + 1.8964808793049515*exp(-0.080000000000000002*v_m))*(35/(164.58847636550695*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 1.8964808793049515*exp(-0.080000000000000002*v_m))));
    return -1. * (1.0*m - m_p_open_eq) + (1.0 * dm_dv) * (m_e_rev - v);
}

void K_v::calcFunStatevar(double v){
    m_n_inf = (0.02*v - 0.5)/((1.0 - 16.083240672062946*exp(-0.1111111111111111*v))*((-0.002*v + 0.050000000000000003)/(-0.06217652402211632*exp(0.1111111111111111*v) + 1.0) + (0.02*v - 0.5)/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v))));
    m_tau_n = 1.0/(3.21*(-0.002*v + 0.050000000000000003)/(-0.06217652402211632*exp(0.1111111111111111*v) + 1.0) + 3.21*(0.02*v - 0.5)/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v)));
}
double K_v::calcPOpen(){
    return 1.0*m_n;
}
void K_v::setPOpen(){
    m_p_open = calcPOpen();
}
void K_v::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_n = m_n_inf;
    m_p_open_eq =1.0*m_n_inf;
}
void K_v::advance(double dt){
    double p0_n = exp(-dt / m_tau_n);
    m_n *= p0_n ;
    m_n += (1. - p0_n ) *  m_n_inf;
}
double K_v::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double K_v::getCondNewton(){
    return m_g_bar;
}
double K_v::f(double v){
    return (m_e_rev - v);
}
double K_v::DfDv(double v){
    return -1.;
}
void K_v::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_n = vs[0];
}
double K_v::fNewton(double v){
    double v_n;
    if(m_v_n > 1000.){
        v_n = v;
    } else{
        v_n = m_v_n;
    }
    double n = (0.02*v_n - 0.5)/((1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n))*((-0.002*v_n + 0.050000000000000003)/(-0.06217652402211632*exp(0.1111111111111111*v_n) + 1.0) + (0.02*v_n - 0.5)/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n))));
    return (m_e_rev - v) * (1.0*n - m_p_open_eq);
}
double K_v::DfDvNewton(double v){
    double v_n;
    double dn_dv;
    if(m_v_n > 1000.){
        v_n = v;
        dn_dv = (0.02*v_n - 0.5)*(-0.0069085026691240352*(-0.002*v_n + 0.050000000000000003)*exp(0.1111111111111111*v_n)/pow(-0.06217652402211632*exp(0.1111111111111111*v_n) + 1.0, 2) + 0.002/(-0.06217652402211632*exp(0.1111111111111111*v_n) + 1.0) - 0.02/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n)) + 1.7870267413403271*(0.02*v_n - 0.5)*exp(-0.1111111111111111*v_n)/pow(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n), 2))/((1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n))*pow((-0.002*v_n + 0.050000000000000003)/(-0.06217652402211632*exp(0.1111111111111111*v_n) + 1.0) + (0.02*v_n - 0.5)/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n)), 2)) + 0.02/((1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n))*((-0.002*v_n + 0.050000000000000003)/(-0.06217652402211632*exp(0.1111111111111111*v_n) + 1.0) + (0.02*v_n - 0.5)/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n)))) - 1.7870267413403271*(0.02*v_n - 0.5)*exp(-0.1111111111111111*v_n)/(pow(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n), 2)*((-0.002*v_n + 0.050000000000000003)/(-0.06217652402211632*exp(0.1111111111111111*v_n) + 1.0) + (0.02*v_n - 0.5)/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n))));
    } else{
        v_n = m_v_n;
        dn_dv = 0;
    }
    double n = (0.02*v_n - 0.5)/((1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n))*((-0.002*v_n + 0.050000000000000003)/(-0.06217652402211632*exp(0.1111111111111111*v_n) + 1.0) + (0.02*v_n - 0.5)/(1.0 - 16.083240672062946*exp(-0.1111111111111111*v_n))));
    return -1. * (1.0*n - m_p_open_eq) + (1.0 * dn_dv) * (m_e_rev - v);
}

void K_v_shift::calcFunStatevar(double v){
    m_n_inf = 1.0/(exp(-0.10309278350515465*v + 3.989690721649485) + 1.0);
    m_tau_n = 4.0/(exp(-0.022655188038060714*v - 0.60172179429089256) + 1.0);
}
double K_v_shift::calcPOpen(){
    return 1.0*m_n;
}
void K_v_shift::setPOpen(){
    m_p_open = calcPOpen();
}
void K_v_shift::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_n = m_n_inf;
    m_p_open_eq =1.0*m_n_inf;
}
void K_v_shift::advance(double dt){
    double p0_n = exp(-dt / m_tau_n);
    m_n *= p0_n ;
    m_n += (1. - p0_n ) *  m_n_inf;
}
double K_v_shift::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double K_v_shift::getCondNewton(){
    return m_g_bar;
}
double K_v_shift::f(double v){
    return (m_e_rev - v);
}
double K_v_shift::DfDv(double v){
    return -1.;
}
void K_v_shift::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_n = vs[0];
}
double K_v_shift::fNewton(double v){
    double v_n;
    if(m_v_n > 1000.){
        v_n = v;
    } else{
        v_n = m_v_n;
    }
    double n = 1.0/(1.0 + 54.038173941299348*exp(-0.10309278350515465*v_n));
    return (m_e_rev - v) * (1.0*n - m_p_open_eq);
}
double K_v_shift::DfDvNewton(double v){
    double v_n;
    double dn_dv;
    if(m_v_n > 1000.){
        v_n = v;
        dn_dv = 5.5709457671442628*exp(-0.10309278350515465*v_n)/pow(1.0 + 54.038173941299348*exp(-0.10309278350515465*v_n), 2);
    } else{
        v_n = m_v_n;
        dn_dv = 0;
    }
    double n = 1.0/(1.0 + 54.038173941299348*exp(-0.10309278350515465*v_n));
    return -1. * (1.0*n - m_p_open_eq) + (1.0 * dn_dv) * (m_e_rev - v);
}

void K_m35::calcFunStatevar(double v){
    m_n_inf = (0.001*v + 0.029999999999999999)/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v))*((-0.001*v - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v) + 1.0) + (0.001*v + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v))));
    m_tau_n = 1.0/(2.71*(-0.001*v - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v) + 1.0) + 2.71*(0.001*v + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v)));
}
double K_m35::calcPOpen(){
    return 1.0*m_n;
}
void K_m35::setPOpen(){
    m_p_open = calcPOpen();
}
void K_m35::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_n = m_n_inf;
    m_p_open_eq =1.0*m_n_inf;
}
void K_m35::advance(double dt){
    double p0_n = exp(-dt / m_tau_n);
    m_n *= p0_n ;
    m_n += (1. - p0_n ) *  m_n_inf;
}
double K_m35::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double K_m35::getCondNewton(){
    return m_g_bar;
}
double K_m35::f(double v){
    return (m_e_rev - v);
}
double K_m35::DfDv(double v){
    return -1.;
}
void K_m35::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_n = vs[0];
}
double K_m35::fNewton(double v){
    double v_n;
    if(m_v_n > 1000.){
        v_n = v;
    } else{
        v_n = m_v_n;
    }
    double n = (0.001*v_n + 0.029999999999999999)/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))));
    return (m_e_rev - v) * (1.0*n - m_p_open_eq);
}
double K_m35::DfDvNewton(double v){
    double v_n;
    double dn_dv;
    if(m_v_n > 1000.){
        v_n = v;
        dn_dv = (0.001*v_n + 0.029999999999999999)*(-3.1146249882806805*(-0.001*v_n - 0.029999999999999999)*exp(0.1111111111111111*v_n)/pow(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0, 2) + 0.001/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) - 0.001/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n)) + 0.0039637770385836006*(0.001*v_n + 0.029999999999999999)*exp(-0.1111111111111111*v_n)/pow(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n), 2))/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*pow((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n)), 2)) + 0.001/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n)))) - 0.0039637770385836006*(0.001*v_n + 0.029999999999999999)*exp(-0.1111111111111111*v_n)/(pow(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n), 2)*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))));
    } else{
        v_n = m_v_n;
        dn_dv = 0;
    }
    double n = (0.001*v_n + 0.029999999999999999)/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))));
    return -1. * (1.0*n - m_p_open_eq) + (1.0 * dn_dv) * (m_e_rev - v);
}

void h_u::calcFunStatevar(double v){
    m_q_inf = (0.00643*v + 0.99600699999999998)/(((0.00643*v + 0.99600699999999998)/(exp(0.084033613445378144*v + 13.016806722689076) - 1.0) + 0.193*exp(0.030211480362537763*v))*(exp(0.084033613445378144*v + 13.016806722689076) - 1.0));
    m_tau_q = 1.0/((0.00643*v + 0.99600699999999998)/(exp(0.084033613445378144*v + 13.016806722689076) - 1.0) + 0.193*exp(0.030211480362537763*v));
}
double h_u::calcPOpen(){
    return 1.0*m_q;
}
void h_u::setPOpen(){
    m_p_open = calcPOpen();
}
void h_u::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_q = m_q_inf;
    m_p_open_eq =1.0*m_q_inf;
}
void h_u::advance(double dt){
    double p0_q = exp(-dt / m_tau_q);
    m_q *= p0_q ;
    m_q += (1. - p0_q ) *  m_q_inf;
}
double h_u::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double h_u::getCondNewton(){
    return m_g_bar;
}
double h_u::f(double v){
    return (m_e_rev - v);
}
double h_u::DfDv(double v){
    return -1.;
}
void h_u::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_q = vs[0];
}
double h_u::fNewton(double v){
    double v_q;
    if(m_v_q > 1000.){
        v_q = v;
    } else{
        v_q = m_v_q;
    }
    double q = (0.00643*v_q + 0.99600699999999998)/(((0.00643*v_q + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0) + 0.193*exp(0.030211480362537763*v_q))*(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0));
    return (m_e_rev - v) * (1.0*q - m_p_open_eq);
}
double h_u::DfDvNewton(double v){
    double v_q;
    double dq_dv;
    if(m_v_q > 1000.){
        v_q = v;
        dq_dv = -37807.709754576565*(0.00643*v_q + 0.99600699999999998)*exp(0.084033613445378144*v_q)/(((0.00643*v_q + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0) + 0.193*exp(0.030211480362537763*v_q))*pow(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0, 2)) + (0.00643*v_q + 0.99600699999999998)*(37807.709754576565*(0.00643*v_q + 0.99600699999999998)*exp(0.084033613445378144*v_q)/pow(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0, 2) - 0.0058308157099697883*exp(0.030211480362537763*v_q) - 0.00643/(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0))/(pow((0.00643*v_q + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0) + 0.193*exp(0.030211480362537763*v_q), 2)*(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0)) + 0.00643/(((0.00643*v_q + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0) + 0.193*exp(0.030211480362537763*v_q))*(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0));
    } else{
        v_q = m_v_q;
        dq_dv = 0;
    }
    double q = (0.00643*v_q + 0.99600699999999998)/(((0.00643*v_q + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0) + 0.193*exp(0.030211480362537763*v_q))*(449911.74607946118*exp(0.084033613445378144*v_q) - 1.0));
    return -1. * (1.0*q - m_p_open_eq) + (1.0 * dq_dv) * (m_e_rev - v);
}

void Ktst::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.052631578947368418*v - 0.52631578947368418) + 1.0);
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double Ktst::getCondNewton(){
    return m_g_bar;
}
double Ktst::f(double v){
    return (m_e_rev - v);
}
double Ktst::DfDv(double v){
    return -1.;
}
void Ktst::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Ktst::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(1.0 + 0.59077751390123168*exp(-0.052631578947368418*v_m));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(1998.195895104119*exp(0.10000000000000001*v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 2) - m_p_open_eq);
}
double Ktst::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.031093553363222719*exp(-0.052631578947368418*v_m)/pow(1.0 + 0.59077751390123168*exp(-0.052631578947368418*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(1.0 + 0.59077751390123168*exp(-0.052631578947368418*v_m));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -199.8195895104119*exp(0.10000000000000001*v_h)/pow(1998.195895104119*exp(0.10000000000000001*v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(1998.195895104119*exp(0.10000000000000001*v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 2) - m_p_open_eq) + (2.0*h*m * dm_dv+1.0*pow(m, 2) * dh_dv) * (m_e_rev - v);
}

void K_ca::calcFunStatevar(double v){
    m_n_inf = 4.9997500124993755e-5;
    m_tau_n = 15.575545210278424;
}
double K_ca::calcPOpen(){
    return 1.0*m_n;
}
void K_ca::setPOpen(){
    m_p_open = calcPOpen();
}
void K_ca::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_n = m_n_inf;
    m_p_open_eq =1.0*m_n_inf;
}
void K_ca::advance(double dt){
    double p0_n = exp(-dt / m_tau_n);
    m_n *= p0_n ;
    m_n += (1. - p0_n ) *  m_n_inf;
}
double K_ca::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double K_ca::getCondNewton(){
    return m_g_bar;
}
double K_ca::f(double v){
    return (m_e_rev - v);
}
double K_ca::DfDv(double v){
    return -1.;
}
void K_ca::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_n = vs[0];
}
double K_ca::fNewton(double v){
    double v_n;
    if(m_v_n > 1000.){
        v_n = v;
    } else{
        v_n = m_v_n;
    }
    double n = 4.9997500124993755e-5;
    return (m_e_rev - v) * (1.0*n - m_p_open_eq);
}
double K_ca::DfDvNewton(double v){
    double v_n;
    double dn_dv;
    if(m_v_n > 1000.){
        v_n = v;
        dn_dv = 0;
    } else{
        v_n = m_v_n;
        dn_dv = 0;
    }
    double n = 4.9997500124993755e-5;
    return -1. * (1.0*n - m_p_open_eq) + (1.0 * dn_dv) * (m_e_rev - v);
}

void KC3::calcFunStatevar(double v){
    m_m_inf = 7.5/(7.5 + 1.1522521003029376*exp(-0.067114093959731544*v));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double KC3::getCondNewton(){
    return m_g_bar;
}
double KC3::f(double v){
    return (m_e_rev - v);
}
double KC3::DfDv(double v){
    return -1.;
}
void KC3::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_z = vs[1];
}
double KC3::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 7.5/(7.5 + 1.1522521003029376*exp(-0.067114093959731544*v_m));
    double v_z;
    if(m_v_z > 1000.){
        v_z = v;
    } else{
        v_z = m_v_z;
    }
    double z = 0.00024993751562109475;
    return (m_e_rev - v) * (1.0*m*pow(z, 2) - m_p_open_eq);
}
double KC3::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.5799926679377202*exp(-0.067114093959731544*v_m)/pow(7.5 + 1.1522521003029376*exp(-0.067114093959731544*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 7.5/(7.5 + 1.1522521003029376*exp(-0.067114093959731544*v_m));
    double v_z;
    double dz_dv;
    if(m_v_z > 1000.){
        v_z = v;
        dz_dv = 0;
    } else{
        v_z = m_v_z;
        dz_dv = 0;
    }
    double z = 0.00024993751562109475;
    return -1. * (1.0*m*pow(z, 2) - m_p_open_eq) + (1.0*pow(z, 2) * dm_dv+2.0*m*z * dz_dv) * (m_e_rev - v);
}

void Ca_LVA::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.16666666666666666*v - 6.6666666666666661) + 1.0);
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double Ca_LVA::getCondNewton(){
    return m_g_bar;
}
double Ca_LVA::f(double v){
    return (m_e_rev - v);
}
double Ca_LVA::DfDv(double v){
    return -1.;
}
void Ca_LVA::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Ca_LVA::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(1.0 + 0.001272633801339809*exp(-0.16666666666666666*v_m));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(1280165.5967642837*exp(0.15625*v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 2) - m_p_open_eq);
}
double Ca_LVA::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.00021210563355663481*exp(-0.16666666666666666*v_m)/pow(1.0 + 0.001272633801339809*exp(-0.16666666666666666*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(1.0 + 0.001272633801339809*exp(-0.16666666666666666*v_m));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -200025.87449441932*exp(0.15625*v_h)/pow(1280165.5967642837*exp(0.15625*v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(1280165.5967642837*exp(0.15625*v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 2) - m_p_open_eq) + (2.0*h*m * dm_dv+1.0*pow(m, 2) * dh_dv) * (m_e_rev - v);
}

void Ca_R::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.33333333333333331*v - 20.0) + 1.0);
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
        m_tau_m = 100.0;
    m_h_inf = 1.0/(exp(v + 62.0) + 1.0);
    m_tau_h = 5.0;
}
double Ca_R::calcPOpen(){
    return 1.0*m_h*pow(m_m, 3);
}
void Ca_R::setPOpen(){
    m_p_open = calcPOpen();
}
void Ca_R::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 3);
}
void Ca_R::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Ca_R::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Ca_R::getCondNewton(){
    return m_g_bar;
}
double Ca_R::f(double v){
    return (m_e_rev - v);
}
double Ca_R::DfDv(double v){
    return -1.;
}
void Ca_R::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Ca_R::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(1.0 + 2.0611536224385579e-9*exp(-0.33333333333333331*v_m));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(8.4383566687414538e+26*exp(v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 3) - m_p_open_eq);
}
double Ca_R::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 6.8705120747951929e-10*exp(-0.33333333333333331*v_m)/pow(1.0 + 2.0611536224385579e-9*exp(-0.33333333333333331*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(1.0 + 2.0611536224385579e-9*exp(-0.33333333333333331*v_m));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -8.4383566687414538e+26*exp(v_h)/pow(8.4383566687414538e+26*exp(v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(8.4383566687414538e+26*exp(v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 3) - m_p_open_eq) + (3.0*h*pow(m, 2) * dm_dv+1.0*pow(m, 3) * dh_dv) * (m_e_rev - v);
}

void NaF::calcFunStatevar(double v){
    m_m_inf = 35*exp((1.0/10.0)*v + 1.0/2.0)/(7*exp(-1.0/20.0*v - 13.0/4.0) + 35*exp((1.0/10.0)*v + 1.0/2.0));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double NaF::getCondNewton(){
    return m_g_bar;
}
double NaF::f(double v){
    return (m_e_rev - v);
}
double NaF::DfDv(double v){
    return -1.;
}
void NaF::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double NaF::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 35*exp((1.0/10.0)*v_m + 1.0/2.0)/(7*exp(-1.0/20.0*v_m - 13.0/4.0) + 35*exp((1.0/10.0)*v_m + 1.0/2.0));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.22500000000000001/((7.5*exp((1.0/18.0)*v_h - 1.0/6.0) + 0.22500000000000001/(exp((1.0/10.0)*v_h + 8) + 1))*(exp((1.0/10.0)*v_h + 8) + 1));
    return (m_e_rev - v) * (1.0*h*pow(m, 3) - m_p_open_eq);
}
double NaF::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 35*((7.0/20.0)*exp(-1.0/20.0*v_m - 13.0/4.0) - 7.0/2.0*exp((1.0/10.0)*v_m + 1.0/2.0))*exp((1.0/10.0)*v_m + 1.0/2.0)/pow(7*exp(-1.0/20.0*v_m - 13.0/4.0) + 35*exp((1.0/10.0)*v_m + 1.0/2.0), 2) + (7.0/2.0)*exp((1.0/10.0)*v_m + 1.0/2.0)/(7*exp(-1.0/20.0*v_m - 13.0/4.0) + 35*exp((1.0/10.0)*v_m + 1.0/2.0));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 35*exp((1.0/10.0)*v_m + 1.0/2.0)/(7*exp(-1.0/20.0*v_m - 13.0/4.0) + 35*exp((1.0/10.0)*v_m + 1.0/2.0));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.22500000000000001*(-0.41666666666666663*exp((1.0/18.0)*v_h - 1.0/6.0) + 0.022500000000000003*exp((1.0/10.0)*v_h + 8)/pow(exp((1.0/10.0)*v_h + 8) + 1, 2))/(pow(7.5*exp((1.0/18.0)*v_h - 1.0/6.0) + 0.22500000000000001/(exp((1.0/10.0)*v_h + 8) + 1), 2)*(exp((1.0/10.0)*v_h + 8) + 1)) - 0.022500000000000003*exp((1.0/10.0)*v_h + 8)/((7.5*exp((1.0/18.0)*v_h - 1.0/6.0) + 0.22500000000000001/(exp((1.0/10.0)*v_h + 8) + 1))*pow(exp((1.0/10.0)*v_h + 8) + 1, 2));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.22500000000000001/((7.5*exp((1.0/18.0)*v_h - 1.0/6.0) + 0.22500000000000001/(exp((1.0/10.0)*v_h + 8) + 1))*(exp((1.0/10.0)*v_h + 8) + 1));
    return -1. * (1.0*h*pow(m, 3) - m_p_open_eq) + (3.0*h*pow(m, 2) * dm_dv+1.0*pow(m, 3) * dh_dv) * (m_e_rev - v);
}

void Ca_T::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.13513513513513511*v - 6.7567567567567561) + 1.0);
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
        m_tau_m = 3.0 + 1.0/(exp(-0.066666666666666666*v - 6.666666666666667) + exp(0.050000000000000003*v + 1.25));
    m_h_inf = 1.0/(exp(0.20000000000000001*v + 15.600000000000001) + 1.0);
    m_tau_h = 85.0 + 1.0/(exp(-0.02*v - 8.0999999999999996) + exp(0.25*v + 11.5));
}
double Ca_T::calcPOpen(){
    return 1.0*m_h*pow(m_m, 2);
}
void Ca_T::setPOpen(){
    m_p_open = calcPOpen();
}
void Ca_T::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 2);
}
void Ca_T::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Ca_T::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Ca_T::getCondNewton(){
    return m_g_bar;
}
double Ca_T::f(double v){
    return (m_e_rev - v);
}
double Ca_T::DfDv(double v){
    return -1.;
}
void Ca_T::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Ca_T::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(1.0 + 0.0011629949394361468*exp(-0.13513513513513511*v_m));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(5956538.0131846247*exp(0.20000000000000001*v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 2) - m_p_open_eq);
}
double Ca_T::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.00015716147830218197*exp(-0.13513513513513511*v_m)/pow(1.0 + 0.0011629949394361468*exp(-0.13513513513513511*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(1.0 + 0.0011629949394361468*exp(-0.13513513513513511*v_m));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -1191307.6026369249*exp(0.20000000000000001*v_h)/pow(5956538.0131846247*exp(0.20000000000000001*v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(5956538.0131846247*exp(0.20000000000000001*v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 2) - m_p_open_eq) + (2.0*h*m * dm_dv+1.0*pow(m, 2) * dh_dv) * (m_e_rev - v);
}

void Na_Ta::calcFunStatevar(double v){
    m_m_inf = (0.182*v + 6.9159999999999995)/((1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v))*((-0.124*v - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v) + 1.0) + (0.182*v + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v))));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Na_Ta::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = (0.182*v_m + 6.9159999999999995)/((1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m))*((-0.124*v_m - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v_m) + 1.0) + (0.182*v_m + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = (-0.014999999999999999*v_h - 0.98999999999999999)/(((-0.014999999999999999*v_h - 0.98999999999999999)/(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0) + (0.014999999999999999*v_h + 0.98999999999999999)/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v_h)))*(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0));
    return (m_e_rev - v) * (1.0*h*pow(m, 3) - m_p_open_eq);
}
double Na_Ta::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = (0.182*v_m + 6.9159999999999995)*(-93.838372805991838*(-0.124*v_m - 4.7119999999999997)*exp(0.16666666666666666*v_m)/pow(-563.03023683595109*exp(0.16666666666666666*v_m) + 1.0, 2) + 0.124/(-563.03023683595109*exp(0.16666666666666666*v_m) + 1.0) - 0.182/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m)) + 0.00029601725762239649*(0.182*v_m + 6.9159999999999995)*exp(-0.16666666666666666*v_m)/pow(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m), 2))/((1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m))*pow((-0.124*v_m - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v_m) + 1.0) + (0.182*v_m + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m)), 2)) + 0.182/((1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m))*((-0.124*v_m - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v_m) + 1.0) + (0.182*v_m + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m)))) - 0.00029601725762239649*(0.182*v_m + 6.9159999999999995)*exp(-0.16666666666666666*v_m)/(pow(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m), 2)*((-0.124*v_m - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v_m) + 1.0) + (0.182*v_m + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = (0.182*v_m + 6.9159999999999995)/((1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m))*((-0.124*v_m - 4.7119999999999997)/(-563.03023683595109*exp(0.16666666666666666*v_m) + 1.0) + (0.182*v_m + 6.9159999999999995)/(1.0 - 0.0017761035457343791*exp(-0.16666666666666666*v_m))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 9979.0236191996355*(-0.014999999999999999*v_h - 0.98999999999999999)*exp(0.16666666666666666*v_h)/(((-0.014999999999999999*v_h - 0.98999999999999999)/(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0) + (0.014999999999999999*v_h + 0.98999999999999999)/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v_h)))*pow(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0, 2)) + (-0.014999999999999999*v_h - 0.98999999999999999)*(-9979.0236191996355*(-0.014999999999999999*v_h - 0.98999999999999999)*exp(0.16666666666666666*v_h)/pow(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0, 2) + 0.014999999999999999/(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0) - 0.014999999999999999/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v_h)) + 2.7836167983742764e-6*(0.014999999999999999*v_h + 0.98999999999999999)*exp(-0.16666666666666666*v_h)/pow(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v_h), 2))/(pow((-0.014999999999999999*v_h - 0.98999999999999999)/(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0) + (0.014999999999999999*v_h + 0.98999999999999999)/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v_h)), 2)*(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0)) - 0.014999999999999999/(((-0.014999999999999999*v_h - 0.98999999999999999)/(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0) + (0.014999999999999999*v_h + 0.98999999999999999)/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v_h)))*(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = (-0.014999999999999999*v_h - 0.98999999999999999)/(((-0.014999999999999999*v_h - 0.98999999999999999)/(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0) + (0.014999999999999999*v_h + 0.98999999999999999)/(1.0 - 1.6701700790245659e-5*exp(-0.16666666666666666*v_h)))*(-59874.141715197817*exp(0.16666666666666666*v_h) + 1.0));
    return -1. * (1.0*h*pow(m, 3) - m_p_open_eq) + (3.0*h*pow(m, 2) * dm_dv+1.0*pow(m, 3) * dh_dv) * (m_e_rev - v);
}

void Kpst::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.083333333333333329*v - 0.91666666666666663) + 1.0);
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double Kpst::getCondNewton(){
    return m_g_bar;
}
double Kpst::f(double v){
    return (m_e_rev - v);
}
double Kpst::DfDv(double v){
    return -1.;
}
void Kpst::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Kpst::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(1.0 + 0.39984965434484737*exp(-0.083333333333333329*v_m));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(336.35993381011735*exp(0.090909090909090912*v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 2) - m_p_open_eq);
}
double Kpst::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.033320804528737279*exp(-0.083333333333333329*v_m)/pow(1.0 + 0.39984965434484737*exp(-0.083333333333333329*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(1.0 + 0.39984965434484737*exp(-0.083333333333333329*v_m));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -30.578175800919759*exp(0.090909090909090912*v_h)/pow(336.35993381011735*exp(0.090909090909090912*v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(336.35993381011735*exp(0.090909090909090912*v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 2) - m_p_open_eq) + (2.0*h*m * dm_dv+1.0*pow(m, 2) * dh_dv) * (m_e_rev - v);
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
double NaP::getCondNewton(){
    return m_g_bar;
}
double NaP::f(double v){
    return (m_e_rev - v);
}
double NaP::DfDv(double v){
    return -1.;
}
void NaP::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
}
double NaP::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 200.0/((25/(exp((1.0/8.0)*v_m + 29.0/4.0) + 1) + 200.0/(exp(-1.0/16.0*v_m + 9.0/8.0) + 1))*(exp(-1.0/16.0*v_m + 9.0/8.0) + 1));
    return (m_e_rev - v) * (1.0*pow(m, 3) - m_p_open_eq);
}
double NaP::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 200.0*((25.0/8.0)*exp((1.0/8.0)*v_m + 29.0/4.0)/pow(exp((1.0/8.0)*v_m + 29.0/4.0) + 1, 2) - 12.5*exp(-1.0/16.0*v_m + 9.0/8.0)/pow(exp(-1.0/16.0*v_m + 9.0/8.0) + 1, 2))/(pow(25/(exp((1.0/8.0)*v_m + 29.0/4.0) + 1) + 200.0/(exp(-1.0/16.0*v_m + 9.0/8.0) + 1), 2)*(exp(-1.0/16.0*v_m + 9.0/8.0) + 1)) + 12.5*exp(-1.0/16.0*v_m + 9.0/8.0)/((25/(exp((1.0/8.0)*v_m + 29.0/4.0) + 1) + 200.0/(exp(-1.0/16.0*v_m + 9.0/8.0) + 1))*pow(exp(-1.0/16.0*v_m + 9.0/8.0) + 1, 2));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 200.0/((25/(exp((1.0/8.0)*v_m + 29.0/4.0) + 1) + 200.0/(exp(-1.0/16.0*v_m + 9.0/8.0) + 1))*(exp(-1.0/16.0*v_m + 9.0/8.0) + 1));
    return -1. * (1.0*pow(m, 3) - m_p_open_eq) + (3.0*pow(m, 2) * dm_dv) * (m_e_rev - v);
}

void Ca_H::calcFunStatevar(double v){
    m_m_inf = (-0.055*v - 1.4850000000000001)/(((-0.055*v - 1.4850000000000001)/(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0) + 0.012133746930834877*0.93999999999999995*exp(-0.058823529411764705*v))*(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
        m_tau_m = 0.3115264797507788/((-0.055*v - 1.4850000000000001)/(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0) + 0.012133746930834877*0.93999999999999995*exp(-0.058823529411764705*v));
    m_h_inf = 0.00035237057471222975*exp(-0.02*v)/(0.000457*0.77105158580356625*exp(-0.02*v) + 0.0064999999999999997/(exp(-0.035714285714285712*v - 0.5357142857142857) + 1.0));
    m_tau_h = 0.3115264797507788/(0.000457*0.77105158580356625*exp(-0.02*v) + 0.0064999999999999997/(exp(-0.035714285714285712*v - 0.5357142857142857) + 1.0));
}
double Ca_H::calcPOpen(){
    return 1.0*m_h*pow(m_m, 2);
}
void Ca_H::setPOpen(){
    m_p_open = calcPOpen();
}
void Ca_H::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 2);
}
void Ca_H::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Ca_H::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Ca_H::getCondNewton(){
    return m_g_bar;
}
double Ca_H::f(double v){
    return (m_e_rev - v);
}
double Ca_H::DfDv(double v){
    return -1.;
}
void Ca_H::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Ca_H::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = (-0.055*v_m - 1.4850000000000001)/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.00035237057471222975*exp(-0.02*v_h)/(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)));
    return (m_e_rev - v) * (1.0*h*pow(m, 2) - m_p_open_eq);
}
double Ca_H::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = (-0.055*v_m - 1.4850000000000001)*(0.00067092483029322256*exp(-0.058823529411764705*v_m) + 0.055/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m)) - 0.00021599307205216037*(-0.055*v_m - 1.4850000000000001)*exp(-0.26315789473684209*v_m)/pow(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m), 2))/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*pow(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m)), 2)) - 0.055/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m)))) + 0.00021599307205216037*(-0.055*v_m - 1.4850000000000001)*exp(-0.26315789473684209*v_m)/(pow(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m), 2)*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = (-0.055*v_m - 1.4850000000000001)/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.00035237057471222975*(7.0474114942445952e-6*exp(-0.02*v_h) - 0.000135861863499935*exp(-0.035714285714285712*v_h)/pow(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h), 2))*exp(-0.02*v_h)/pow(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)), 2) - 7.0474114942445952e-6*exp(-0.02*v_h)/(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.00035237057471222975*exp(-0.02*v_h)/(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)));
    return -1. * (1.0*h*pow(m, 2) - m_p_open_eq) + (2.0*h*m * dm_dv+1.0*pow(m, 2) * dh_dv) * (m_e_rev - v);
}

void H_distal::calcFunStatevar(double v){
    m_l_inf = exp(0.125*v + 10.125) + 1.0;
    m_tau_l = (12.11930941430424*exp(0.033264000000000009*v))/(0.02002*511.32224096716681*exp(0.083160000000000012*v) + 0.02002);
}
double H_distal::calcPOpen(){
    return 1.0*m_l;
}
void H_distal::setPOpen(){
    m_p_open = calcPOpen();
}
void H_distal::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_l = m_l_inf;
    m_p_open_eq =1.0*m_l_inf;
}
void H_distal::advance(double dt){
    double p0_l = exp(-dt / m_tau_l);
    m_l *= p0_l ;
    m_l += (1. - p0_l ) *  m_l_inf;
}
double H_distal::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double H_distal::getCondNewton(){
    return m_g_bar;
}
double H_distal::f(double v){
    return (m_e_rev - v);
}
double H_distal::DfDv(double v){
    return -1.;
}
void H_distal::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_l = vs[0];
}
double H_distal::fNewton(double v){
    double v_l;
    if(m_v_l > 1000.){
        v_l = v;
    } else{
        v_l = m_v_l;
    }
    double l = 24959.255641914595*exp(0.125*v_l) + 1.0;
    return (m_e_rev - v) * (1.0*l - m_p_open_eq);
}
double H_distal::DfDvNewton(double v){
    double v_l;
    double dl_dv;
    if(m_v_l > 1000.){
        v_l = v;
        dl_dv = 3119.9069552393244*exp(0.125*v_l);
    } else{
        v_l = m_v_l;
        dl_dv = 0;
    }
    double l = 24959.255641914595*exp(0.125*v_l) + 1.0;
    return -1. * (1.0*l - m_p_open_eq) + (1.0 * dl_dv) * (m_e_rev - v);
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
    double m = 1.0/(1.0 + 6.8746109409659972*exp(-0.10309278350515465*v_m));
    return (m_e_rev - v) * (1.0*m - m_p_open_eq);
}
double Kv3_1::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.708722777419175*exp(-0.10309278350515465*v_m)/pow(1.0 + 6.8746109409659972*exp(-0.10309278350515465*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(1.0 + 6.8746109409659972*exp(-0.10309278350515465*v_m));
    return -1. * (1.0*m - m_p_open_eq) + (1.0 * dm_dv) * (m_e_rev - v);
}

void K23::calcFunStatevar(double v){
    m_m_inf = 25.0/(25.0 + 0.045489799478447508*exp(-0.10000000000000001*v));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double K23::getCondNewton(){
    return m_g_bar;
}
double K23::f(double v){
    return (m_e_rev - v);
}
double K23::DfDv(double v){
    return -1.;
}
void K23::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_z = vs[1];
}
double K23::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 25.0/(25.0 + 0.045489799478447508*exp(-0.10000000000000001*v_m));
    double v_z;
    if(m_v_z > 1000.){
        v_z = v;
    } else{
        v_z = m_v_z;
    }
    double z = 0.0049751243781094526;
    return (m_e_rev - v) * (1.0*m*pow(z, 2) - m_p_open_eq);
}
double K23::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.11372449869611878*exp(-0.10000000000000001*v_m)/pow(25.0 + 0.045489799478447508*exp(-0.10000000000000001*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 25.0/(25.0 + 0.045489799478447508*exp(-0.10000000000000001*v_m));
    double v_z;
    double dz_dv;
    if(m_v_z > 1000.){
        v_z = v;
        dz_dv = 0;
    } else{
        v_z = m_v_z;
        dz_dv = 0;
    }
    double z = 0.0049751243781094526;
    return -1. * (1.0*m*pow(z, 2) - m_p_open_eq) + (1.0*pow(z, 2) * dm_dv+2.0*m*z * dz_dv) * (m_e_rev - v);
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
double Kh::getCondNewton(){
    return m_g_bar;
}
double Kh::f(double v){
    return (m_e_rev - v);
}
double Kh::DfDv(double v){
    return -1.;
}
void Kh::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_n = vs[1];
}
double Kh::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(exp((1.0/7.0)*v_m + 78.0/7.0) + 1);
    double v_n;
    if(m_v_n > 1000.){
        v_n = v;
    } else{
        v_n = m_v_n;
    }
    double n = 1.0/(exp((1.0/7.0)*v_n + 78.0/7.0) + 1);
    return (m_e_rev - v) * (0.80000000000000004*m + 0.20000000000000001*n - m_p_open_eq);
}
double Kh::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = -1.0/7.0*exp((1.0/7.0)*v_m + 78.0/7.0)/pow(exp((1.0/7.0)*v_m + 78.0/7.0) + 1, 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(exp((1.0/7.0)*v_m + 78.0/7.0) + 1);
    double v_n;
    double dn_dv;
    if(m_v_n > 1000.){
        v_n = v;
        dn_dv = -1.0/7.0*exp((1.0/7.0)*v_n + 78.0/7.0)/pow(exp((1.0/7.0)*v_n + 78.0/7.0) + 1, 2);
    } else{
        v_n = m_v_n;
        dn_dv = 0;
    }
    double n = 1.0/(exp((1.0/7.0)*v_n + 78.0/7.0) + 1);
    return -1. * (0.80000000000000004*m + 0.20000000000000001*n - m_p_open_eq) + (0.80000000000000004 * dm_dv+0.20000000000000001 * dn_dv) * (m_e_rev - v);
}

void Na_p::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(-0.21739130434782611*v - 11.434782608695654) + 1.0);
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double Na_p::getCondNewton(){
    return m_g_bar;
}
double Na_p::f(double v){
    return (m_e_rev - v);
}
double Na_p::DfDv(double v){
    return -1.;
}
void Na_p::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Na_p::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(1.0 + 1.0812771148577138e-5*exp(-0.21739130434782611*v_m));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(131.63066388583022*exp(0.10000000000000001*v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 3) - m_p_open_eq);
}
double Na_p::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 2.3506024236037257e-6*exp(-0.21739130434782611*v_m)/pow(1.0 + 1.0812771148577138e-5*exp(-0.21739130434782611*v_m), 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(1.0 + 1.0812771148577138e-5*exp(-0.21739130434782611*v_m));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -13.163066388583022*exp(0.10000000000000001*v_h)/pow(131.63066388583022*exp(0.10000000000000001*v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(131.63066388583022*exp(0.10000000000000001*v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 3) - m_p_open_eq) + (3.0*h*pow(m, 2) * dm_dv+1.0*pow(m, 3) * dh_dv) * (m_e_rev - v);
}

void TestChannel2::calcFunStatevar(double v){
    m_a00_inf = 0.29999999999999999;
    if(m_instantaneous)
        m_tau_a00 = 1.0000000000000001e-5;
    else
        m_tau_a00 = 1.0;
    m_a01_inf = 0.5;
    m_tau_a01 = 2.0;
    m_a10_inf = 0.40000000000000002;
    m_tau_a10 = 2.0;
    m_a11_inf = 0.59999999999999998;
    m_tau_a11 = 2.0;
}
double TestChannel2::calcPOpen(){
    return 0.90000000000000002*pow(m_a00, 3)*pow(m_a01, 2) + 0.10000000000000001*pow(m_a10, 2)*m_a11;
}
void TestChannel2::setPOpen(){
    m_p_open = calcPOpen();
}
void TestChannel2::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_a00 = m_a00_inf;
    m_a01 = m_a01_inf;
    m_a10 = m_a10_inf;
    m_a11 = m_a11_inf;
    m_p_open_eq =0.90000000000000002*pow(m_a00_inf, 3)*pow(m_a01_inf, 2) + 0.10000000000000001*pow(m_a10_inf, 2)*m_a11_inf;
}
void TestChannel2::advance(double dt){
    double p0_a00 = exp(-dt / m_tau_a00);
    m_a00 *= p0_a00 ;
    m_a00 += (1. - p0_a00 ) *  m_a00_inf;
    double p0_a01 = exp(-dt / m_tau_a01);
    m_a01 *= p0_a01 ;
    m_a01 += (1. - p0_a01 ) *  m_a01_inf;
    double p0_a10 = exp(-dt / m_tau_a10);
    m_a10 *= p0_a10 ;
    m_a10 += (1. - p0_a10 ) *  m_a10_inf;
    double p0_a11 = exp(-dt / m_tau_a11);
    m_a11 *= p0_a11 ;
    m_a11 += (1. - p0_a11 ) *  m_a11_inf;
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
    double v_a00;
    if(m_v_a00 > 1000.){
        v_a00 = v;
    } else{
        v_a00 = m_v_a00;
    }
    double a00 = 0.29999999999999999;
    double v_a01;
    if(m_v_a01 > 1000.){
        v_a01 = v;
    } else{
        v_a01 = m_v_a01;
    }
    double a01 = 0.5;
    double v_a10;
    if(m_v_a10 > 1000.){
        v_a10 = v;
    } else{
        v_a10 = m_v_a10;
    }
    double a10 = 0.40000000000000002;
    double v_a11;
    if(m_v_a11 > 1000.){
        v_a11 = v;
    } else{
        v_a11 = m_v_a11;
    }
    double a11 = 0.59999999999999998;
    return (m_e_rev - v) * (0.90000000000000002*pow(a00, 3)*pow(a01, 2) + 0.10000000000000001*pow(a10, 2)*a11 - m_p_open_eq);
}
double TestChannel2::DfDvNewton(double v){
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
    return -1. * (0.90000000000000002*pow(a00, 3)*pow(a01, 2) + 0.10000000000000001*pow(a10, 2)*a11 - m_p_open_eq) + (2.7000000000000002*pow(a00, 2)*pow(a01, 2) * da00_dv+1.8*pow(a00, 3)*a01 * da01_dv+0.20000000000000001*a10*a11 * da10_dv+0.10000000000000001*pow(a10, 2) * da11_dv) * (m_e_rev - v);
}

void K_ir::calcFunStatevar(double v){
    m_m_inf = 1.0/(exp(0.076923076923076927*v + 6.3076923076923084) + 1.0);
    m_tau_m = 12.0;
}
double K_ir::calcPOpen(){
    return 1.0*m_m;
}
void K_ir::setPOpen(){
    m_p_open = calcPOpen();
}
void K_ir::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_p_open_eq =1.0*m_m_inf;
}
void K_ir::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
}
double K_ir::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double K_ir::getCondNewton(){
    return m_g_bar;
}
double K_ir::f(double v){
    return (m_e_rev - v);
}
double K_ir::DfDv(double v){
    return -1.;
}
void K_ir::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
}
double K_ir::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(548.77707780552998*exp(0.076923076923076927*v_m) + 1.0);
    return (m_e_rev - v) * (1.0*m - m_p_open_eq);
}
double K_ir::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = -42.213621369656153*exp(0.076923076923076927*v_m)/pow(548.77707780552998*exp(0.076923076923076927*v_m) + 1.0, 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(548.77707780552998*exp(0.076923076923076927*v_m) + 1.0);
    return -1. * (1.0*m - m_p_open_eq) + (1.0 * dm_dv) * (m_e_rev - v);
}

void Na_shift::calcFunStatevar(double v){
    m_m_inf = (0.182*v + 3.6423659999999995)/((1.0 - 0.10821160462897761*exp(-0.1111111111111111*v))*((-0.124*v - 2.4816119999999997)/(-9.2411530484985853*exp(0.1111111111111111*v) + 1.0) + (0.182*v + 3.6423659999999995)/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v))));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
        m_tau_m = 0.3115264797507788/((-0.124*v - 2.4816119999999997)/(-9.2411530484985853*exp(0.1111111111111111*v) + 1.0) + (0.182*v + 3.6423659999999995)/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v)));
    m_h_inf = 1.0/(exp(0.16129032258064516*v + 8.064516129032258) + 1.0);
    m_tau_h = 0.3115264797507788/((-0.0091000000000000004*v - 0.54611830000000006)/(-163178.50446496159*exp(0.20000000000000001*v) + 1.0) + (0.024*v + 0.84031199999999995)/(1.0 - 0.00090951415193564705*exp(-0.20000000000000001*v)));
}
double Na_shift::calcPOpen(){
    return 1.0*m_h*pow(m_m, 3);
}
void Na_shift::setPOpen(){
    m_p_open = calcPOpen();
}
void Na_shift::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_m = m_m_inf;
    m_h = m_h_inf;
    m_p_open_eq =1.0*m_h_inf*pow(m_m_inf, 3);
}
void Na_shift::advance(double dt){
    double p0_m = exp(-dt / m_tau_m);
    m_m *= p0_m ;
    m_m += (1. - p0_m ) *  m_m_inf;
    double p0_h = exp(-dt / m_tau_h);
    m_h *= p0_h ;
    m_h += (1. - p0_h ) *  m_h_inf;
}
double Na_shift::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double Na_shift::getCondNewton(){
    return m_g_bar;
}
double Na_shift::f(double v){
    return (m_e_rev - v);
}
double Na_shift::DfDv(double v){
    return -1.;
}
void Na_shift::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Na_shift::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = (0.182*v_m + 3.6423659999999995)/((1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m))*((-0.124*v_m - 2.4816119999999997)/(-9.2411530484985853*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 3.6423659999999995)/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 1.0/(3179.6173203881372*exp(0.16129032258064516*v_h) + 1.0);
    return (m_e_rev - v) * (1.0*h*pow(m, 3) - m_p_open_eq);
}
double Na_shift::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = (0.182*v_m + 3.6423659999999995)*(-1.0267947831665094*(-0.124*v_m - 2.4816119999999997)*exp(0.1111111111111111*v_m)/pow(-9.2411530484985853*exp(0.1111111111111111*v_m) + 1.0, 2) + 0.124/(-9.2411530484985853*exp(0.1111111111111111*v_m) + 1.0) - 0.182/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m)) + 0.012023511625441956*(0.182*v_m + 3.6423659999999995)*exp(-0.1111111111111111*v_m)/pow(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m), 2))/((1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m))*pow((-0.124*v_m - 2.4816119999999997)/(-9.2411530484985853*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 3.6423659999999995)/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m)), 2)) + 0.182/((1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m))*((-0.124*v_m - 2.4816119999999997)/(-9.2411530484985853*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 3.6423659999999995)/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m)))) - 0.012023511625441956*(0.182*v_m + 3.6423659999999995)*exp(-0.1111111111111111*v_m)/(pow(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m), 2)*((-0.124*v_m - 2.4816119999999997)/(-9.2411530484985853*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 3.6423659999999995)/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = (0.182*v_m + 3.6423659999999995)/((1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m))*((-0.124*v_m - 2.4816119999999997)/(-9.2411530484985853*exp(0.1111111111111111*v_m) + 1.0) + (0.182*v_m + 3.6423659999999995)/(1.0 - 0.10821160462897761*exp(-0.1111111111111111*v_m))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = -512.84150328840917*exp(0.16129032258064516*v_h)/pow(3179.6173203881372*exp(0.16129032258064516*v_h) + 1.0, 2);
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 1.0/(3179.6173203881372*exp(0.16129032258064516*v_h) + 1.0);
    return -1. * (1.0*h*pow(m, 3) - m_p_open_eq) + (3.0*h*pow(m, 2) * dm_dv+1.0*pow(m, 3) * dh_dv) * (m_e_rev - v);
}

void KA::calcFunStatevar(double v){
    m_m_inf = 1.3999999999999999/((0.48999999999999999/(exp((1.0/4.0)*v + 15.0/2.0) + 1) + 1.3999999999999999/(exp(-1.0/12.0*v - 9.0/4.0) + 1))*(exp(-1.0/12.0*v - 9.0/4.0) + 1));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double KA::getCondNewton(){
    return m_g_bar;
}
double KA::f(double v){
    return (m_e_rev - v);
}
double KA::DfDv(double v){
    return -1.;
}
void KA::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double KA::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.3999999999999999/((0.48999999999999999/(exp((1.0/4.0)*v_m + 15.0/2.0) + 1) + 1.3999999999999999/(exp(-1.0/12.0*v_m - 9.0/4.0) + 1))*(exp(-1.0/12.0*v_m - 9.0/4.0) + 1));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.017500000000000002/((0.017500000000000002/(exp((1.0/8.0)*v_h + 25.0/4.0) + 1) + 1.3/(exp(-1.0/10.0*v_h - 13.0/10.0) + 1))*(exp((1.0/8.0)*v_h + 25.0/4.0) + 1));
    return (m_e_rev - v) * (1.0*h*pow(m, 4) - m_p_open_eq);
}
double KA::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 1.3999999999999999*(0.1225*exp((1.0/4.0)*v_m + 15.0/2.0)/pow(exp((1.0/4.0)*v_m + 15.0/2.0) + 1, 2) - 0.11666666666666665*exp(-1.0/12.0*v_m - 9.0/4.0)/pow(exp(-1.0/12.0*v_m - 9.0/4.0) + 1, 2))/(pow(0.48999999999999999/(exp((1.0/4.0)*v_m + 15.0/2.0) + 1) + 1.3999999999999999/(exp(-1.0/12.0*v_m - 9.0/4.0) + 1), 2)*(exp(-1.0/12.0*v_m - 9.0/4.0) + 1)) + 0.11666666666666665*exp(-1.0/12.0*v_m - 9.0/4.0)/((0.48999999999999999/(exp((1.0/4.0)*v_m + 15.0/2.0) + 1) + 1.3999999999999999/(exp(-1.0/12.0*v_m - 9.0/4.0) + 1))*pow(exp(-1.0/12.0*v_m - 9.0/4.0) + 1, 2));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.3999999999999999/((0.48999999999999999/(exp((1.0/4.0)*v_m + 15.0/2.0) + 1) + 1.3999999999999999/(exp(-1.0/12.0*v_m - 9.0/4.0) + 1))*(exp(-1.0/12.0*v_m - 9.0/4.0) + 1));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.017500000000000002*(0.0021875000000000002*exp((1.0/8.0)*v_h + 25.0/4.0)/pow(exp((1.0/8.0)*v_h + 25.0/4.0) + 1, 2) - 0.13*exp(-1.0/10.0*v_h - 13.0/10.0)/pow(exp(-1.0/10.0*v_h - 13.0/10.0) + 1, 2))/(pow(0.017500000000000002/(exp((1.0/8.0)*v_h + 25.0/4.0) + 1) + 1.3/(exp(-1.0/10.0*v_h - 13.0/10.0) + 1), 2)*(exp((1.0/8.0)*v_h + 25.0/4.0) + 1)) - 0.0021875000000000002*exp((1.0/8.0)*v_h + 25.0/4.0)/((0.017500000000000002/(exp((1.0/8.0)*v_h + 25.0/4.0) + 1) + 1.3/(exp(-1.0/10.0*v_h - 13.0/10.0) + 1))*pow(exp((1.0/8.0)*v_h + 25.0/4.0) + 1, 2));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.017500000000000002/((0.017500000000000002/(exp((1.0/8.0)*v_h + 25.0/4.0) + 1) + 1.3/(exp(-1.0/10.0*v_h - 13.0/10.0) + 1))*(exp((1.0/8.0)*v_h + 25.0/4.0) + 1));
    return -1. * (1.0*h*pow(m, 4) - m_p_open_eq) + (4.0*h*pow(m, 3) * dm_dv+1.0*pow(m, 4) * dh_dv) * (m_e_rev - v);
}

void Khh::calcFunStatevar(double v){
    m_n_inf = (-0.01*v - 0.551234)/(((-0.01*v - 0.551234)/(exp(-0.10000000000000001*v - 5.51234) - 1.0) + 0.125*0.44374731008107987*exp(-0.012500000000000001*v))*(exp(-0.10000000000000001*v - 5.51234) - 1.0));
    m_tau_n = 1.0/((-0.01*v - 0.551234)/(exp(-0.10000000000000001*v - 5.51234) - 1.0) + 0.125*0.44374731008107987*exp(-0.012500000000000001*v));
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
double Khh::getCondNewton(){
    return m_g_bar;
}
double Khh::f(double v){
    return (m_e_rev - v);
}
double Khh::DfDv(double v){
    return -1.;
}
void Khh::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_n = vs[0];
}
double Khh::fNewton(double v){
    double v_n;
    if(m_v_n > 1000.){
        v_n = v;
    } else{
        v_n = m_v_n;
    }
    double n = (-0.01*v_n - 0.551234)/((-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n))*(0.055468413760134984*exp(-0.012500000000000001*v_n) + (-0.01*v_n - 0.551234)/(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n))));
    return (m_e_rev - v) * (1.0*pow(n, 4) - m_p_open_eq);
}
double Khh::DfDvNewton(double v){
    double v_n;
    double dn_dv;
    if(m_v_n > 1000.){
        v_n = v;
        dn_dv = (-0.01*v_n - 0.551234)*(0.0006933551720016873*exp(-0.012500000000000001*v_n) + 0.01/(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n)) - 0.00040366505607429066*(-0.01*v_n - 0.551234)*exp(-0.10000000000000001*v_n)/pow(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n), 2))/((-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n))*pow(0.055468413760134984*exp(-0.012500000000000001*v_n) + (-0.01*v_n - 0.551234)/(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n)), 2)) - 0.01/((-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n))*(0.055468413760134984*exp(-0.012500000000000001*v_n) + (-0.01*v_n - 0.551234)/(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n)))) + 0.00040366505607429066*(-0.01*v_n - 0.551234)*exp(-0.10000000000000001*v_n)/(pow(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n), 2)*(0.055468413760134984*exp(-0.012500000000000001*v_n) + (-0.01*v_n - 0.551234)/(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n))));
    } else{
        v_n = m_v_n;
        dn_dv = 0;
    }
    double n = (-0.01*v_n - 0.551234)/((-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n))*(0.055468413760134984*exp(-0.012500000000000001*v_n) + (-0.01*v_n - 0.551234)/(-1.0 + 0.0040366505607429062*exp(-0.10000000000000001*v_n))));
    return -1. * (1.0*pow(n, 4) - m_p_open_eq) + (4.0*pow(n, 3) * dn_dv) * (m_e_rev - v);
}

void Ca_HVA::calcFunStatevar(double v){
    m_m_inf = (-0.055*v - 1.4850000000000001)/(((-0.055*v - 1.4850000000000001)/(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0) + 0.012133746930834877*0.93999999999999995*exp(-0.058823529411764705*v))*(exp(-0.26315789473684209*v - 7.1052631578947363) - 1.0));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double Ca_HVA::getCondNewton(){
    return m_g_bar;
}
double Ca_HVA::f(double v){
    return (m_e_rev - v);
}
double Ca_HVA::DfDv(double v){
    return -1.;
}
void Ca_HVA::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double Ca_HVA::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = (-0.055*v_m - 1.4850000000000001)/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.00035237057471222975*exp(-0.02*v_h)/(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)));
    return (m_e_rev - v) * (1.0*h*pow(m, 2) - m_p_open_eq);
}
double Ca_HVA::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = (-0.055*v_m - 1.4850000000000001)*(0.00067092483029322256*exp(-0.058823529411764705*v_m) + 0.055/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m)) - 0.00021599307205216037*(-0.055*v_m - 1.4850000000000001)*exp(-0.26315789473684209*v_m)/pow(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m), 2))/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*pow(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m)), 2)) - 0.055/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m)))) + 0.00021599307205216037*(-0.055*v_m - 1.4850000000000001)*exp(-0.26315789473684209*v_m)/(pow(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m), 2)*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = (-0.055*v_m - 1.4850000000000001)/((-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))*(0.011405722114984784*exp(-0.058823529411764705*v_m) + (-0.055*v_m - 1.4850000000000001)/(-1.0 + 0.0008207736737982094*exp(-0.26315789473684209*v_m))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.00035237057471222975*(7.0474114942445952e-6*exp(-0.02*v_h) - 0.000135861863499935*exp(-0.035714285714285712*v_h)/pow(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h), 2))*exp(-0.02*v_h)/pow(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)), 2) - 7.0474114942445952e-6*exp(-0.02*v_h)/(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.00035237057471222975*exp(-0.02*v_h)/(0.00035237057471222975*exp(-0.02*v_h) + 0.0064999999999999997/(1.0 + 0.58525110430741234*exp(-0.035714285714285712*v_h)));
    return -1. * (1.0*h*pow(m, 2) - m_p_open_eq) + (2.0*h*m * dm_dv+1.0*pow(m, 2) * dh_dv) * (m_e_rev - v);
}

void KD::calcFunStatevar(double v){
    m_m_inf = 8.5/((35/(exp(0.068965517241379309*v + 6.8275862068965516) + 1) + 8.5/(exp(-0.080000000000000002*v - 1.3600000000000001) + 1))*(exp(-0.080000000000000002*v - 1.3600000000000001) + 1));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double KD::getCondNewton(){
    return m_g_bar;
}
double KD::f(double v){
    return (m_e_rev - v);
}
double KD::DfDv(double v){
    return -1.;
}
void KD::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double KD::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 8.5/((1 + 0.25666077695355582*exp(-0.080000000000000002*v_m))*(35/(922.96028637378186*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 0.25666077695355582*exp(-0.080000000000000002*v_m))));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.0015/((0.0015/(exp((1.0/8.0)*v_h + 89.0/8.0) + 1) + 0.0054999999999999997/(exp(-1.0/8.0*v_h - 83.0/8.0) + 1))*(exp((1.0/8.0)*v_h + 89.0/8.0) + 1));
    return (m_e_rev - v) * (1.0*h*m - m_p_open_eq);
}
double KD::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 8.5*(2227.8351740056801*exp(0.068965517241379309*v_m)/pow(922.96028637378186*exp(0.068965517241379309*v_m) + 1, 2) - 0.17452932832841797*exp(-0.080000000000000002*v_m)/pow(1 + 0.25666077695355582*exp(-0.080000000000000002*v_m), 2))/((1 + 0.25666077695355582*exp(-0.080000000000000002*v_m))*pow(35/(922.96028637378186*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 0.25666077695355582*exp(-0.080000000000000002*v_m)), 2)) + 0.17452932832841797*exp(-0.080000000000000002*v_m)/(pow(1 + 0.25666077695355582*exp(-0.080000000000000002*v_m), 2)*(35/(922.96028637378186*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 0.25666077695355582*exp(-0.080000000000000002*v_m))));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 8.5/((1 + 0.25666077695355582*exp(-0.080000000000000002*v_m))*(35/(922.96028637378186*exp(0.068965517241379309*v_m) + 1) + 8.5/(1 + 0.25666077695355582*exp(-0.080000000000000002*v_m))));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.0015*(0.0001875*exp((1.0/8.0)*v_h + 89.0/8.0)/pow(exp((1.0/8.0)*v_h + 89.0/8.0) + 1, 2) - 0.00068749999999999996*exp(-1.0/8.0*v_h - 83.0/8.0)/pow(exp(-1.0/8.0*v_h - 83.0/8.0) + 1, 2))/(pow(0.0015/(exp((1.0/8.0)*v_h + 89.0/8.0) + 1) + 0.0054999999999999997/(exp(-1.0/8.0*v_h - 83.0/8.0) + 1), 2)*(exp((1.0/8.0)*v_h + 89.0/8.0) + 1)) - 0.0001875*exp((1.0/8.0)*v_h + 89.0/8.0)/((0.0015/(exp((1.0/8.0)*v_h + 89.0/8.0) + 1) + 0.0054999999999999997/(exp(-1.0/8.0*v_h - 83.0/8.0) + 1))*pow(exp((1.0/8.0)*v_h + 89.0/8.0) + 1, 2));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.0015/((0.0015/(exp((1.0/8.0)*v_h + 89.0/8.0) + 1) + 0.0054999999999999997/(exp(-1.0/8.0*v_h - 83.0/8.0) + 1))*(exp((1.0/8.0)*v_h + 89.0/8.0) + 1));
    return -1. * (1.0*h*m - m_p_open_eq) + (1.0*h * dm_dv+1.0*m * dh_dv) * (m_e_rev - v);
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
    double hf = 1.0/(122306.53009058574*exp(0.14285714285714285*v_hf) + 1.0);
    double v_hs;
    if(m_v_hs > 1000.){
        v_hs = v;
    } else{
        v_hs = m_v_hs;
    }
    double hs = 1.0/(122306.53009058574*exp(0.14285714285714285*v_hs) + 1.0);
    return (m_e_rev - v) * (0.80000000000000004*hf + 0.20000000000000001*hs - m_p_open_eq);
}
double h::DfDvNewton(double v){
    double v_hf;
    double dhf_dv;
    if(m_v_hf > 1000.){
        v_hf = v;
        dhf_dv = -17472.361441512247*exp(0.14285714285714285*v_hf)/pow(122306.53009058574*exp(0.14285714285714285*v_hf) + 1.0, 2);
    } else{
        v_hf = m_v_hf;
        dhf_dv = 0;
    }
    double hf = 1.0/(122306.53009058574*exp(0.14285714285714285*v_hf) + 1.0);
    double v_hs;
    double dhs_dv;
    if(m_v_hs > 1000.){
        v_hs = v;
        dhs_dv = -17472.361441512247*exp(0.14285714285714285*v_hs)/pow(122306.53009058574*exp(0.14285714285714285*v_hs) + 1.0, 2);
    } else{
        v_hs = m_v_hs;
        dhs_dv = 0;
    }
    double hs = 1.0/(122306.53009058574*exp(0.14285714285714285*v_hs) + 1.0);
    return -1. * (0.80000000000000004*hf + 0.20000000000000001*hs - m_p_open_eq) + (0.80000000000000004 * dhf_dv+0.20000000000000001 * dhs_dv) * (m_e_rev - v);
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
double m::getCondNewton(){
    return m_g_bar;
}
double m::f(double v){
    return (m_e_rev - v);
}
double m::DfDv(double v){
    return -1.;
}
void m::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
}
double m::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 0.10928099146368463*exp(0.10000000000000001*v_m)/(0.10928099146368463*exp(0.10000000000000001*v_m) + 9.9651365293651053e-5*exp(-0.10000000000000001*v_m));
    return (m_e_rev - v) * (1.0*m - m_p_open_eq);
}
double m::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.10928099146368463*(-0.010928099146368463*exp(0.10000000000000001*v_m) + 9.9651365293651063e-6*exp(-0.10000000000000001*v_m))*exp(0.10000000000000001*v_m)/pow(0.10928099146368463*exp(0.10000000000000001*v_m) + 9.9651365293651053e-5*exp(-0.10000000000000001*v_m), 2) + 0.010928099146368463*exp(0.10000000000000001*v_m)/(0.10928099146368463*exp(0.10000000000000001*v_m) + 9.9651365293651053e-5*exp(-0.10000000000000001*v_m));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 0.10928099146368463*exp(0.10000000000000001*v_m)/(0.10928099146368463*exp(0.10000000000000001*v_m) + 9.9651365293651053e-5*exp(-0.10000000000000001*v_m));
    return -1. * (1.0*m - m_p_open_eq) + (1.0 * dm_dv) * (m_e_rev - v);
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
double KM::getCondNewton(){
    return m_g_bar;
}
double KM::f(double v){
    return (m_e_rev - v);
}
double KM::DfDv(double v){
    return -1.;
}
void KM::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
}
double KM::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 1.0/(exp(-1.0/10.0*v_m - 7.0/2.0) + 1);
    return (m_e_rev - v) * (1.0*m - m_p_open_eq);
}
double KM::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 0.10000000000000001*exp(-1.0/10.0*v_m - 7.0/2.0)/pow(exp(-1.0/10.0*v_m - 7.0/2.0) + 1, 2);
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 1.0/(exp(-1.0/10.0*v_m - 7.0/2.0) + 1);
    return -1. * (1.0*m - m_p_open_eq) + (1.0 * dm_dv) * (m_e_rev - v);
}

void CaT::calcFunStatevar(double v){
    m_m_inf = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v + 10) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v - 21.0/8.0) + 1))*(exp(-1.0/8.0*v - 21.0/8.0) + 1));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double CaT::getCondNewton(){
    return m_g_bar;
}
double CaT::f(double v){
    return (m_e_rev - v);
}
double CaT::DfDv(double v){
    return -1.;
}
void CaT::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double CaT::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v_m + 10) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 21.0/8.0) + 1))*(exp(-1.0/8.0*v_m - 21.0/8.0) + 1));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.0025000000000000001/((0.0025000000000000001/(exp((1.0/8.0)*v_h + 5) + 1) + 0.19/(exp(-1.0/10.0*v_h - 5) + 1))*(exp((1.0/8.0)*v_h + 5) + 1));
    return (m_e_rev - v) * (1.0*h*m - m_p_open_eq);
}
double CaT::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 2.6000000000000001*(0.044999999999999998*exp((1.0/4.0)*v_m + 10)/pow(exp((1.0/4.0)*v_m + 10) + 1, 2) - 0.32500000000000001*exp(-1.0/8.0*v_m - 21.0/8.0)/pow(exp(-1.0/8.0*v_m - 21.0/8.0) + 1, 2))/(pow(0.17999999999999999/(exp((1.0/4.0)*v_m + 10) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 21.0/8.0) + 1), 2)*(exp(-1.0/8.0*v_m - 21.0/8.0) + 1)) + 0.32500000000000001*exp(-1.0/8.0*v_m - 21.0/8.0)/((0.17999999999999999/(exp((1.0/4.0)*v_m + 10) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 21.0/8.0) + 1))*pow(exp(-1.0/8.0*v_m - 21.0/8.0) + 1, 2));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v_m + 10) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 21.0/8.0) + 1))*(exp(-1.0/8.0*v_m - 21.0/8.0) + 1));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.0025000000000000001*(0.00031250000000000001*exp((1.0/8.0)*v_h + 5)/pow(exp((1.0/8.0)*v_h + 5) + 1, 2) - 0.019000000000000003*exp(-1.0/10.0*v_h - 5)/pow(exp(-1.0/10.0*v_h - 5) + 1, 2))/(pow(0.0025000000000000001/(exp((1.0/8.0)*v_h + 5) + 1) + 0.19/(exp(-1.0/10.0*v_h - 5) + 1), 2)*(exp((1.0/8.0)*v_h + 5) + 1)) - 0.00031250000000000001*exp((1.0/8.0)*v_h + 5)/((0.0025000000000000001/(exp((1.0/8.0)*v_h + 5) + 1) + 0.19/(exp(-1.0/10.0*v_h - 5) + 1))*pow(exp((1.0/8.0)*v_h + 5) + 1, 2));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.0025000000000000001/((0.0025000000000000001/(exp((1.0/8.0)*v_h + 5) + 1) + 0.19/(exp(-1.0/10.0*v_h - 5) + 1))*(exp((1.0/8.0)*v_h + 5) + 1));
    return -1. * (1.0*h*m - m_p_open_eq) + (1.0*h * dm_dv+1.0*m * dh_dv) * (m_e_rev - v);
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
    double z = 0.0009098213063332165;
    return (m_e_rev - v) * (1.0*z - m_p_open_eq);
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
    double z = 0.0009098213063332165;
    return -1. * (1.0*z - m_p_open_eq) + (1.0 * dz_dv) * (m_e_rev - v);
}

void CaE::calcFunStatevar(double v){
    m_m_inf = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v + 13.0/2.0) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v - 7.0/8.0) + 1))*(exp(-1.0/8.0*v - 7.0/8.0) + 1));
    if(m_instantaneous)
        m_tau_m = 1.0000000000000001e-5;
    else
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
double CaE::getCondNewton(){
    return m_g_bar;
}
double CaE::f(double v){
    return (m_e_rev - v);
}
double CaE::DfDv(double v){
    return -1.;
}
void CaE::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 2)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
    m_v_h = vs[1];
}
double CaE::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v_m + 13.0/2.0) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 7.0/8.0) + 1))*(exp(-1.0/8.0*v_m - 7.0/8.0) + 1));
    double v_h;
    if(m_v_h > 1000.){
        v_h = v;
    } else{
        v_h = m_v_h;
    }
    double h = 0.0025000000000000001/((0.0025000000000000001/(exp((1.0/8.0)*v_h + 4) + 1) + 0.19/(exp(-1.0/10.0*v_h - 21.0/5.0) + 1))*(exp((1.0/8.0)*v_h + 4) + 1));
    return (m_e_rev - v) * (1.0*h*m - m_p_open_eq);
}
double CaE::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = 2.6000000000000001*(0.044999999999999998*exp((1.0/4.0)*v_m + 13.0/2.0)/pow(exp((1.0/4.0)*v_m + 13.0/2.0) + 1, 2) - 0.32500000000000001*exp(-1.0/8.0*v_m - 7.0/8.0)/pow(exp(-1.0/8.0*v_m - 7.0/8.0) + 1, 2))/(pow(0.17999999999999999/(exp((1.0/4.0)*v_m + 13.0/2.0) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 7.0/8.0) + 1), 2)*(exp(-1.0/8.0*v_m - 7.0/8.0) + 1)) + 0.32500000000000001*exp(-1.0/8.0*v_m - 7.0/8.0)/((0.17999999999999999/(exp((1.0/4.0)*v_m + 13.0/2.0) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 7.0/8.0) + 1))*pow(exp(-1.0/8.0*v_m - 7.0/8.0) + 1, 2));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = 2.6000000000000001/((0.17999999999999999/(exp((1.0/4.0)*v_m + 13.0/2.0) + 1) + 2.6000000000000001/(exp(-1.0/8.0*v_m - 7.0/8.0) + 1))*(exp(-1.0/8.0*v_m - 7.0/8.0) + 1));
    double v_h;
    double dh_dv;
    if(m_v_h > 1000.){
        v_h = v;
        dh_dv = 0.0025000000000000001*(0.00031250000000000001*exp((1.0/8.0)*v_h + 4)/pow(exp((1.0/8.0)*v_h + 4) + 1, 2) - 0.019000000000000003*exp(-1.0/10.0*v_h - 21.0/5.0)/pow(exp(-1.0/10.0*v_h - 21.0/5.0) + 1, 2))/(pow(0.0025000000000000001/(exp((1.0/8.0)*v_h + 4) + 1) + 0.19/(exp(-1.0/10.0*v_h - 21.0/5.0) + 1), 2)*(exp((1.0/8.0)*v_h + 4) + 1)) - 0.00031250000000000001*exp((1.0/8.0)*v_h + 4)/((0.0025000000000000001/(exp((1.0/8.0)*v_h + 4) + 1) + 0.19/(exp(-1.0/10.0*v_h - 21.0/5.0) + 1))*pow(exp((1.0/8.0)*v_h + 4) + 1, 2));
    } else{
        v_h = m_v_h;
        dh_dv = 0;
    }
    double h = 0.0025000000000000001/((0.0025000000000000001/(exp((1.0/8.0)*v_h + 4) + 1) + 0.19/(exp(-1.0/10.0*v_h - 21.0/5.0) + 1))*(exp((1.0/8.0)*v_h + 4) + 1));
    return -1. * (1.0*h*m - m_p_open_eq) + (1.0*h * dm_dv+1.0*m * dh_dv) * (m_e_rev - v);
}

void K_m::calcFunStatevar(double v){
    m_n_inf = (0.001*v + 0.029999999999999999)/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v))*((-0.001*v - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v) + 1.0) + (0.001*v + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v))));
    m_tau_n = 1.0/(3.21*(-0.001*v - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v) + 1.0) + 3.21*(0.001*v + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v)));
}
double K_m::calcPOpen(){
    return 1.0*m_n;
}
void K_m::setPOpen(){
    m_p_open = calcPOpen();
}
void K_m::setPOpenEQ(double v){
    calcFunStatevar(v);
    m_n = m_n_inf;
    m_p_open_eq =1.0*m_n_inf;
}
void K_m::advance(double dt){
    double p0_n = exp(-dt / m_tau_n);
    m_n *= p0_n ;
    m_n += (1. - p0_n ) *  m_n_inf;
}
double K_m::getCond(){
    return m_g_bar * (m_p_open - m_p_open_eq);
}
double K_m::getCondNewton(){
    return m_g_bar;
}
double K_m::f(double v){
    return (m_e_rev - v);
}
double K_m::DfDv(double v){
    return -1.;
}
void K_m::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_n = vs[0];
}
double K_m::fNewton(double v){
    double v_n;
    if(m_v_n > 1000.){
        v_n = v;
    } else{
        v_n = m_v_n;
    }
    double n = (0.001*v_n + 0.029999999999999999)/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))));
    return (m_e_rev - v) * (1.0*n - m_p_open_eq);
}
double K_m::DfDvNewton(double v){
    double v_n;
    double dn_dv;
    if(m_v_n > 1000.){
        v_n = v;
        dn_dv = (0.001*v_n + 0.029999999999999999)*(-3.1146249882806805*(-0.001*v_n - 0.029999999999999999)*exp(0.1111111111111111*v_n)/pow(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0, 2) + 0.001/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) - 0.001/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n)) + 0.0039637770385836006*(0.001*v_n + 0.029999999999999999)*exp(-0.1111111111111111*v_n)/pow(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n), 2))/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*pow((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n)), 2)) + 0.001/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n)))) - 0.0039637770385836006*(0.001*v_n + 0.029999999999999999)*exp(-0.1111111111111111*v_n)/(pow(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n), 2)*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))));
    } else{
        v_n = m_v_n;
        dn_dv = 0;
    }
    double n = (0.001*v_n + 0.029999999999999999)/((1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))*((-0.001*v_n - 0.029999999999999999)/(-28.031624894526125*exp(0.1111111111111111*v_n) + 1.0) + (0.001*v_n + 0.029999999999999999)/(1.0 - 0.035673993347252408*exp(-0.1111111111111111*v_n))));
    return -1. * (1.0*n - m_p_open_eq) + (1.0 * dn_dv) * (m_e_rev - v);
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
double h_HAY::getCondNewton(){
    return m_g_bar;
}
double h_HAY::f(double v){
    return (m_e_rev - v);
}
double h_HAY::DfDv(double v){
    return -1.;
}
void h_HAY::setfNewtonConstant(double* vs, int v_size){
    if(v_size != 1)
        cerr << "input arg [vs] has incorrect size, should have same size as number of channel state variables" << endl;
    m_v_m = vs[0];
}
double h_HAY::fNewton(double v){
    double v_m;
    if(m_v_m > 1000.){
        v_m = v;
    } else{
        v_m = m_v_m;
    }
    double m = (0.00643*v_m + 0.99600699999999998)/(((0.00643*v_m + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0) + 0.193*exp(0.030211480362537763*v_m))*(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0));
    return (m_e_rev - v) * (1.0*m - m_p_open_eq);
}
double h_HAY::DfDvNewton(double v){
    double v_m;
    double dm_dv;
    if(m_v_m > 1000.){
        v_m = v;
        dm_dv = -37807.709754576565*(0.00643*v_m + 0.99600699999999998)*exp(0.084033613445378144*v_m)/(((0.00643*v_m + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0) + 0.193*exp(0.030211480362537763*v_m))*pow(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0, 2)) + (0.00643*v_m + 0.99600699999999998)*(37807.709754576565*(0.00643*v_m + 0.99600699999999998)*exp(0.084033613445378144*v_m)/pow(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0, 2) - 0.0058308157099697883*exp(0.030211480362537763*v_m) - 0.00643/(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0))/(pow((0.00643*v_m + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0) + 0.193*exp(0.030211480362537763*v_m), 2)*(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0)) + 0.00643/(((0.00643*v_m + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0) + 0.193*exp(0.030211480362537763*v_m))*(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0));
    } else{
        v_m = m_v_m;
        dm_dv = 0;
    }
    double m = (0.00643*v_m + 0.99600699999999998)/(((0.00643*v_m + 0.99600699999999998)/(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0) + 0.193*exp(0.030211480362537763*v_m))*(449911.74607946118*exp(0.084033613445378144*v_m) - 1.0));
    return -1. * (1.0*m - m_p_open_eq) + (1.0 * dm_dv) * (m_e_rev - v);
}

