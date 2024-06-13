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
#include <time.h>
using namespace std;

class IonChannel{
protected:
    double m_g_bar = 0.0, m_e_rev = 50.00000000;
    bool m_instantaneous = false;
public:
    void init(double g_bar, double e_rev){m_g_bar = g_bar; m_e_rev = e_rev;};
    void setInstantaneous(bool b){m_instantaneous = b;};
    virtual void calcFunStatevar(double v){};
    virtual double calcPOpen(){return 0.0;};
    virtual void setPOpen(){};
    virtual void setPOpenEQ(double v){};
    virtual void advance(double dt){};
    virtual double getCond(){return 0.0;};
    virtual double getCondNewton(){return 0.0;};
    virtual double f(double v){return 0.0;};
    virtual double DfDv(double v){return 0.0;};
    virtual void setfNewtonConstant(double* vs, int v_size){};
    virtual double fNewton(double v){return 0.0;};
    virtual double DfDvNewton(double v){return 0.0;};
};

class test_channel: public IonChannel{
private:
    double m_a01;
    double m_a01_inf, m_tau_a01;
    double m_v_a01 = 10000.;
    double m_a12;
    double m_a12_inf, m_tau_a12;
    double m_v_a12 = 10000.;
    double m_a11;
    double m_a11_inf, m_tau_a11;
    double m_v_a11 = 10000.;
    double m_a10;
    double m_a10_inf, m_tau_a10;
    double m_v_a10 = 10000.;
    double m_a02;
    double m_a02_inf, m_tau_a02;
    double m_v_a02 = 10000.;
    double m_a00;
    double m_a00_inf, m_tau_a00;
    double m_v_a00 = 10000.;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double getCondNewton() override;
    double f(double v) override;
    double DfDv(double v) override;
    void setfNewtonConstant(double* vs, int v_size) override;
    double fNewton(double v) override;
    double DfDvNewton(double v) override;
};

class test_channel2: public IonChannel{
private:
    double m_a11;
    double m_a11_inf, m_tau_a11;
    double m_v_a11 = 10000.;
    double m_a10;
    double m_a10_inf, m_tau_a10;
    double m_v_a10 = 10000.;
    double m_a01;
    double m_a01_inf, m_tau_a01;
    double m_v_a01 = 10000.;
    double m_a00;
    double m_a00_inf, m_tau_a00;
    double m_v_a00 = 10000.;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double getCondNewton() override;
    double f(double v) override;
    double DfDv(double v) override;
    void setfNewtonConstant(double* vs, int v_size) override;
    double fNewton(double v) override;
    double DfDvNewton(double v) override;
};

class Na_Ta: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_v_m = 10000.;
    double m_h;
    double m_h_inf, m_tau_h;
    double m_v_h = 10000.;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double getCondNewton() override;
    double f(double v) override;
    double DfDv(double v) override;
    void setfNewtonConstant(double* vs, int v_size) override;
    double fNewton(double v) override;
    double DfDvNewton(double v) override;
};

class Kv3_1: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_v_m = 10000.;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double getCondNewton() override;
    double f(double v) override;
    double DfDv(double v) override;
    void setfNewtonConstant(double* vs, int v_size) override;
    double fNewton(double v) override;
    double DfDvNewton(double v) override;
};

class SK: public IonChannel{
private:
    double m_z;
    double m_z_inf, m_tau_z;
    double m_v_z = 10000.;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
    double m_ca = 0.00010000;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double getCondNewton() override;
    double f(double v) override;
    double DfDv(double v) override;
    void setfNewtonConstant(double* vs, int v_size) override;
    double fNewton(double v) override;
    double DfDvNewton(double v) override;
};

class h: public IonChannel{
private:
    double m_hf;
    double m_hf_inf, m_tau_hf;
    double m_v_hf = 10000.;
    double m_hs;
    double m_hs_inf, m_tau_hs;
    double m_v_hs = 10000.;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double getCondNewton() override;
    double f(double v) override;
    double DfDv(double v) override;
    void setfNewtonConstant(double* vs, int v_size) override;
    double fNewton(double v) override;
    double DfDvNewton(double v) override;
};

class ChannelCreator{
public:
    IonChannel* createInstance(string channel_name){
        if(channel_name == "test_channel"){
            return new test_channel();
        }
        else if(channel_name == "test_channel2"){
            return new test_channel2();
        }
        else if(channel_name == "Na_Ta"){
            return new Na_Ta();
        }
        else if(channel_name == "Kv3_1"){
            return new Kv3_1();
        }
        else if(channel_name == "SK"){
            return new SK();
        }
        else if(channel_name == "h"){
            return new h();
        }
    };
};
