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
public:
    void init(double g_bar, double e_rev){m_g_bar = g_bar; m_e_rev = e_rev;};
    virtual void calcFunStatevar(double v){};
    virtual double calcPOpen(){};
    virtual void setPOpen(){};
    virtual void setPOpenEQ(double v){};
    virtual void advance(double dt){};
    virtual double getCond(){return 0.0;};
    virtual double f(double v){return 0.0;};
    virtual double DfDv(double v){return 0.0;};
};

class CaP: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class TestChannel: public IonChannel{
private:
    double m_a00;
    double m_a01;
    double m_a02;
    double m_a10;
    double m_a11;
    double m_a12;
    double m_a00_inf, m_tau_a00;
    double m_a01_inf, m_tau_a01;
    double m_a02_inf, m_tau_a02;
    double m_a10_inf, m_tau_a10;
    double m_a11_inf, m_tau_a11;
    double m_a12_inf, m_tau_a12;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class CaP2: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Ktst: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class KC3: public IonChannel{
private:
    double m_m;
    double m_z;
    double m_m_inf, m_tau_m;
    double m_z_inf, m_tau_z;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Ca_LVA: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class NaF: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Na_Ta: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Kpst: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class NaP: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Kv3_1: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class K23: public IonChannel{
private:
    double m_m;
    double m_z;
    double m_m_inf, m_tau_m;
    double m_z_inf, m_tau_z;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Kh: public IonChannel{
private:
    double m_m;
    double m_n;
    double m_m_inf, m_tau_m;
    double m_n_inf, m_tau_n;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Na_p: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class KA: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Khh: public IonChannel{
private:
    double m_n;
    double m_n_inf, m_tau_n;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class Ca_HVA: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class KD: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class h: public IonChannel{
private:
    double m_hf;
    double m_hs;
    double m_hf_inf, m_tau_hf;
    double m_hs_inf, m_tau_hs;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class m: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class KM: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class CaT: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class SK: public IonChannel{
private:
    double m_z;
    double m_z_inf, m_tau_z;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class CaE: public IonChannel{
private:
    double m_m;
    double m_h;
    double m_m_inf, m_tau_m;
    double m_h_inf, m_tau_h;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class h_HAY: public IonChannel{
private:
    double m_m;
    double m_m_inf, m_tau_m;
    double m_p_open_eq = 0.0, m_p_open = 0.0;
public:
    void calcFunStatevar(double v) override;
    double calcPOpen() override;
    void setPOpen() override;
    void setPOpenEQ(double v) override;
    void advance(double dt) override;
    double getCond() override;
    double f(double v) override;
    double DfDv(double v) override;
};

class ChannelCreator{
public:
    IonChannel* createInstance(string channel_name){
        if(channel_name == "CaP"){
            return new CaP();
        }
        else if(channel_name == "TestChannel"){
            return new TestChannel();
        }
        else if(channel_name == "CaP2"){
            return new CaP2();
        }
        else if(channel_name == "Ktst"){
            return new Ktst();
        }
        else if(channel_name == "KC3"){
            return new KC3();
        }
        else if(channel_name == "Ca_LVA"){
            return new Ca_LVA();
        }
        else if(channel_name == "NaF"){
            return new NaF();
        }
        else if(channel_name == "Na_Ta"){
            return new Na_Ta();
        }
        else if(channel_name == "Kpst"){
            return new Kpst();
        }
        else if(channel_name == "NaP"){
            return new NaP();
        }
        else if(channel_name == "Kv3_1"){
            return new Kv3_1();
        }
        else if(channel_name == "K23"){
            return new K23();
        }
        else if(channel_name == "Kh"){
            return new Kh();
        }
        else if(channel_name == "Na_p"){
            return new Na_p();
        }
        else if(channel_name == "KA"){
            return new KA();
        }
        else if(channel_name == "Khh"){
            return new Khh();
        }
        else if(channel_name == "Ca_HVA"){
            return new Ca_HVA();
        }
        else if(channel_name == "KD"){
            return new KD();
        }
        else if(channel_name == "h"){
            return new h();
        }
        else if(channel_name == "m"){
            return new m();
        }
        else if(channel_name == "KM"){
            return new KM();
        }
        else if(channel_name == "CaT"){
            return new CaT();
        }
        else if(channel_name == "SK"){
            return new SK();
        }
        else if(channel_name == "CaE"){
            return new CaE();
        }
        else if(channel_name == "h_HAY"){
            return new h_HAY();
        }
    };
};
