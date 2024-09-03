
#ifndef SYNAPSES_NEAT_H_MULTICHANNELTESTMODEL
#define SYNAPSES_NEAT_H_MULTICHANNELTESTMODEL

#include <stdlib.h>
#include <vector>

#include "ring_buffer.h"



namespace nest
{

///////////////////////////////////// channels

class i_hMultichannelTestModel{
private:
    // states
    std::vector< double > hf_h = {};
    std::vector< double > hs_h = {};

    // parameters
    std::vector< double > gbar_h = {};
    std::vector< double > e_h = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_h = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_hMultichannelTestModel(){};
    ~i_hMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_h_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions h
double hf_inf_h ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_hf_h ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double hs_inf_h ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_hs_h ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_TestChannel2MultichannelTestModel{
private:
    // states
    std::vector< double > a00_TestChannel2 = {};
    std::vector< double > a01_TestChannel2 = {};
    std::vector< double > a10_TestChannel2 = {};
    std::vector< double > a11_TestChannel2 = {};

    // parameters
    std::vector< double > gbar_TestChannel2 = {};
    std::vector< double > e_TestChannel2 = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_TestChannel2 = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_TestChannel2MultichannelTestModel(){};
    ~i_TestChannel2MultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_TestChannel2_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions TestChannel2
double a00_inf_TestChannel2 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a00_TestChannel2 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a01_inf_TestChannel2 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a01_TestChannel2 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a10_inf_TestChannel2 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a10_TestChannel2 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a11_inf_TestChannel2 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a11_TestChannel2 ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_TestChannelMultichannelTestModel{
private:
    // states
    std::vector< double > a00_TestChannel = {};
    std::vector< double > a01_TestChannel = {};
    std::vector< double > a02_TestChannel = {};
    std::vector< double > a10_TestChannel = {};
    std::vector< double > a11_TestChannel = {};
    std::vector< double > a12_TestChannel = {};

    // parameters
    std::vector< double > gbar_TestChannel = {};
    std::vector< double > e_TestChannel = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_TestChannel = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_TestChannelMultichannelTestModel(){};
    ~i_TestChannelMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_TestChannel_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions TestChannel
double a00_inf_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a00_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a01_inf_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a01_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a02_inf_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a02_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a10_inf_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a10_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a11_inf_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a11_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double a12_inf_TestChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a12_TestChannel ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_SKv3_1MultichannelTestModel{
private:
    // states
    std::vector< double > z_SKv3_1 = {};

    // parameters
    std::vector< double > gbar_SKv3_1 = {};
    std::vector< double > e_SKv3_1 = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_SKv3_1 = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_SKv3_1MultichannelTestModel(){};
    ~i_SKv3_1MultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_SKv3_1_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions SKv3_1
double z_inf_SKv3_1 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_z_SKv3_1 ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_SK_E2MultichannelTestModel{
private:
    // states
    std::vector< double > z_SK_E2 = {};

    // parameters
    std::vector< double > gbar_SK_E2 = {};
    std::vector< double > e_SK_E2 = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_SK_E2 = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_SK_E2MultichannelTestModel(){};
    ~i_SK_E2MultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_SK_E2_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp, std::vector< double > c_ca);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions SK_E2
double z_inf_SK_E2 ( double v_comp, double ca) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_z_SK_E2 ( double v_comp, double ca) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_SKMultichannelTestModel{
private:
    // states
    std::vector< double > z_SK = {};

    // parameters
    std::vector< double > gbar_SK = {};
    std::vector< double > e_SK = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_SK = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_SKMultichannelTestModel(){};
    ~i_SKMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_SK_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp, std::vector< double > c_ca);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions SK
double z_inf_SK ( double v_comp, double ca) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_z_SK ( double v_comp, double ca) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_PiecewiseChannelMultichannelTestModel{
private:
    // states
    std::vector< double > a_PiecewiseChannel = {};
    std::vector< double > b_PiecewiseChannel = {};

    // parameters
    std::vector< double > gbar_PiecewiseChannel = {};
    std::vector< double > e_PiecewiseChannel = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_PiecewiseChannel = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_PiecewiseChannelMultichannelTestModel(){};
    ~i_PiecewiseChannelMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_PiecewiseChannel_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions PiecewiseChannel
double a_inf_PiecewiseChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_a_PiecewiseChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double b_inf_PiecewiseChannel ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_b_PiecewiseChannel ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_Na_TaMultichannelTestModel{
private:
    // states
    std::vector< double > h_Na_Ta = {};
    std::vector< double > m_Na_Ta = {};

    // parameters
    std::vector< double > gbar_Na_Ta = {};
    std::vector< double > e_Na_Ta = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_Na_Ta = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_Na_TaMultichannelTestModel(){};
    ~i_Na_TaMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_Na_Ta_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions Na_Ta
double h_inf_Na_Ta ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_h_Na_Ta ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double m_inf_Na_Ta ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_m_Na_Ta ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_NaTa_tMultichannelTestModel{
private:
    // states
    std::vector< double > h_NaTa_t = {};
    std::vector< double > m_NaTa_t = {};

    // parameters
    std::vector< double > gbar_NaTa_t = {};
    std::vector< double > e_NaTa_t = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_NaTa_t = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_NaTa_tMultichannelTestModel(){};
    ~i_NaTa_tMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_NaTa_t_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions NaTa_t
double h_inf_NaTa_t ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_h_NaTa_t ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double m_inf_NaTa_t ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_m_NaTa_t ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_Kv3_1MultichannelTestModel{
private:
    // states
    std::vector< double > m_Kv3_1 = {};

    // parameters
    std::vector< double > gbar_Kv3_1 = {};
    std::vector< double > e_Kv3_1 = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_Kv3_1 = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_Kv3_1MultichannelTestModel(){};
    ~i_Kv3_1MultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_Kv3_1_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions Kv3_1
double m_inf_Kv3_1 ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_m_Kv3_1 ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_Ca_LVAstMultichannelTestModel{
private:
    // states
    std::vector< double > h_Ca_LVAst = {};
    std::vector< double > m_Ca_LVAst = {};

    // parameters
    std::vector< double > gbar_Ca_LVAst = {};
    std::vector< double > e_Ca_LVAst = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_Ca_LVAst = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_Ca_LVAstMultichannelTestModel(){};
    ~i_Ca_LVAstMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_Ca_LVAst_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions Ca_LVAst
double h_inf_Ca_LVAst ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_h_Ca_LVAst ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double m_inf_Ca_LVAst ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_m_Ca_LVAst ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};


class i_Ca_HVAMultichannelTestModel{
private:
    // states
    std::vector< double > h_Ca_HVA = {};
    std::vector< double > m_Ca_HVA = {};

    // parameters
    std::vector< double > gbar_Ca_HVA = {};
    std::vector< double > e_Ca_HVA = {};

    // internals

    // ion-channel root-inline value
    std::vector< double > i_tot_i_Ca_HVA = {};

    //zero recordable variable in case of zero contribution channel
    double zero_recordable = 0;

public:
    // constructor, destructor
    i_Ca_HVAMultichannelTestModel(){};
    ~i_Ca_HVAMultichannelTestModel(){};

    void new_channel(std::size_t comp_ass);
    void new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params);

    //number of channels
    std::size_t neuron_i_Ca_HVA_channel_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization channel
    void pre_run_hook() {
    };

    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp);

    // function declarations
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//  functions Ca_HVA
double h_inf_Ca_HVA ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_h_Ca_HVA ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double m_inf_Ca_HVA ( double v_comp) const;
    #pragma omp declare simd
    __attribute__((always_inline)) inline 
//
double tau_m_Ca_HVA ( double v_comp) const;

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};
///////////////////////////////////////////// concentrations

class c_caMultichannelTestModel{
private:
    // parameters
    std::vector< double > gamma_ca = {};
    std::vector< double > tau_ca = {};
    std::vector< double > inf_ca = {};

    // states

    // internals

    // concentration value (root-ode state)
    std::vector< double > c_ca = {};

    //zero recordable variable in case of zero contribution concentration
    double zero_recordable = 0;

public:
    // constructor, destructor
    c_caMultichannelTestModel(){};
    ~c_caMultichannelTestModel(){};

    void new_concentration(std::size_t comp_ass);
    void new_concentration(std::size_t comp_ass, const DictionaryDatum& concentration_params);

    //number of channels
    std::size_t neuron_c_ca_concentration_count = 0;

    std::vector< size_t > compartment_association = {};

    // initialization concentration
    void pre_run_hook() {
    for(std::size_t concentration_id = 0; concentration_id < neuron_c_ca_concentration_count; concentration_id++){
    // states
    }
    };
    void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);

    // numerical integration step
    void f_numstep( std::vector< double > v_comp
                        , std::vector< double > i_Ca_HVA, std::vector< double > i_Ca_LVAst);

    // function declarations

    // root_ode getter
    void get_concentrations_per_compartment(std::vector< double >& compartment_to_concentration);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};
////////////////////////////////////////////////// synapses



class i_AMPAMultichannelTestModel{
private:
  // global synapse index
  std::vector< long > syn_idx = {};

  // propagators, initialized via pre_run_hook() or calibrate()
  std::vector< double > __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA;
  std::vector< double > __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA__d;
  std::vector< double > __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA;
  std::vector< double > __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA__d;

  // kernel state variables, initialized via pre_run_hook() or calibrate()
  std::vector< double > g_AMPA__X__spikes_AMPA;
  std::vector< double > g_AMPA__X__spikes_AMPA__d;

  // user defined parameters, initialized via pre_run_hook() or calibrate()
  std::vector< double > e_AMPA;
  std::vector< double > tau_r_AMPA;
  std::vector< double > tau_d_AMPA;

      // states

  std::vector< double > i_tot_i_AMPA = {};

  // user declared internals in order they were declared, initialized via pre_run_hook() or calibrate()
  std::vector< double > tp_AMPA;
  std::vector< double > g_norm_AMPA;



  // spike buffer
  std::vector< RingBuffer* > spikes_AMPA_;

public:
  // constructor, destructor
  i_AMPAMultichannelTestModel(){};
  ~i_AMPAMultichannelTestModel(){};

  void new_synapse(std::size_t comp_ass, const long syn_index);
  void new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params);

  //number of synapses
  std::size_t neuron_i_AMPA_synapse_count = 0;

  std::vector< size_t > compartment_association = {};

  // numerical integration step
  std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp, const long lag );

  // calibration
  void pre_run_hook();
  void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);
  void set_buffer_ptr( std::vector< RingBuffer >& syn_buffers )
  {
    for(std::size_t i = 0; i < syn_idx.size(); i++){
        spikes_AMPA_.push_back(&(syn_buffers[syn_idx[i]]));
    }
  };

  // function declarations

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};



class i_GABAMultichannelTestModel{
private:
  // global synapse index
  std::vector< long > syn_idx = {};

  // propagators, initialized via pre_run_hook() or calibrate()
  std::vector< double > __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA;
  std::vector< double > __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA__d;
  std::vector< double > __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA;
  std::vector< double > __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA__d;

  // kernel state variables, initialized via pre_run_hook() or calibrate()
  std::vector< double > g_GABA__X__spikes_GABA;
  std::vector< double > g_GABA__X__spikes_GABA__d;

  // user defined parameters, initialized via pre_run_hook() or calibrate()
  std::vector< double > e_GABA;
  std::vector< double > tau_r_GABA;
  std::vector< double > tau_d_GABA;

      // states

  std::vector< double > i_tot_i_GABA = {};

  // user declared internals in order they were declared, initialized via pre_run_hook() or calibrate()
  std::vector< double > tp_GABA;
  std::vector< double > g_norm_GABA;



  // spike buffer
  std::vector< RingBuffer* > spikes_GABA_;

public:
  // constructor, destructor
  i_GABAMultichannelTestModel(){};
  ~i_GABAMultichannelTestModel(){};

  void new_synapse(std::size_t comp_ass, const long syn_index);
  void new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params);

  //number of synapses
  std::size_t neuron_i_GABA_synapse_count = 0;

  std::vector< size_t > compartment_association = {};

  // numerical integration step
  std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp, const long lag );

  // calibration
  void pre_run_hook();
  void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);
  void set_buffer_ptr( std::vector< RingBuffer >& syn_buffers )
  {
    for(std::size_t i = 0; i < syn_idx.size(); i++){
        spikes_GABA_.push_back(&(syn_buffers[syn_idx[i]]));
    }
  };

  // function declarations

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};



class i_NMDAMultichannelTestModel{
private:
  // global synapse index
  std::vector< long > syn_idx = {};

  // propagators, initialized via pre_run_hook() or calibrate()
  std::vector< double > __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA;
  std::vector< double > __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA__d;
  std::vector< double > __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA;
  std::vector< double > __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA__d;

  // kernel state variables, initialized via pre_run_hook() or calibrate()
  std::vector< double > g_NMDA__X__spikes_NMDA;
  std::vector< double > g_NMDA__X__spikes_NMDA__d;

  // user defined parameters, initialized via pre_run_hook() or calibrate()
  std::vector< double > e_NMDA;
  std::vector< double > tau_r_NMDA;
  std::vector< double > tau_d_NMDA;

      // states

  std::vector< double > i_tot_i_NMDA = {};

  // user declared internals in order they were declared, initialized via pre_run_hook() or calibrate()
  std::vector< double > tp_NMDA;
  std::vector< double > g_norm_NMDA;



  // spike buffer
  std::vector< RingBuffer* > spikes_NMDA_;

public:
  // constructor, destructor
  i_NMDAMultichannelTestModel(){};
  ~i_NMDAMultichannelTestModel(){};

  void new_synapse(std::size_t comp_ass, const long syn_index);
  void new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params);

  //number of synapses
  std::size_t neuron_i_NMDA_synapse_count = 0;

  std::vector< size_t > compartment_association = {};

  // numerical integration step
  std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp, const long lag );

  // calibration
  void pre_run_hook();
  void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);
  void set_buffer_ptr( std::vector< RingBuffer >& syn_buffers )
  {
    for(std::size_t i = 0; i < syn_idx.size(); i++){
        spikes_NMDA_.push_back(&(syn_buffers[syn_idx[i]]));
    }
  };

  // function declarations

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};



class i_AMPA_NMDAMultichannelTestModel{
private:
  // global synapse index
  std::vector< long > syn_idx = {};

  // propagators, initialized via pre_run_hook() or calibrate()
  std::vector< double > __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN;
  std::vector< double > __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN__d;
  std::vector< double > __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN;
  std::vector< double > __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN__d;
  std::vector< double > __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN;
  std::vector< double > __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN__d;
  std::vector< double > __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN;
  std::vector< double > __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN__d;

  // kernel state variables, initialized via pre_run_hook() or calibrate()
  std::vector< double > g_AN_NMDA__X__spikes_AN;
  std::vector< double > g_AN_NMDA__X__spikes_AN__d;
  std::vector< double > g_AN_AMPA__X__spikes_AN;
  std::vector< double > g_AN_AMPA__X__spikes_AN__d;

  // user defined parameters, initialized via pre_run_hook() or calibrate()
  std::vector< double > e_AN_AMPA;
  std::vector< double > tau_r_AN_AMPA;
  std::vector< double > tau_d_AN_AMPA;
  std::vector< double > e_AN_NMDA;
  std::vector< double > tau_r_AN_NMDA;
  std::vector< double > tau_d_AN_NMDA;
  std::vector< double > NMDA_ratio;

      // states

  std::vector< double > i_tot_i_AMPA_NMDA = {};

  // user declared internals in order they were declared, initialized via pre_run_hook() or calibrate()
  std::vector< double > tp_AN_AMPA;
  std::vector< double > g_norm_AN_AMPA;
  std::vector< double > tp_AN_NMDA;
  std::vector< double > g_norm_AN_NMDA;



  // spike buffer
  std::vector< RingBuffer* > spikes_AN_;

public:
  // constructor, destructor
  i_AMPA_NMDAMultichannelTestModel(){};
  ~i_AMPA_NMDAMultichannelTestModel(){};

  void new_synapse(std::size_t comp_ass, const long syn_index);
  void new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params);

  //number of synapses
  std::size_t neuron_i_AMPA_NMDA_synapse_count = 0;

  std::vector< size_t > compartment_association = {};

  // numerical integration step
  std::pair< std::vector< double >, std::vector< double > > f_numstep( std::vector< double > v_comp, const long lag );

  // calibration
  void pre_run_hook();
  void append_recordables(std::map< Name, double* >* recordables, const long compartment_idx);
  void set_buffer_ptr( std::vector< RingBuffer >& syn_buffers )
  {
    for(std::size_t i = 0; i < syn_idx.size(); i++){
        spikes_AN_.push_back(&(syn_buffers[syn_idx[i]]));
    }
  };

  // function declarations

    // root_inline getter
    void get_currents_per_compartment(std::vector< double >& compartment_to_current);

    std::vector< double > distribute_shared_vector(std::vector< double > shared_vector);

};

////////////////////////////////////////////////// continuous inputs///////////////////////////////////////////// currents

class NeuronCurrentsMultichannelTestModel {
private:
  //mechanisms
  // ion channels

  i_hMultichannelTestModel i_h_chan_;
  
  i_TestChannel2MultichannelTestModel i_TestChannel2_chan_;
  
  i_TestChannelMultichannelTestModel i_TestChannel_chan_;
  
  i_SKv3_1MultichannelTestModel i_SKv3_1_chan_;
  
  i_SK_E2MultichannelTestModel i_SK_E2_chan_;
  
  i_SKMultichannelTestModel i_SK_chan_;
  
  i_PiecewiseChannelMultichannelTestModel i_PiecewiseChannel_chan_;
  
  i_Na_TaMultichannelTestModel i_Na_Ta_chan_;
  
  i_NaTa_tMultichannelTestModel i_NaTa_t_chan_;
  
  i_Kv3_1MultichannelTestModel i_Kv3_1_chan_;
  
  i_Ca_LVAstMultichannelTestModel i_Ca_LVAst_chan_;
  
  i_Ca_HVAMultichannelTestModel i_Ca_HVA_chan_;
  
  // concentrations

  c_caMultichannelTestModel c_ca_conc_;
  
  // synapses

  i_AMPAMultichannelTestModel i_AMPA_syn_;
  
  i_GABAMultichannelTestModel i_GABA_syn_;
  
  i_NMDAMultichannelTestModel i_NMDA_syn_;
  
  i_AMPA_NMDAMultichannelTestModel i_AMPA_NMDA_syn_;
  
  // continuous inputs


  //number of compartments
  std::size_t compartment_number = 0;

  //interdependency shared reference vectors and consecutive area vectors
  // ion channels

  std::vector < double > i_h_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_h_chan__con_area;
  
  std::vector < double > i_TestChannel2_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_TestChannel2_chan__con_area;
  
  std::vector < double > i_TestChannel_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_TestChannel_chan__con_area;
  
  std::vector < double > i_SKv3_1_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_SKv3_1_chan__con_area;
  
  std::vector < double > i_SK_E2_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_SK_E2_chan__con_area;
  
  std::vector < double > i_SK_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_SK_chan__con_area;
  
  std::vector < double > i_PiecewiseChannel_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_PiecewiseChannel_chan__con_area;
  
  std::vector < double > i_Na_Ta_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_Na_Ta_chan__con_area;
  
  std::vector < double > i_NaTa_t_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_NaTa_t_chan__con_area;
  
  std::vector < double > i_Kv3_1_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_Kv3_1_chan__con_area;
  
  std::vector < double > i_Ca_LVAst_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_Ca_LVAst_chan__con_area;
  
  std::vector < double > i_Ca_HVA_chan__shared_current;
  std::vector < std::pair< std::size_t, int > > i_Ca_HVA_chan__con_area;
  
  // concentrations

  std::vector < double > c_ca_conc__shared_concentration;
  std::vector < std::pair< std::size_t, int > > c_ca_conc__con_area;
  
  // synapses

  std::vector < double > i_AMPA_syn__shared_current;
  std::vector < std::pair< std::size_t, int > > i_AMPA_syn__con_area;
  
  std::vector < double > i_GABA_syn__shared_current;
  std::vector < std::pair< std::size_t, int > > i_GABA_syn__con_area;
  
  std::vector < double > i_NMDA_syn__shared_current;
  std::vector < std::pair< std::size_t, int > > i_NMDA_syn__con_area;
  
  std::vector < double > i_AMPA_NMDA_syn__shared_current;
  std::vector < std::pair< std::size_t, int > > i_AMPA_NMDA_syn__con_area;
  
// continuous inputs


  //compartment gi states
  std::vector < std::pair < double, double > > comps_gi;

public:
  NeuronCurrentsMultichannelTestModel(){};
  ~NeuronCurrentsMultichannelTestModel(){};
  void pre_run_hook() {
    // initialization of ion channels
    i_h_chan_.pre_run_hook();
    
    i_TestChannel2_chan_.pre_run_hook();
    
    i_TestChannel_chan_.pre_run_hook();
    
    i_SKv3_1_chan_.pre_run_hook();
    
    i_SK_E2_chan_.pre_run_hook();
    
    i_SK_chan_.pre_run_hook();
    
    i_PiecewiseChannel_chan_.pre_run_hook();
    
    i_Na_Ta_chan_.pre_run_hook();
    
    i_NaTa_t_chan_.pre_run_hook();
    
    i_Kv3_1_chan_.pre_run_hook();
    
    i_Ca_LVAst_chan_.pre_run_hook();
    
    i_Ca_HVA_chan_.pre_run_hook();
    
    c_ca_conc_.pre_run_hook();
    
    i_AMPA_syn_.pre_run_hook();
    
    i_GABA_syn_.pre_run_hook();
    
    i_NMDA_syn_.pre_run_hook();
    
    i_AMPA_NMDA_syn_.pre_run_hook();
    
    int con_end_index;
    if(i_h_chan_.neuron_i_h_channel_count){
        con_end_index = int(i_h_chan_.compartment_association[0]);
        i_h_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_h_chan_.neuron_i_h_channel_count; chan_id++){
        if(!(i_h_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_h_chan_.compartment_association[chan_id]) - int(chan_id);
            i_h_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_TestChannel2_chan_.neuron_i_TestChannel2_channel_count){
        con_end_index = int(i_TestChannel2_chan_.compartment_association[0]);
        i_TestChannel2_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_TestChannel2_chan_.neuron_i_TestChannel2_channel_count; chan_id++){
        if(!(i_TestChannel2_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_TestChannel2_chan_.compartment_association[chan_id]) - int(chan_id);
            i_TestChannel2_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_TestChannel_chan_.neuron_i_TestChannel_channel_count){
        con_end_index = int(i_TestChannel_chan_.compartment_association[0]);
        i_TestChannel_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_TestChannel_chan_.neuron_i_TestChannel_channel_count; chan_id++){
        if(!(i_TestChannel_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_TestChannel_chan_.compartment_association[chan_id]) - int(chan_id);
            i_TestChannel_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_SKv3_1_chan_.neuron_i_SKv3_1_channel_count){
        con_end_index = int(i_SKv3_1_chan_.compartment_association[0]);
        i_SKv3_1_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_SKv3_1_chan_.neuron_i_SKv3_1_channel_count; chan_id++){
        if(!(i_SKv3_1_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_SKv3_1_chan_.compartment_association[chan_id]) - int(chan_id);
            i_SKv3_1_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_SK_E2_chan_.neuron_i_SK_E2_channel_count){
        con_end_index = int(i_SK_E2_chan_.compartment_association[0]);
        i_SK_E2_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_SK_E2_chan_.neuron_i_SK_E2_channel_count; chan_id++){
        if(!(i_SK_E2_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_SK_E2_chan_.compartment_association[chan_id]) - int(chan_id);
            i_SK_E2_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_SK_chan_.neuron_i_SK_channel_count){
        con_end_index = int(i_SK_chan_.compartment_association[0]);
        i_SK_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_SK_chan_.neuron_i_SK_channel_count; chan_id++){
        if(!(i_SK_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_SK_chan_.compartment_association[chan_id]) - int(chan_id);
            i_SK_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_PiecewiseChannel_chan_.neuron_i_PiecewiseChannel_channel_count){
        con_end_index = int(i_PiecewiseChannel_chan_.compartment_association[0]);
        i_PiecewiseChannel_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_PiecewiseChannel_chan_.neuron_i_PiecewiseChannel_channel_count; chan_id++){
        if(!(i_PiecewiseChannel_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_PiecewiseChannel_chan_.compartment_association[chan_id]) - int(chan_id);
            i_PiecewiseChannel_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_Na_Ta_chan_.neuron_i_Na_Ta_channel_count){
        con_end_index = int(i_Na_Ta_chan_.compartment_association[0]);
        i_Na_Ta_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_Na_Ta_chan_.neuron_i_Na_Ta_channel_count; chan_id++){
        if(!(i_Na_Ta_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_Na_Ta_chan_.compartment_association[chan_id]) - int(chan_id);
            i_Na_Ta_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_NaTa_t_chan_.neuron_i_NaTa_t_channel_count){
        con_end_index = int(i_NaTa_t_chan_.compartment_association[0]);
        i_NaTa_t_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_NaTa_t_chan_.neuron_i_NaTa_t_channel_count; chan_id++){
        if(!(i_NaTa_t_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_NaTa_t_chan_.compartment_association[chan_id]) - int(chan_id);
            i_NaTa_t_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_Kv3_1_chan_.neuron_i_Kv3_1_channel_count){
        con_end_index = int(i_Kv3_1_chan_.compartment_association[0]);
        i_Kv3_1_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_Kv3_1_chan_.neuron_i_Kv3_1_channel_count; chan_id++){
        if(!(i_Kv3_1_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_Kv3_1_chan_.compartment_association[chan_id]) - int(chan_id);
            i_Kv3_1_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_Ca_LVAst_chan_.neuron_i_Ca_LVAst_channel_count){
        con_end_index = int(i_Ca_LVAst_chan_.compartment_association[0]);
        i_Ca_LVAst_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_Ca_LVAst_chan_.neuron_i_Ca_LVAst_channel_count; chan_id++){
        if(!(i_Ca_LVAst_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_Ca_LVAst_chan_.compartment_association[chan_id]) - int(chan_id);
            i_Ca_LVAst_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(i_Ca_HVA_chan_.neuron_i_Ca_HVA_channel_count){
        con_end_index = int(i_Ca_HVA_chan_.compartment_association[0]);
        i_Ca_HVA_chan__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t chan_id = 1; chan_id < i_Ca_HVA_chan_.neuron_i_Ca_HVA_channel_count; chan_id++){
        if(!(i_Ca_HVA_chan_.compartment_association[chan_id] == size_t(int(chan_id) + con_end_index))){
            con_end_index = int(i_Ca_HVA_chan_.compartment_association[chan_id]) - int(chan_id);
            i_Ca_HVA_chan__con_area.push_back(std::pair< std::size_t, int >(chan_id, con_end_index));
        }
    }
    
    if(c_ca_conc_.neuron_c_ca_concentration_count){
        con_end_index = int(c_ca_conc_.compartment_association[0]);
        c_ca_conc__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t conc_id = 0; conc_id < c_ca_conc_.neuron_c_ca_concentration_count; conc_id++){
        if(!(c_ca_conc_.compartment_association[conc_id] == size_t(int(conc_id) + con_end_index))){
            con_end_index = int(c_ca_conc_.compartment_association[conc_id]) - int(conc_id);
            c_ca_conc__con_area.push_back(std::pair< std::size_t, int >(conc_id, con_end_index));
        }
    }
    
    if(i_AMPA_syn_.neuron_i_AMPA_synapse_count){
        con_end_index = int(i_AMPA_syn_.compartment_association[0]);
        i_AMPA_syn__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t syn_id = 0; syn_id < i_AMPA_syn_.neuron_i_AMPA_synapse_count; syn_id++){
        if(!(i_AMPA_syn_.compartment_association[syn_id] == size_t(int(syn_id) + con_end_index))){
            con_end_index = int(i_AMPA_syn_.compartment_association[syn_id]) - int(syn_id);
            i_AMPA_syn__con_area.push_back(std::pair< std::size_t, int >(syn_id, con_end_index));
        }
    }
    
    if(i_GABA_syn_.neuron_i_GABA_synapse_count){
        con_end_index = int(i_GABA_syn_.compartment_association[0]);
        i_GABA_syn__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t syn_id = 0; syn_id < i_GABA_syn_.neuron_i_GABA_synapse_count; syn_id++){
        if(!(i_GABA_syn_.compartment_association[syn_id] == size_t(int(syn_id) + con_end_index))){
            con_end_index = int(i_GABA_syn_.compartment_association[syn_id]) - int(syn_id);
            i_GABA_syn__con_area.push_back(std::pair< std::size_t, int >(syn_id, con_end_index));
        }
    }
    
    if(i_NMDA_syn_.neuron_i_NMDA_synapse_count){
        con_end_index = int(i_NMDA_syn_.compartment_association[0]);
        i_NMDA_syn__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t syn_id = 0; syn_id < i_NMDA_syn_.neuron_i_NMDA_synapse_count; syn_id++){
        if(!(i_NMDA_syn_.compartment_association[syn_id] == size_t(int(syn_id) + con_end_index))){
            con_end_index = int(i_NMDA_syn_.compartment_association[syn_id]) - int(syn_id);
            i_NMDA_syn__con_area.push_back(std::pair< std::size_t, int >(syn_id, con_end_index));
        }
    }
    
    if(i_AMPA_NMDA_syn_.neuron_i_AMPA_NMDA_synapse_count){
        con_end_index = int(i_AMPA_NMDA_syn_.compartment_association[0]);
        i_AMPA_NMDA_syn__con_area.push_back(std::pair< std::size_t, int >(0, con_end_index));
    }
    for(std::size_t syn_id = 0; syn_id < i_AMPA_NMDA_syn_.neuron_i_AMPA_NMDA_synapse_count; syn_id++){
        if(!(i_AMPA_NMDA_syn_.compartment_association[syn_id] == size_t(int(syn_id) + con_end_index))){
            con_end_index = int(i_AMPA_NMDA_syn_.compartment_association[syn_id]) - int(syn_id);
            i_AMPA_NMDA_syn__con_area.push_back(std::pair< std::size_t, int >(syn_id, con_end_index));
        }
    }
    };

  void add_mechanism( const std::string& type, const std::size_t compartment_id, const long multi_mech_index = 0)
  {
    bool mech_found = false;
    if ( type == "i_h" )
    {
      i_h_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_TestChannel2" )
    {
      i_TestChannel2_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_TestChannel" )
    {
      i_TestChannel_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_SKv3_1" )
    {
      i_SKv3_1_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_SK_E2" )
    {
      i_SK_E2_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_SK" )
    {
      i_SK_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_PiecewiseChannel" )
    {
      i_PiecewiseChannel_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_Na_Ta" )
    {
      i_Na_Ta_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_NaTa_t" )
    {
      i_NaTa_t_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_Kv3_1" )
    {
      i_Kv3_1_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_Ca_LVAst" )
    {
      i_Ca_LVAst_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_Ca_HVA" )
    {
      i_Ca_HVA_chan_.new_channel(compartment_id);
      mech_found = true;
    }
  
    if ( type == "c_ca" )
    {
      c_ca_conc_.new_concentration(compartment_id);
      mech_found = true;
    }
  
    if ( type == "i_AMPA" )
    {
      i_AMPA_syn_.new_synapse(compartment_id, multi_mech_index);
      mech_found = true;
    }
  
    if ( type == "i_GABA" )
    {
      i_GABA_syn_.new_synapse(compartment_id, multi_mech_index);
      mech_found = true;
    }
  
    if ( type == "i_NMDA" )
    {
      i_NMDA_syn_.new_synapse(compartment_id, multi_mech_index);
      mech_found = true;
    }
  
    if ( type == "i_AMPA_NMDA" )
    {
      i_AMPA_NMDA_syn_.new_synapse(compartment_id, multi_mech_index);
      mech_found = true;
    }
  if(!mech_found)
    {
      assert( false );
    }
  };

  void add_mechanism( const std::string& type, const std::size_t compartment_id, const DictionaryDatum& mechanism_params, const long multi_mech_index = 0)
  {
    bool mech_found = false;
    if ( type == "i_h" )
    {
      i_h_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_TestChannel2" )
    {
      i_TestChannel2_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_TestChannel" )
    {
      i_TestChannel_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_SKv3_1" )
    {
      i_SKv3_1_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_SK_E2" )
    {
      i_SK_E2_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_SK" )
    {
      i_SK_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_PiecewiseChannel" )
    {
      i_PiecewiseChannel_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_Na_Ta" )
    {
      i_Na_Ta_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_NaTa_t" )
    {
      i_NaTa_t_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_Kv3_1" )
    {
      i_Kv3_1_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_Ca_LVAst" )
    {
      i_Ca_LVAst_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_Ca_HVA" )
    {
      i_Ca_HVA_chan_.new_channel(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "c_ca" )
    {
      c_ca_conc_.new_concentration(compartment_id, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_AMPA" )
    {
      i_AMPA_syn_.new_synapse(compartment_id, multi_mech_index, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_GABA" )
    {
      i_GABA_syn_.new_synapse(compartment_id, multi_mech_index, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_NMDA" )
    {
      i_NMDA_syn_.new_synapse(compartment_id, multi_mech_index, mechanism_params);
      mech_found = true;
    }
  
    if ( type == "i_AMPA_NMDA" )
    {
      i_AMPA_NMDA_syn_.new_synapse(compartment_id, multi_mech_index, mechanism_params);
      mech_found = true;
    }
  if(!mech_found)
    {
      assert( false );
    }
  };

  void add_compartment(){
    this->add_mechanism("i_h", compartment_number);
    
    this->add_mechanism("i_TestChannel2", compartment_number);
    
    this->add_mechanism("i_TestChannel", compartment_number);
    
    this->add_mechanism("i_SKv3_1", compartment_number);
    
    this->add_mechanism("i_SK_E2", compartment_number);
    
    this->add_mechanism("i_SK", compartment_number);
    
    this->add_mechanism("i_PiecewiseChannel", compartment_number);
    
    this->add_mechanism("i_Na_Ta", compartment_number);
    
    this->add_mechanism("i_NaTa_t", compartment_number);
    
    this->add_mechanism("i_Kv3_1", compartment_number);
    
    this->add_mechanism("i_Ca_LVAst", compartment_number);
    
    this->add_mechanism("i_Ca_HVA", compartment_number);
    
    this->add_mechanism("c_ca", compartment_number);
    compartment_number++;
    this->i_h_chan__shared_current.push_back(0.0);
    
    this->i_TestChannel2_chan__shared_current.push_back(0.0);
    
    this->i_TestChannel_chan__shared_current.push_back(0.0);
    
    this->i_SKv3_1_chan__shared_current.push_back(0.0);
    
    this->i_SK_E2_chan__shared_current.push_back(0.0);
    
    this->i_SK_chan__shared_current.push_back(0.0);
    
    this->i_PiecewiseChannel_chan__shared_current.push_back(0.0);
    
    this->i_Na_Ta_chan__shared_current.push_back(0.0);
    
    this->i_NaTa_t_chan__shared_current.push_back(0.0);
    
    this->i_Kv3_1_chan__shared_current.push_back(0.0);
    
    this->i_Ca_LVAst_chan__shared_current.push_back(0.0);
    
    this->i_Ca_HVA_chan__shared_current.push_back(0.0);
    
    this->c_ca_conc__shared_concentration.push_back(0.0);
    
    this->i_AMPA_syn__shared_current.push_back(0.0);
    
    this->i_GABA_syn__shared_current.push_back(0.0);
    
    this->i_NMDA_syn__shared_current.push_back(0.0);
    
    this->i_AMPA_NMDA_syn__shared_current.push_back(0.0);
    };

  void add_compartment(const DictionaryDatum& compartment_params){
    this->add_mechanism("i_h", compartment_number, compartment_params);
    
    this->add_mechanism("i_TestChannel2", compartment_number, compartment_params);
    
    this->add_mechanism("i_TestChannel", compartment_number, compartment_params);
    
    this->add_mechanism("i_SKv3_1", compartment_number, compartment_params);
    
    this->add_mechanism("i_SK_E2", compartment_number, compartment_params);
    
    this->add_mechanism("i_SK", compartment_number, compartment_params);
    
    this->add_mechanism("i_PiecewiseChannel", compartment_number, compartment_params);
    
    this->add_mechanism("i_Na_Ta", compartment_number, compartment_params);
    
    this->add_mechanism("i_NaTa_t", compartment_number, compartment_params);
    
    this->add_mechanism("i_Kv3_1", compartment_number, compartment_params);
    
    this->add_mechanism("i_Ca_LVAst", compartment_number, compartment_params);
    
    this->add_mechanism("i_Ca_HVA", compartment_number, compartment_params);
    
    this->add_mechanism("c_ca", compartment_number, compartment_params);
    compartment_number++;
    this->i_h_chan__shared_current.push_back(0.0);
    
    this->i_TestChannel2_chan__shared_current.push_back(0.0);
    
    this->i_TestChannel_chan__shared_current.push_back(0.0);
    
    this->i_SKv3_1_chan__shared_current.push_back(0.0);
    
    this->i_SK_E2_chan__shared_current.push_back(0.0);
    
    this->i_SK_chan__shared_current.push_back(0.0);
    
    this->i_PiecewiseChannel_chan__shared_current.push_back(0.0);
    
    this->i_Na_Ta_chan__shared_current.push_back(0.0);
    
    this->i_NaTa_t_chan__shared_current.push_back(0.0);
    
    this->i_Kv3_1_chan__shared_current.push_back(0.0);
    
    this->i_Ca_LVAst_chan__shared_current.push_back(0.0);
    
    this->i_Ca_HVA_chan__shared_current.push_back(0.0);
    
    this->c_ca_conc__shared_concentration.push_back(0.0);
    
    this->i_AMPA_syn__shared_current.push_back(0.0);
    
    this->i_GABA_syn__shared_current.push_back(0.0);
    
    this->i_NMDA_syn__shared_current.push_back(0.0);
    
    this->i_AMPA_NMDA_syn__shared_current.push_back(0.0);
    };

  void add_receptor_info( ArrayDatum& ad, long compartment_index )
  {
    for( std::size_t syn_it = 0; syn_it != i_AMPA_syn_.neuron_i_AMPA_synapse_count; syn_it++)
    {
      DictionaryDatum dd = DictionaryDatum( new Dictionary );
      def< long >( dd, names::receptor_idx, syn_it );
      def< long >( dd, names::comp_idx, compartment_index );
      def< std::string >( dd, names::receptor_type, "i_AMPA" );
      ad.push_back( dd );
    }
    
    for( std::size_t syn_it = 0; syn_it != i_GABA_syn_.neuron_i_GABA_synapse_count; syn_it++)
    {
      DictionaryDatum dd = DictionaryDatum( new Dictionary );
      def< long >( dd, names::receptor_idx, syn_it );
      def< long >( dd, names::comp_idx, compartment_index );
      def< std::string >( dd, names::receptor_type, "i_GABA" );
      ad.push_back( dd );
    }
    
    for( std::size_t syn_it = 0; syn_it != i_NMDA_syn_.neuron_i_NMDA_synapse_count; syn_it++)
    {
      DictionaryDatum dd = DictionaryDatum( new Dictionary );
      def< long >( dd, names::receptor_idx, syn_it );
      def< long >( dd, names::comp_idx, compartment_index );
      def< std::string >( dd, names::receptor_type, "i_NMDA" );
      ad.push_back( dd );
    }
    
    for( std::size_t syn_it = 0; syn_it != i_AMPA_NMDA_syn_.neuron_i_AMPA_NMDA_synapse_count; syn_it++)
    {
      DictionaryDatum dd = DictionaryDatum( new Dictionary );
      def< long >( dd, names::receptor_idx, syn_it );
      def< long >( dd, names::comp_idx, compartment_index );
      def< std::string >( dd, names::receptor_type, "i_AMPA_NMDA" );
      ad.push_back( dd );
    }
    };

  void set_buffers( std::vector< RingBuffer >& buffers)
  {
    // spike and continuous buffers for synapses and continuous inputs
      i_AMPA_syn_.set_buffer_ptr( buffers );
    
      i_GABA_syn_.set_buffer_ptr( buffers );
    
      i_NMDA_syn_.set_buffer_ptr( buffers );
    
      i_AMPA_NMDA_syn_.set_buffer_ptr( buffers );
    

  };

  std::map< Name, double* > get_recordables( const long compartment_idx )
  {
    std::map< Name, double* > recordables;

    // append ion channel state variables to recordables
    i_h_chan_.append_recordables( &recordables, compartment_idx );
    
    i_TestChannel2_chan_.append_recordables( &recordables, compartment_idx );
    
    i_TestChannel_chan_.append_recordables( &recordables, compartment_idx );
    
    i_SKv3_1_chan_.append_recordables( &recordables, compartment_idx );
    
    i_SK_E2_chan_.append_recordables( &recordables, compartment_idx );
    
    i_SK_chan_.append_recordables( &recordables, compartment_idx );
    
    i_PiecewiseChannel_chan_.append_recordables( &recordables, compartment_idx );
    
    i_Na_Ta_chan_.append_recordables( &recordables, compartment_idx );
    
    i_NaTa_t_chan_.append_recordables( &recordables, compartment_idx );
    
    i_Kv3_1_chan_.append_recordables( &recordables, compartment_idx );
    
    i_Ca_LVAst_chan_.append_recordables( &recordables, compartment_idx );
    
    i_Ca_HVA_chan_.append_recordables( &recordables, compartment_idx );
    
    

    // append concentration state variables to recordables
    c_ca_conc_.append_recordables( &recordables, compartment_idx );
    
    

    // append synapse state variables to recordables
    i_AMPA_syn_.append_recordables( &recordables, compartment_idx );
    
    i_GABA_syn_.append_recordables( &recordables, compartment_idx );
    
    i_NMDA_syn_.append_recordables( &recordables, compartment_idx );
    
    i_AMPA_NMDA_syn_.append_recordables( &recordables, compartment_idx );
    
    

    // append continuous input state variables to recordables
    

    return recordables;
  };

  std::vector< std::pair< double, double > > f_numstep( std::vector< double > v_comp_vec, const long lag )
  {
    std::vector< std::pair< double, double > > comp_to_gi(compartment_number, std::make_pair(0., 0.));
    i_AMPA_syn_.get_currents_per_compartment(i_AMPA_syn__shared_current);

    i_GABA_syn_.get_currents_per_compartment(i_GABA_syn__shared_current);

    i_NMDA_syn_.get_currents_per_compartment(i_NMDA_syn__shared_current);

    i_AMPA_NMDA_syn_.get_currents_per_compartment(i_AMPA_NMDA_syn__shared_current);

    c_ca_conc_.get_concentrations_per_compartment(c_ca_conc__shared_concentration);

    i_h_chan_.get_currents_per_compartment(i_h_chan__shared_current);

    i_TestChannel2_chan_.get_currents_per_compartment(i_TestChannel2_chan__shared_current);

    i_TestChannel_chan_.get_currents_per_compartment(i_TestChannel_chan__shared_current);

    i_SKv3_1_chan_.get_currents_per_compartment(i_SKv3_1_chan__shared_current);

    i_SK_E2_chan_.get_currents_per_compartment(i_SK_E2_chan__shared_current);

    i_SK_chan_.get_currents_per_compartment(i_SK_chan__shared_current);

    i_PiecewiseChannel_chan_.get_currents_per_compartment(i_PiecewiseChannel_chan__shared_current);

    i_Na_Ta_chan_.get_currents_per_compartment(i_Na_Ta_chan__shared_current);

    i_NaTa_t_chan_.get_currents_per_compartment(i_NaTa_t_chan__shared_current);

    i_Kv3_1_chan_.get_currents_per_compartment(i_Kv3_1_chan__shared_current);

    i_Ca_LVAst_chan_.get_currents_per_compartment(i_Ca_LVAst_chan__shared_current);

    i_Ca_HVA_chan_.get_currents_per_compartment(i_Ca_HVA_chan__shared_current);

    // computation of c_ca concentration
    c_ca_conc_.f_numstep( c_ca_conc_.distribute_shared_vector(v_comp_vec)
                        , c_ca_conc_.distribute_shared_vector(i_Ca_HVA_chan__shared_current), c_ca_conc_.distribute_shared_vector(i_Ca_LVAst_chan__shared_current));

    std::pair< std::vector< double >, std::vector< double > > gi_mech;
    std::size_t con_area_count;
    // contribution of i_h channel
    gi_mech = i_h_chan_.f_numstep( i_h_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_h_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_h_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_h_chan__con_area[con_area_index+1].first;
            int offset = i_h_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_h_chan__con_area[con_area_count-1].first;
        int offset = i_h_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_h_chan_.neuron_i_h_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_TestChannel2 channel
    gi_mech = i_TestChannel2_chan_.f_numstep( i_TestChannel2_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_TestChannel2_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_TestChannel2_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_TestChannel2_chan__con_area[con_area_index+1].first;
            int offset = i_TestChannel2_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_TestChannel2_chan__con_area[con_area_count-1].first;
        int offset = i_TestChannel2_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_TestChannel2_chan_.neuron_i_TestChannel2_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_TestChannel channel
    gi_mech = i_TestChannel_chan_.f_numstep( i_TestChannel_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_TestChannel_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_TestChannel_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_TestChannel_chan__con_area[con_area_index+1].first;
            int offset = i_TestChannel_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_TestChannel_chan__con_area[con_area_count-1].first;
        int offset = i_TestChannel_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_TestChannel_chan_.neuron_i_TestChannel_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_SKv3_1 channel
    gi_mech = i_SKv3_1_chan_.f_numstep( i_SKv3_1_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_SKv3_1_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_SKv3_1_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_SKv3_1_chan__con_area[con_area_index+1].first;
            int offset = i_SKv3_1_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_SKv3_1_chan__con_area[con_area_count-1].first;
        int offset = i_SKv3_1_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_SKv3_1_chan_.neuron_i_SKv3_1_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_SK_E2 channel
    gi_mech = i_SK_E2_chan_.f_numstep( i_SK_E2_chan_.distribute_shared_vector(v_comp_vec), i_SK_E2_chan_.distribute_shared_vector(c_ca_conc__shared_concentration));

    con_area_count = i_SK_E2_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_SK_E2_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_SK_E2_chan__con_area[con_area_index+1].first;
            int offset = i_SK_E2_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_SK_E2_chan__con_area[con_area_count-1].first;
        int offset = i_SK_E2_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_SK_E2_chan_.neuron_i_SK_E2_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_SK channel
    gi_mech = i_SK_chan_.f_numstep( i_SK_chan_.distribute_shared_vector(v_comp_vec), i_SK_chan_.distribute_shared_vector(c_ca_conc__shared_concentration));

    con_area_count = i_SK_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_SK_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_SK_chan__con_area[con_area_index+1].first;
            int offset = i_SK_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_SK_chan__con_area[con_area_count-1].first;
        int offset = i_SK_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_SK_chan_.neuron_i_SK_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_PiecewiseChannel channel
    gi_mech = i_PiecewiseChannel_chan_.f_numstep( i_PiecewiseChannel_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_PiecewiseChannel_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_PiecewiseChannel_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_PiecewiseChannel_chan__con_area[con_area_index+1].first;
            int offset = i_PiecewiseChannel_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_PiecewiseChannel_chan__con_area[con_area_count-1].first;
        int offset = i_PiecewiseChannel_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_PiecewiseChannel_chan_.neuron_i_PiecewiseChannel_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_Na_Ta channel
    gi_mech = i_Na_Ta_chan_.f_numstep( i_Na_Ta_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_Na_Ta_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_Na_Ta_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_Na_Ta_chan__con_area[con_area_index+1].first;
            int offset = i_Na_Ta_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_Na_Ta_chan__con_area[con_area_count-1].first;
        int offset = i_Na_Ta_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_Na_Ta_chan_.neuron_i_Na_Ta_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_NaTa_t channel
    gi_mech = i_NaTa_t_chan_.f_numstep( i_NaTa_t_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_NaTa_t_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_NaTa_t_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_NaTa_t_chan__con_area[con_area_index+1].first;
            int offset = i_NaTa_t_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_NaTa_t_chan__con_area[con_area_count-1].first;
        int offset = i_NaTa_t_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_NaTa_t_chan_.neuron_i_NaTa_t_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_Kv3_1 channel
    gi_mech = i_Kv3_1_chan_.f_numstep( i_Kv3_1_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_Kv3_1_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_Kv3_1_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_Kv3_1_chan__con_area[con_area_index+1].first;
            int offset = i_Kv3_1_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_Kv3_1_chan__con_area[con_area_count-1].first;
        int offset = i_Kv3_1_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_Kv3_1_chan_.neuron_i_Kv3_1_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_Ca_LVAst channel
    gi_mech = i_Ca_LVAst_chan_.f_numstep( i_Ca_LVAst_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_Ca_LVAst_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_Ca_LVAst_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_Ca_LVAst_chan__con_area[con_area_index+1].first;
            int offset = i_Ca_LVAst_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_Ca_LVAst_chan__con_area[con_area_count-1].first;
        int offset = i_Ca_LVAst_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_Ca_LVAst_chan_.neuron_i_Ca_LVAst_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_Ca_HVA channel
    gi_mech = i_Ca_HVA_chan_.f_numstep( i_Ca_HVA_chan_.distribute_shared_vector(v_comp_vec));

    con_area_count = i_Ca_HVA_chan__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_Ca_HVA_chan__con_area[con_area_index].first;
            std::size_t next_con_area = i_Ca_HVA_chan__con_area[con_area_index+1].first;
            int offset = i_Ca_HVA_chan__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t chan_id = con_area; chan_id < next_con_area; chan_id++){
                comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
                comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
            }
        }

        std::size_t con_area = i_Ca_HVA_chan__con_area[con_area_count-1].first;
        int offset = i_Ca_HVA_chan__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t chan_id = con_area; chan_id < i_Ca_HVA_chan_.neuron_i_Ca_HVA_channel_count; chan_id++){
            comp_to_gi[chan_id+offset].first += gi_mech.first[chan_id];
            comp_to_gi[chan_id+offset].second += gi_mech.second[chan_id];
        }
    }
    
    // contribution of i_AMPA synapses
    gi_mech = i_AMPA_syn_.f_numstep( i_AMPA_syn_.distribute_shared_vector(v_comp_vec), lag );

    con_area_count = i_AMPA_syn__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_AMPA_syn__con_area[con_area_index].first;
            std::size_t next_con_area = i_AMPA_syn__con_area[con_area_index+1].first;
            int offset = i_AMPA_syn__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t syn_id = con_area; syn_id < next_con_area; syn_id++){
                comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
                comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
            }
        }

        std::size_t con_area = i_AMPA_syn__con_area[con_area_count-1].first;
        int offset = i_AMPA_syn__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t syn_id = con_area; syn_id < i_AMPA_syn_.neuron_i_AMPA_synapse_count; syn_id++){
            comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
            comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
        }
    }
    
    // contribution of i_GABA synapses
    gi_mech = i_GABA_syn_.f_numstep( i_GABA_syn_.distribute_shared_vector(v_comp_vec), lag );

    con_area_count = i_GABA_syn__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_GABA_syn__con_area[con_area_index].first;
            std::size_t next_con_area = i_GABA_syn__con_area[con_area_index+1].first;
            int offset = i_GABA_syn__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t syn_id = con_area; syn_id < next_con_area; syn_id++){
                comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
                comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
            }
        }

        std::size_t con_area = i_GABA_syn__con_area[con_area_count-1].first;
        int offset = i_GABA_syn__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t syn_id = con_area; syn_id < i_GABA_syn_.neuron_i_GABA_synapse_count; syn_id++){
            comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
            comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
        }
    }
    
    // contribution of i_NMDA synapses
    gi_mech = i_NMDA_syn_.f_numstep( i_NMDA_syn_.distribute_shared_vector(v_comp_vec), lag );

    con_area_count = i_NMDA_syn__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_NMDA_syn__con_area[con_area_index].first;
            std::size_t next_con_area = i_NMDA_syn__con_area[con_area_index+1].first;
            int offset = i_NMDA_syn__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t syn_id = con_area; syn_id < next_con_area; syn_id++){
                comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
                comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
            }
        }

        std::size_t con_area = i_NMDA_syn__con_area[con_area_count-1].first;
        int offset = i_NMDA_syn__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t syn_id = con_area; syn_id < i_NMDA_syn_.neuron_i_NMDA_synapse_count; syn_id++){
            comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
            comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
        }
    }
    
    // contribution of i_AMPA_NMDA synapses
    gi_mech = i_AMPA_NMDA_syn_.f_numstep( i_AMPA_NMDA_syn_.distribute_shared_vector(v_comp_vec), lag );

    con_area_count = i_AMPA_NMDA_syn__con_area.size();
    if(con_area_count > 0){
        for(std::size_t con_area_index = 0; con_area_index < con_area_count-1; con_area_index++){
            std::size_t con_area = i_AMPA_NMDA_syn__con_area[con_area_index].first;
            std::size_t next_con_area = i_AMPA_NMDA_syn__con_area[con_area_index+1].first;
            int offset = i_AMPA_NMDA_syn__con_area[con_area_index].second;

            #pragma omp simd
            for(std::size_t syn_id = con_area; syn_id < next_con_area; syn_id++){
                comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
                comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
            }
        }

        std::size_t con_area = i_AMPA_NMDA_syn__con_area[con_area_count-1].first;
        int offset = i_AMPA_NMDA_syn__con_area[con_area_count-1].second;

        #pragma omp simd
        for(std::size_t syn_id = con_area; syn_id < i_AMPA_NMDA_syn_.neuron_i_AMPA_NMDA_synapse_count; syn_id++){
            comp_to_gi[syn_id+offset].first += gi_mech.first[syn_id];
            comp_to_gi[syn_id+offset].second += gi_mech.second[syn_id];
        }
    }
    return comp_to_gi;
  };
};

} // namespace

#endif /* #ifndef SYNAPSES_NEAT_H_MULTICHANNELTESTMODEL */