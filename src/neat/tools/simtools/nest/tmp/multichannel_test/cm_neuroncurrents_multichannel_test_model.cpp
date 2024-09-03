
#include "cm_neuroncurrents_multichannel_test_model.h"








// i_h channel //////////////////////////////////////////////////////////////////
void nest::i_hMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_h_channel_count++;
        i_tot_i_h.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable hf_h
        hf_h.push_back(0.26894142);
        // state variable hs_h
        hs_h.push_back(0.26894142);

        
        // channel parameter gbar_h
        gbar_h.push_back(0.0);
        // channel parameter e_h
        e_h.push_back((-43.0));

        
    }
}

void nest::i_hMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_h channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_h" ) ){
        if(getValue< double >( channel_params, "gbar_h" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_h_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_h.push_back(0);
        // state variable hf_h
        hf_h.push_back(0.26894142);
        // state variable hs_h
        hs_h.push_back(0.26894142);
        // i_h channel parameter 
        if( channel_params->known( "hf_h" ) )
            hf_h[neuron_i_h_channel_count-1] = getValue< double >( channel_params, "hf_h" );
        // i_h channel parameter 
        if( channel_params->known( "hs_h" ) )
            hs_h[neuron_i_h_channel_count-1] = getValue< double >( channel_params, "hs_h" );
        
        // i_h channel ODE state 
        if( channel_params->known( "hf_h" ) )
            hf_h[neuron_i_h_channel_count-1] = getValue< double >( channel_params, "hf_h" );
        // i_h channel ODE state 
        if( channel_params->known( "hs_h" ) )
            hs_h[neuron_i_h_channel_count-1] = getValue< double >( channel_params, "hs_h" );
        

        
        // channel parameter gbar_h
        gbar_h.push_back(0.0);
        // channel parameter e_h
        e_h.push_back((-43.0));
        // i_h channel parameter 
        if( channel_params->known( "gbar_h" ) )
            gbar_h[neuron_i_h_channel_count-1] = getValue< double >( channel_params, "gbar_h" );
        // i_h channel parameter 
        if( channel_params->known( "e_h" ) )
            e_h[neuron_i_h_channel_count-1] = getValue< double >( channel_params, "e_h" );
        
    }
}

void
nest::i_hMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_h_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("hf_h") + std::to_string(compartment_idx))] = &hf_h[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("hf_h") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_h_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("hs_h") + std::to_string(compartment_idx))] = &hs_h[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("hs_h") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_h_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_h") + std::to_string(compartment_idx))] = &i_tot_i_h[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_h") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_hMultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_h_channel_count, 0.);
    std::vector< double > i_val(neuron_i_h_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_h_channel_count, 0.);

         std::vector< double > __h(neuron_i_h_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__hf_h__hf_h(neuron_i_h_channel_count, 0);
        std::vector< double > __P__hs_h__hs_h(neuron_i_h_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_h_channel_count; i++){
            __P__hf_h__hf_h[i] = std::exp((-__h[i]) / tau_hf_h(v_comp[i]));
            hf_h[i] = __P__hf_h__hf_h[i] * (hf_h[i] - hf_inf_h(v_comp[i])) + hf_inf_h(v_comp[i]);
            __P__hs_h__hs_h[i] = std::exp((-__h[i]) / tau_hs_h(v_comp[i]));
            hs_h[i] = __P__hs_h__hs_h[i] * (hs_h[i] - hs_inf_h(v_comp[i])) + hs_inf_h(v_comp[i]);

            // compute the conductance of the i_h channel
            this->i_tot_i_h[i] = gbar_h[i] * (0.8 * hf_h[i] + 0.2 * hs_h[i]) * (e_h[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_h[i]) * (0.8 * hf_h[i] + 0.2 * hs_h[i]);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_h[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions h
double nest::i_hMultichannelTestModel::hf_inf_h ( double v_comp) const
{  
  double val;
  val = 1.0 / (122306.530090586 * std::exp(0.142857142857143 * v_comp) + 1.0);
  return val;
}


inline 
//
double nest::i_hMultichannelTestModel::tau_hf_h ( double v_comp) const
{  
  double val;
  val = 40.0;
  return val;
}


inline 
//
double nest::i_hMultichannelTestModel::hs_inf_h ( double v_comp) const
{  
  double val;
  val = 1.0 / (122306.530090586 * std::exp(0.142857142857143 * v_comp) + 1.0);
  return val;
}


inline 
//
double nest::i_hMultichannelTestModel::tau_hs_h ( double v_comp) const
{  
  double val;
  val = 300.0;
  return val;
}

void nest::i_hMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_h_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_h[chan_id];
    }
}

std::vector< double > nest::i_hMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_h_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_h_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_h channel end ///////////////////////////////////////////////////////////


// i_TestChannel2 channel //////////////////////////////////////////////////////////////////
void nest::i_TestChannel2MultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_TestChannel2_channel_count++;
        i_tot_i_TestChannel2.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable a00_TestChannel2
        a00_TestChannel2.push_back(0.3);
        // state variable a01_TestChannel2
        a01_TestChannel2.push_back(0.5);
        // state variable a10_TestChannel2
        a10_TestChannel2.push_back(0.4);
        // state variable a11_TestChannel2
        a11_TestChannel2.push_back(0.6);

        
        // channel parameter gbar_TestChannel2
        gbar_TestChannel2.push_back(0.0);
        // channel parameter e_TestChannel2
        e_TestChannel2.push_back((-23.0));

        
    }
}

void nest::i_TestChannel2MultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_TestChannel2 channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_TestChannel2" ) ){
        if(getValue< double >( channel_params, "gbar_TestChannel2" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_TestChannel2_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_TestChannel2.push_back(0);
        // state variable a00_TestChannel2
        a00_TestChannel2.push_back(0.3);
        // state variable a01_TestChannel2
        a01_TestChannel2.push_back(0.5);
        // state variable a10_TestChannel2
        a10_TestChannel2.push_back(0.4);
        // state variable a11_TestChannel2
        a11_TestChannel2.push_back(0.6);
        // i_TestChannel2 channel parameter 
        if( channel_params->known( "a00_TestChannel2" ) )
            a00_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a00_TestChannel2" );
        // i_TestChannel2 channel parameter 
        if( channel_params->known( "a01_TestChannel2" ) )
            a01_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a01_TestChannel2" );
        // i_TestChannel2 channel parameter 
        if( channel_params->known( "a10_TestChannel2" ) )
            a10_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a10_TestChannel2" );
        // i_TestChannel2 channel parameter 
        if( channel_params->known( "a11_TestChannel2" ) )
            a11_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a11_TestChannel2" );
        
        // i_TestChannel2 channel ODE state 
        if( channel_params->known( "a00_TestChannel2" ) )
            a00_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a00_TestChannel2" );
        // i_TestChannel2 channel ODE state 
        if( channel_params->known( "a01_TestChannel2" ) )
            a01_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a01_TestChannel2" );
        // i_TestChannel2 channel ODE state 
        if( channel_params->known( "a10_TestChannel2" ) )
            a10_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a10_TestChannel2" );
        // i_TestChannel2 channel ODE state 
        if( channel_params->known( "a11_TestChannel2" ) )
            a11_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "a11_TestChannel2" );
        

        
        // channel parameter gbar_TestChannel2
        gbar_TestChannel2.push_back(0.0);
        // channel parameter e_TestChannel2
        e_TestChannel2.push_back((-23.0));
        // i_TestChannel2 channel parameter 
        if( channel_params->known( "gbar_TestChannel2" ) )
            gbar_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "gbar_TestChannel2" );
        // i_TestChannel2 channel parameter 
        if( channel_params->known( "e_TestChannel2" ) )
            e_TestChannel2[neuron_i_TestChannel2_channel_count-1] = getValue< double >( channel_params, "e_TestChannel2" );
        
    }
}

void
nest::i_TestChannel2MultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel2_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a00_TestChannel2") + std::to_string(compartment_idx))] = &a00_TestChannel2[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a00_TestChannel2") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel2_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a01_TestChannel2") + std::to_string(compartment_idx))] = &a01_TestChannel2[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a01_TestChannel2") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel2_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a10_TestChannel2") + std::to_string(compartment_idx))] = &a10_TestChannel2[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a10_TestChannel2") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel2_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a11_TestChannel2") + std::to_string(compartment_idx))] = &a11_TestChannel2[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a11_TestChannel2") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel2_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_TestChannel2") + std::to_string(compartment_idx))] = &i_tot_i_TestChannel2[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_TestChannel2") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_TestChannel2MultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_TestChannel2_channel_count, 0.);
    std::vector< double > i_val(neuron_i_TestChannel2_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_TestChannel2_channel_count, 0.);

         std::vector< double > __h(neuron_i_TestChannel2_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__a00_TestChannel2__a00_TestChannel2(neuron_i_TestChannel2_channel_count, 0);
        std::vector< double > __P__a01_TestChannel2__a01_TestChannel2(neuron_i_TestChannel2_channel_count, 0);
        std::vector< double > __P__a10_TestChannel2__a10_TestChannel2(neuron_i_TestChannel2_channel_count, 0);
        std::vector< double > __P__a11_TestChannel2__a11_TestChannel2(neuron_i_TestChannel2_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_TestChannel2_channel_count; i++){
            __P__a00_TestChannel2__a00_TestChannel2[i] = std::exp((-__h[i]) / tau_a00_TestChannel2(v_comp[i]));
            a00_TestChannel2[i] = __P__a00_TestChannel2__a00_TestChannel2[i] * (a00_TestChannel2[i] - a00_inf_TestChannel2(v_comp[i])) + a00_inf_TestChannel2(v_comp[i]);
            __P__a01_TestChannel2__a01_TestChannel2[i] = std::exp((-__h[i]) / tau_a01_TestChannel2(v_comp[i]));
            a01_TestChannel2[i] = __P__a01_TestChannel2__a01_TestChannel2[i] * (a01_TestChannel2[i] - a01_inf_TestChannel2(v_comp[i])) + a01_inf_TestChannel2(v_comp[i]);
            __P__a10_TestChannel2__a10_TestChannel2[i] = std::exp((-__h[i]) / tau_a10_TestChannel2(v_comp[i]));
            a10_TestChannel2[i] = __P__a10_TestChannel2__a10_TestChannel2[i] * (a10_TestChannel2[i] - a10_inf_TestChannel2(v_comp[i])) + a10_inf_TestChannel2(v_comp[i]);
            __P__a11_TestChannel2__a11_TestChannel2[i] = std::exp((-__h[i]) / tau_a11_TestChannel2(v_comp[i]));
            a11_TestChannel2[i] = __P__a11_TestChannel2__a11_TestChannel2[i] * (a11_TestChannel2[i] - a11_inf_TestChannel2(v_comp[i])) + a11_inf_TestChannel2(v_comp[i]);

            // compute the conductance of the i_TestChannel2 channel
            this->i_tot_i_TestChannel2[i] = gbar_TestChannel2[i] * (0.9 * pow(a00_TestChannel2[i], 3) * pow(a01_TestChannel2[i], 2) + 0.1 * pow(a10_TestChannel2[i], 2) * a11_TestChannel2[i]) * (e_TestChannel2[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_TestChannel2[i]) * (0.9 * pow(a00_TestChannel2[i], 3) * pow(a01_TestChannel2[i], 2) + 0.1 * pow(a10_TestChannel2[i], 2) * a11_TestChannel2[i]);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_TestChannel2[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions TestChannel2
double nest::i_TestChannel2MultichannelTestModel::a00_inf_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 0.3;
  return val;
}


inline 
//
double nest::i_TestChannel2MultichannelTestModel::tau_a00_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 1.0;
  return val;
}


inline 
//
double nest::i_TestChannel2MultichannelTestModel::a01_inf_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 0.5;
  return val;
}


inline 
//
double nest::i_TestChannel2MultichannelTestModel::tau_a01_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 2.0;
  return val;
}


inline 
//
double nest::i_TestChannel2MultichannelTestModel::a10_inf_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 0.4;
  return val;
}


inline 
//
double nest::i_TestChannel2MultichannelTestModel::tau_a10_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 2.0;
  return val;
}


inline 
//
double nest::i_TestChannel2MultichannelTestModel::a11_inf_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 0.6;
  return val;
}


inline 
//
double nest::i_TestChannel2MultichannelTestModel::tau_a11_TestChannel2 ( double v_comp) const
{  
  double val;
  val = 2.0;
  return val;
}

void nest::i_TestChannel2MultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_TestChannel2_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_TestChannel2[chan_id];
    }
}

std::vector< double > nest::i_TestChannel2MultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_TestChannel2_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_TestChannel2_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_TestChannel2 channel end ///////////////////////////////////////////////////////////


// i_TestChannel channel //////////////////////////////////////////////////////////////////
void nest::i_TestChannelMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_TestChannel_channel_count++;
        i_tot_i_TestChannel.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable a00_TestChannel
        a00_TestChannel.push_back(0.7407749);
        // state variable a01_TestChannel
        a01_TestChannel.push_back(0.2592251);
        // state variable a02_TestChannel
        a02_TestChannel.push_back((-10.0));
        // state variable a10_TestChannel
        a10_TestChannel.push_back(1.4815498);
        // state variable a11_TestChannel
        a11_TestChannel.push_back(0.5184502);
        // state variable a12_TestChannel
        a12_TestChannel.push_back((-30.0));

        
        // channel parameter gbar_TestChannel
        gbar_TestChannel.push_back(0.0);
        // channel parameter e_TestChannel
        e_TestChannel.push_back((-23.0));

        
    }
}

void nest::i_TestChannelMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_TestChannel channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_TestChannel" ) ){
        if(getValue< double >( channel_params, "gbar_TestChannel" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_TestChannel_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_TestChannel.push_back(0);
        // state variable a00_TestChannel
        a00_TestChannel.push_back(0.7407749);
        // state variable a01_TestChannel
        a01_TestChannel.push_back(0.2592251);
        // state variable a02_TestChannel
        a02_TestChannel.push_back((-10.0));
        // state variable a10_TestChannel
        a10_TestChannel.push_back(1.4815498);
        // state variable a11_TestChannel
        a11_TestChannel.push_back(0.5184502);
        // state variable a12_TestChannel
        a12_TestChannel.push_back((-30.0));
        // i_TestChannel channel parameter 
        if( channel_params->known( "a00_TestChannel" ) )
            a00_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a00_TestChannel" );
        // i_TestChannel channel parameter 
        if( channel_params->known( "a01_TestChannel" ) )
            a01_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a01_TestChannel" );
        // i_TestChannel channel parameter 
        if( channel_params->known( "a02_TestChannel" ) )
            a02_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a02_TestChannel" );
        // i_TestChannel channel parameter 
        if( channel_params->known( "a10_TestChannel" ) )
            a10_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a10_TestChannel" );
        // i_TestChannel channel parameter 
        if( channel_params->known( "a11_TestChannel" ) )
            a11_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a11_TestChannel" );
        // i_TestChannel channel parameter 
        if( channel_params->known( "a12_TestChannel" ) )
            a12_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a12_TestChannel" );
        
        // i_TestChannel channel ODE state 
        if( channel_params->known( "a00_TestChannel" ) )
            a00_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a00_TestChannel" );
        // i_TestChannel channel ODE state 
        if( channel_params->known( "a01_TestChannel" ) )
            a01_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a01_TestChannel" );
        // i_TestChannel channel ODE state 
        if( channel_params->known( "a02_TestChannel" ) )
            a02_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a02_TestChannel" );
        // i_TestChannel channel ODE state 
        if( channel_params->known( "a10_TestChannel" ) )
            a10_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a10_TestChannel" );
        // i_TestChannel channel ODE state 
        if( channel_params->known( "a11_TestChannel" ) )
            a11_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a11_TestChannel" );
        // i_TestChannel channel ODE state 
        if( channel_params->known( "a12_TestChannel" ) )
            a12_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "a12_TestChannel" );
        

        
        // channel parameter gbar_TestChannel
        gbar_TestChannel.push_back(0.0);
        // channel parameter e_TestChannel
        e_TestChannel.push_back((-23.0));
        // i_TestChannel channel parameter 
        if( channel_params->known( "gbar_TestChannel" ) )
            gbar_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "gbar_TestChannel" );
        // i_TestChannel channel parameter 
        if( channel_params->known( "e_TestChannel" ) )
            e_TestChannel[neuron_i_TestChannel_channel_count-1] = getValue< double >( channel_params, "e_TestChannel" );
        
    }
}

void
nest::i_TestChannelMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a00_TestChannel") + std::to_string(compartment_idx))] = &a00_TestChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a00_TestChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a01_TestChannel") + std::to_string(compartment_idx))] = &a01_TestChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a01_TestChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a02_TestChannel") + std::to_string(compartment_idx))] = &a02_TestChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a02_TestChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a10_TestChannel") + std::to_string(compartment_idx))] = &a10_TestChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a10_TestChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a11_TestChannel") + std::to_string(compartment_idx))] = &a11_TestChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a11_TestChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a12_TestChannel") + std::to_string(compartment_idx))] = &a12_TestChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a12_TestChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_TestChannel") + std::to_string(compartment_idx))] = &i_tot_i_TestChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_TestChannel") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_TestChannelMultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_TestChannel_channel_count, 0.);
    std::vector< double > i_val(neuron_i_TestChannel_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_TestChannel_channel_count, 0.);

         std::vector< double > __h(neuron_i_TestChannel_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__a00_TestChannel__a00_TestChannel(neuron_i_TestChannel_channel_count, 0);
        std::vector< double > __P__a01_TestChannel__a01_TestChannel(neuron_i_TestChannel_channel_count, 0);
        std::vector< double > __P__a02_TestChannel__a02_TestChannel(neuron_i_TestChannel_channel_count, 0);
        std::vector< double > __P__a10_TestChannel__a10_TestChannel(neuron_i_TestChannel_channel_count, 0);
        std::vector< double > __P__a11_TestChannel__a11_TestChannel(neuron_i_TestChannel_channel_count, 0);
        std::vector< double > __P__a12_TestChannel__a12_TestChannel(neuron_i_TestChannel_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_TestChannel_channel_count; i++){
            __P__a00_TestChannel__a00_TestChannel[i] = std::exp((-__h[i]) / tau_a00_TestChannel(v_comp[i]));
            a00_TestChannel[i] = __P__a00_TestChannel__a00_TestChannel[i] * (a00_TestChannel[i] - a00_inf_TestChannel(v_comp[i])) + a00_inf_TestChannel(v_comp[i]);
            __P__a01_TestChannel__a01_TestChannel[i] = std::exp((-__h[i]) / tau_a01_TestChannel(v_comp[i]));
            a01_TestChannel[i] = __P__a01_TestChannel__a01_TestChannel[i] * (a01_TestChannel[i] - a01_inf_TestChannel(v_comp[i])) + a01_inf_TestChannel(v_comp[i]);
            __P__a02_TestChannel__a02_TestChannel[i] = std::exp((-__h[i]) / tau_a02_TestChannel(v_comp[i]));
            a02_TestChannel[i] = __P__a02_TestChannel__a02_TestChannel[i] * (a02_TestChannel[i] - a02_inf_TestChannel(v_comp[i])) + a02_inf_TestChannel(v_comp[i]);
            __P__a10_TestChannel__a10_TestChannel[i] = std::exp((-__h[i]) / tau_a10_TestChannel(v_comp[i]));
            a10_TestChannel[i] = __P__a10_TestChannel__a10_TestChannel[i] * (a10_TestChannel[i] - a10_inf_TestChannel(v_comp[i])) + a10_inf_TestChannel(v_comp[i]);
            __P__a11_TestChannel__a11_TestChannel[i] = std::exp((-__h[i]) / tau_a11_TestChannel(v_comp[i]));
            a11_TestChannel[i] = __P__a11_TestChannel__a11_TestChannel[i] * (a11_TestChannel[i] - a11_inf_TestChannel(v_comp[i])) + a11_inf_TestChannel(v_comp[i]);
            __P__a12_TestChannel__a12_TestChannel[i] = std::exp((-__h[i]) / tau_a12_TestChannel(v_comp[i]));
            a12_TestChannel[i] = __P__a12_TestChannel__a12_TestChannel[i] * (a12_TestChannel[i] - a12_inf_TestChannel(v_comp[i])) + a12_inf_TestChannel(v_comp[i]);

            // compute the conductance of the i_TestChannel channel
            this->i_tot_i_TestChannel[i] = gbar_TestChannel[i] * (5 * pow(a00_TestChannel[i], 3) * pow(a01_TestChannel[i], 3) * a02_TestChannel[i] + pow(a10_TestChannel[i], 2) * pow(a11_TestChannel[i], 2) * a12_TestChannel[i]) * (e_TestChannel[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_TestChannel[i]) * (5 * pow(a00_TestChannel[i], 3) * pow(a01_TestChannel[i], 3) * a02_TestChannel[i] + pow(a10_TestChannel[i], 2) * pow(a11_TestChannel[i], 2) * a12_TestChannel[i]);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_TestChannel[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions TestChannel
double nest::i_TestChannelMultichannelTestModel::a00_inf_TestChannel ( double v_comp) const
{  
  double val;
  val = 1.0 / (0.740818220681718 * std::exp(0.01 * v_comp) + 1.0);
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::tau_a00_TestChannel ( double v_comp) const
{  
  double val;
  val = 1.0;
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::a01_inf_TestChannel ( double v_comp) const
{  
  double val;
  val = 1.0 * std::exp(0.01 * v_comp) / (1.0 * std::exp(0.01 * v_comp) + 1.349858807576);
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::tau_a01_TestChannel ( double v_comp) const
{  
  double val;
  val = 2.0;
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::a02_inf_TestChannel ( double v_comp) const
{  
  double val;
  val = (-10.0);
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::tau_a02_TestChannel ( double v_comp) const
{  
  double val;
  val = 1.0;
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::a10_inf_TestChannel ( double v_comp) const
{  
  double val;
  val = 2.0 / (0.740818220681718 * std::exp(0.01 * v_comp) + 1.0);
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::tau_a10_TestChannel ( double v_comp) const
{  
  double val;
  val = 2.0;
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::a11_inf_TestChannel ( double v_comp) const
{  
  double val;
  val = 2.0 * std::exp(0.01 * v_comp) / (1.0 * std::exp(0.01 * v_comp) + 1.349858807576);
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::tau_a11_TestChannel ( double v_comp) const
{  
  double val;
  val = 2.0;
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::a12_inf_TestChannel ( double v_comp) const
{  
  double val;
  val = (-30.0);
  return val;
}


inline 
//
double nest::i_TestChannelMultichannelTestModel::tau_a12_TestChannel ( double v_comp) const
{  
  double val;
  val = 3.0;
  return val;
}

void nest::i_TestChannelMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_TestChannel_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_TestChannel[chan_id];
    }
}

std::vector< double > nest::i_TestChannelMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_TestChannel_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_TestChannel_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_TestChannel channel end ///////////////////////////////////////////////////////////


// i_SKv3_1 channel //////////////////////////////////////////////////////////////////
void nest::i_SKv3_1MultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_SKv3_1_channel_count++;
        i_tot_i_SKv3_1.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable z_SKv3_1
        z_SKv3_1.push_back(6.379e-05);

        
        // channel parameter gbar_SKv3_1
        gbar_SKv3_1.push_back(0.0);
        // channel parameter e_SKv3_1
        e_SKv3_1.push_back((-85.0));

        
    }
}

void nest::i_SKv3_1MultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_SKv3_1 channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_SKv3_1" ) ){
        if(getValue< double >( channel_params, "gbar_SKv3_1" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_SKv3_1_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_SKv3_1.push_back(0);
        // state variable z_SKv3_1
        z_SKv3_1.push_back(6.379e-05);
        // i_SKv3_1 channel parameter 
        if( channel_params->known( "z_SKv3_1" ) )
            z_SKv3_1[neuron_i_SKv3_1_channel_count-1] = getValue< double >( channel_params, "z_SKv3_1" );
        
        // i_SKv3_1 channel ODE state 
        if( channel_params->known( "z_SKv3_1" ) )
            z_SKv3_1[neuron_i_SKv3_1_channel_count-1] = getValue< double >( channel_params, "z_SKv3_1" );
        

        
        // channel parameter gbar_SKv3_1
        gbar_SKv3_1.push_back(0.0);
        // channel parameter e_SKv3_1
        e_SKv3_1.push_back((-85.0));
        // i_SKv3_1 channel parameter 
        if( channel_params->known( "gbar_SKv3_1" ) )
            gbar_SKv3_1[neuron_i_SKv3_1_channel_count-1] = getValue< double >( channel_params, "gbar_SKv3_1" );
        // i_SKv3_1 channel parameter 
        if( channel_params->known( "e_SKv3_1" ) )
            e_SKv3_1[neuron_i_SKv3_1_channel_count-1] = getValue< double >( channel_params, "e_SKv3_1" );
        
    }
}

void
nest::i_SKv3_1MultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_SKv3_1_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("z_SKv3_1") + std::to_string(compartment_idx))] = &z_SKv3_1[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("z_SKv3_1") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_SKv3_1_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_SKv3_1") + std::to_string(compartment_idx))] = &i_tot_i_SKv3_1[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_SKv3_1") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_SKv3_1MultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_SKv3_1_channel_count, 0.);
    std::vector< double > i_val(neuron_i_SKv3_1_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_SKv3_1_channel_count, 0.);

         std::vector< double > __h(neuron_i_SKv3_1_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__z_SKv3_1__z_SKv3_1(neuron_i_SKv3_1_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_SKv3_1_channel_count; i++){
            __P__z_SKv3_1__z_SKv3_1[i] = std::exp((-__h[i]) / tau_z_SKv3_1(v_comp[i]));
            z_SKv3_1[i] = __P__z_SKv3_1__z_SKv3_1[i] * (z_SKv3_1[i] - z_inf_SKv3_1(v_comp[i])) + z_inf_SKv3_1(v_comp[i]);

            // compute the conductance of the i_SKv3_1 channel
            this->i_tot_i_SKv3_1[i] = gbar_SKv3_1[i] * (z_SKv3_1[i]) * (e_SKv3_1[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_SKv3_1[i]) * z_SKv3_1[i];
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_SKv3_1[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions SKv3_1
double nest::i_SKv3_1MultichannelTestModel::z_inf_SKv3_1 ( double v_comp) const
{  
  double val;
  val = std::exp(0.103092783505155 * v_comp) / (std::exp(0.103092783505155 * v_comp) + 6.874610940966);
  return val;
}


inline 
//
double nest::i_SKv3_1MultichannelTestModel::tau_z_SKv3_1 ( double v_comp) const
{  
  double val;
  val = 4.0 * std::exp(0.0226551880380607 * v_comp) / (std::exp(0.0226551880380607 * v_comp) + 0.348253173014273);
  return val;
}

void nest::i_SKv3_1MultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_SKv3_1_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_SKv3_1[chan_id];
    }
}

std::vector< double > nest::i_SKv3_1MultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_SKv3_1_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_SKv3_1_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_SKv3_1 channel end ///////////////////////////////////////////////////////////


// i_SK_E2 channel //////////////////////////////////////////////////////////////////
void nest::i_SK_E2MultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_SK_E2_channel_count++;
        i_tot_i_SK_E2.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable z_SK_E2
        z_SK_E2.push_back(0.00090982);

        
        // channel parameter gbar_SK_E2
        gbar_SK_E2.push_back(0.0);
        // channel parameter e_SK_E2
        e_SK_E2.push_back((-85.0));

        
    }
}

void nest::i_SK_E2MultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_SK_E2 channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_SK_E2" ) ){
        if(getValue< double >( channel_params, "gbar_SK_E2" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_SK_E2_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_SK_E2.push_back(0);
        // state variable z_SK_E2
        z_SK_E2.push_back(0.00090982);
        // i_SK_E2 channel parameter 
        if( channel_params->known( "z_SK_E2" ) )
            z_SK_E2[neuron_i_SK_E2_channel_count-1] = getValue< double >( channel_params, "z_SK_E2" );
        
        // i_SK_E2 channel ODE state 
        if( channel_params->known( "z_SK_E2" ) )
            z_SK_E2[neuron_i_SK_E2_channel_count-1] = getValue< double >( channel_params, "z_SK_E2" );
        

        
        // channel parameter gbar_SK_E2
        gbar_SK_E2.push_back(0.0);
        // channel parameter e_SK_E2
        e_SK_E2.push_back((-85.0));
        // i_SK_E2 channel parameter 
        if( channel_params->known( "gbar_SK_E2" ) )
            gbar_SK_E2[neuron_i_SK_E2_channel_count-1] = getValue< double >( channel_params, "gbar_SK_E2" );
        // i_SK_E2 channel parameter 
        if( channel_params->known( "e_SK_E2" ) )
            e_SK_E2[neuron_i_SK_E2_channel_count-1] = getValue< double >( channel_params, "e_SK_E2" );
        
    }
}

void
nest::i_SK_E2MultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_SK_E2_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("z_SK_E2") + std::to_string(compartment_idx))] = &z_SK_E2[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("z_SK_E2") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_SK_E2_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_SK_E2") + std::to_string(compartment_idx))] = &i_tot_i_SK_E2[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_SK_E2") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_SK_E2MultichannelTestModel::f_numstep(std::vector< double > v_comp, std::vector< double > c_ca)
{
    std::vector< double > g_val(neuron_i_SK_E2_channel_count, 0.);
    std::vector< double > i_val(neuron_i_SK_E2_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_SK_E2_channel_count, 0.);

         std::vector< double > __h(neuron_i_SK_E2_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__z_SK_E2__z_SK_E2(neuron_i_SK_E2_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_SK_E2_channel_count; i++){
            __P__z_SK_E2__z_SK_E2[i] = std::exp((-__h[i]) / tau_z_SK_E2(v_comp[i], c_ca[i]));
            z_SK_E2[i] = __P__z_SK_E2__z_SK_E2[i] * (z_SK_E2[i] - z_inf_SK_E2(v_comp[i], c_ca[i])) + z_inf_SK_E2(v_comp[i], c_ca[i]);

            // compute the conductance of the i_SK_E2 channel
            this->i_tot_i_SK_E2[i] = gbar_SK_E2[i] * (z_SK_E2[i]) * (e_SK_E2[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_SK_E2[i]) * z_SK_E2[i];
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_SK_E2[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions SK_E2
double nest::i_SK_E2MultichannelTestModel::z_inf_SK_E2 ( double v_comp, double ca) const
{  
  double val;
  if (ca > 1e-07)
  {  
    val = 1.0 / (6.92864941342586e-17 * pow(1.0 / ca, 4.8) + 1.0);
  }
  else
  {  
    val = 1.0 / (6.92864941342586e-17 * pow(1.0 / (ca + 1e-07), 4.8) + 1.0);
  }
  return val;
}


inline 
//
double nest::i_SK_E2MultichannelTestModel::tau_z_SK_E2 ( double v_comp, double ca) const
{  
  double val;
  val = 1.0;
  return val;
}

void nest::i_SK_E2MultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_SK_E2_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_SK_E2[chan_id];
    }
}

std::vector< double > nest::i_SK_E2MultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_SK_E2_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_SK_E2_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_SK_E2 channel end ///////////////////////////////////////////////////////////


// i_SK channel //////////////////////////////////////////////////////////////////
void nest::i_SKMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_SK_channel_count++;
        i_tot_i_SK.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable z_SK
        z_SK.push_back(0.00090982);

        
        // channel parameter gbar_SK
        gbar_SK.push_back(0.0);
        // channel parameter e_SK
        e_SK.push_back((-85.0));

        
    }
}

void nest::i_SKMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_SK channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_SK" ) ){
        if(getValue< double >( channel_params, "gbar_SK" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_SK_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_SK.push_back(0);
        // state variable z_SK
        z_SK.push_back(0.00090982);
        // i_SK channel parameter 
        if( channel_params->known( "z_SK" ) )
            z_SK[neuron_i_SK_channel_count-1] = getValue< double >( channel_params, "z_SK" );
        
        // i_SK channel ODE state 
        if( channel_params->known( "z_SK" ) )
            z_SK[neuron_i_SK_channel_count-1] = getValue< double >( channel_params, "z_SK" );
        

        
        // channel parameter gbar_SK
        gbar_SK.push_back(0.0);
        // channel parameter e_SK
        e_SK.push_back((-85.0));
        // i_SK channel parameter 
        if( channel_params->known( "gbar_SK" ) )
            gbar_SK[neuron_i_SK_channel_count-1] = getValue< double >( channel_params, "gbar_SK" );
        // i_SK channel parameter 
        if( channel_params->known( "e_SK" ) )
            e_SK[neuron_i_SK_channel_count-1] = getValue< double >( channel_params, "e_SK" );
        
    }
}

void
nest::i_SKMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_SK_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("z_SK") + std::to_string(compartment_idx))] = &z_SK[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("z_SK") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_SK_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_SK") + std::to_string(compartment_idx))] = &i_tot_i_SK[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_SK") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_SKMultichannelTestModel::f_numstep(std::vector< double > v_comp, std::vector< double > c_ca)
{
    std::vector< double > g_val(neuron_i_SK_channel_count, 0.);
    std::vector< double > i_val(neuron_i_SK_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_SK_channel_count, 0.);

         std::vector< double > __h(neuron_i_SK_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__z_SK__z_SK(neuron_i_SK_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_SK_channel_count; i++){
            __P__z_SK__z_SK[i] = std::exp((-__h[i]) / tau_z_SK(v_comp[i], c_ca[i]));
            z_SK[i] = __P__z_SK__z_SK[i] * (z_SK[i] - z_inf_SK(v_comp[i], c_ca[i])) + z_inf_SK(v_comp[i], c_ca[i]);

            // compute the conductance of the i_SK channel
            this->i_tot_i_SK[i] = gbar_SK[i] * (z_SK[i]) * (e_SK[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_SK[i]) * z_SK[i];
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_SK[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions SK
double nest::i_SKMultichannelTestModel::z_inf_SK ( double v_comp, double ca) const
{  
  double val;
  val = 1.0 / (6.92864941342586e-17 * pow(1.0 / ca, 4.8) + 1.0);
  return val;
}


inline 
//
double nest::i_SKMultichannelTestModel::tau_z_SK ( double v_comp, double ca) const
{  
  double val;
  val = 1.0;
  return val;
}

void nest::i_SKMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_SK_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_SK[chan_id];
    }
}

std::vector< double > nest::i_SKMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_SK_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_SK_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_SK channel end ///////////////////////////////////////////////////////////


// i_PiecewiseChannel channel //////////////////////////////////////////////////////////////////
void nest::i_PiecewiseChannelMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_PiecewiseChannel_channel_count++;
        i_tot_i_PiecewiseChannel.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable a_PiecewiseChannel
        a_PiecewiseChannel.push_back(0.1);
        // state variable b_PiecewiseChannel
        b_PiecewiseChannel.push_back(0.8);

        
        // channel parameter gbar_PiecewiseChannel
        gbar_PiecewiseChannel.push_back(0.0);
        // channel parameter e_PiecewiseChannel
        e_PiecewiseChannel.push_back((-28.0));

        
    }
}

void nest::i_PiecewiseChannelMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_PiecewiseChannel channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_PiecewiseChannel" ) ){
        if(getValue< double >( channel_params, "gbar_PiecewiseChannel" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_PiecewiseChannel_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_PiecewiseChannel.push_back(0);
        // state variable a_PiecewiseChannel
        a_PiecewiseChannel.push_back(0.1);
        // state variable b_PiecewiseChannel
        b_PiecewiseChannel.push_back(0.8);
        // i_PiecewiseChannel channel parameter 
        if( channel_params->known( "a_PiecewiseChannel" ) )
            a_PiecewiseChannel[neuron_i_PiecewiseChannel_channel_count-1] = getValue< double >( channel_params, "a_PiecewiseChannel" );
        // i_PiecewiseChannel channel parameter 
        if( channel_params->known( "b_PiecewiseChannel" ) )
            b_PiecewiseChannel[neuron_i_PiecewiseChannel_channel_count-1] = getValue< double >( channel_params, "b_PiecewiseChannel" );
        
        // i_PiecewiseChannel channel ODE state 
        if( channel_params->known( "a_PiecewiseChannel" ) )
            a_PiecewiseChannel[neuron_i_PiecewiseChannel_channel_count-1] = getValue< double >( channel_params, "a_PiecewiseChannel" );
        // i_PiecewiseChannel channel ODE state 
        if( channel_params->known( "b_PiecewiseChannel" ) )
            b_PiecewiseChannel[neuron_i_PiecewiseChannel_channel_count-1] = getValue< double >( channel_params, "b_PiecewiseChannel" );
        

        
        // channel parameter gbar_PiecewiseChannel
        gbar_PiecewiseChannel.push_back(0.0);
        // channel parameter e_PiecewiseChannel
        e_PiecewiseChannel.push_back((-28.0));
        // i_PiecewiseChannel channel parameter 
        if( channel_params->known( "gbar_PiecewiseChannel" ) )
            gbar_PiecewiseChannel[neuron_i_PiecewiseChannel_channel_count-1] = getValue< double >( channel_params, "gbar_PiecewiseChannel" );
        // i_PiecewiseChannel channel parameter 
        if( channel_params->known( "e_PiecewiseChannel" ) )
            e_PiecewiseChannel[neuron_i_PiecewiseChannel_channel_count-1] = getValue< double >( channel_params, "e_PiecewiseChannel" );
        
    }
}

void
nest::i_PiecewiseChannelMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_PiecewiseChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("a_PiecewiseChannel") + std::to_string(compartment_idx))] = &a_PiecewiseChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("a_PiecewiseChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_PiecewiseChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("b_PiecewiseChannel") + std::to_string(compartment_idx))] = &b_PiecewiseChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("b_PiecewiseChannel") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_PiecewiseChannel_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_PiecewiseChannel") + std::to_string(compartment_idx))] = &i_tot_i_PiecewiseChannel[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_PiecewiseChannel") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_PiecewiseChannelMultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_PiecewiseChannel_channel_count, 0.);
    std::vector< double > i_val(neuron_i_PiecewiseChannel_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_PiecewiseChannel_channel_count, 0.);

         std::vector< double > __h(neuron_i_PiecewiseChannel_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__a_PiecewiseChannel__a_PiecewiseChannel(neuron_i_PiecewiseChannel_channel_count, 0);
        std::vector< double > __P__b_PiecewiseChannel__b_PiecewiseChannel(neuron_i_PiecewiseChannel_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_PiecewiseChannel_channel_count; i++){
            __P__a_PiecewiseChannel__a_PiecewiseChannel[i] = std::exp((-__h[i]) / tau_a_PiecewiseChannel(v_comp[i]));
            a_PiecewiseChannel[i] = __P__a_PiecewiseChannel__a_PiecewiseChannel[i] * (a_PiecewiseChannel[i] - a_inf_PiecewiseChannel(v_comp[i])) + a_inf_PiecewiseChannel(v_comp[i]);
            __P__b_PiecewiseChannel__b_PiecewiseChannel[i] = std::exp((-__h[i]) / tau_b_PiecewiseChannel(v_comp[i]));
            b_PiecewiseChannel[i] = __P__b_PiecewiseChannel__b_PiecewiseChannel[i] * (b_PiecewiseChannel[i] - b_inf_PiecewiseChannel(v_comp[i])) + b_inf_PiecewiseChannel(v_comp[i]);

            // compute the conductance of the i_PiecewiseChannel channel
            this->i_tot_i_PiecewiseChannel[i] = gbar_PiecewiseChannel[i] * (a_PiecewiseChannel[i] + b_PiecewiseChannel[i]) * (e_PiecewiseChannel[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_PiecewiseChannel[i]) * (a_PiecewiseChannel[i] + b_PiecewiseChannel[i]);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_PiecewiseChannel[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions PiecewiseChannel
double nest::i_PiecewiseChannelMultichannelTestModel::a_inf_PiecewiseChannel ( double v_comp) const
{  
  double val;
  if (v_comp < (-50.0))
  {  
    val = 0.1;
  }
  else
  {  
    val = 0.9;
  }
  return val;
}


inline 
//
double nest::i_PiecewiseChannelMultichannelTestModel::tau_a_PiecewiseChannel ( double v_comp) const
{  
  double val;
  if (v_comp < (-50.0))
  {  
    val = 10.0;
  }
  else
  {  
    val = 20.0;
  }
  return val;
}


inline 
//
double nest::i_PiecewiseChannelMultichannelTestModel::b_inf_PiecewiseChannel ( double v_comp) const
{  
  double val;
  if (v_comp < (-50.0))
  {  
    val = 0.8;
  }
  else
  {  
    val = 0.2;
  }
  return val;
}


inline 
//
double nest::i_PiecewiseChannelMultichannelTestModel::tau_b_PiecewiseChannel ( double v_comp) const
{  
  double val;
  if (v_comp < (-50.0))
  {  
    val = 0.1;
  }
  else
  {  
    val = 50.0;
  }
  return val;
}

void nest::i_PiecewiseChannelMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_PiecewiseChannel_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_PiecewiseChannel[chan_id];
    }
}

std::vector< double > nest::i_PiecewiseChannelMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_PiecewiseChannel_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_PiecewiseChannel_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_PiecewiseChannel channel end ///////////////////////////////////////////////////////////


// i_Na_Ta channel //////////////////////////////////////////////////////////////////
void nest::i_Na_TaMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_Na_Ta_channel_count++;
        i_tot_i_Na_Ta.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable h_Na_Ta
        h_Na_Ta.push_back(0.81757448);
        // state variable m_Na_Ta
        m_Na_Ta.push_back(0.00307019);

        
        // channel parameter gbar_Na_Ta
        gbar_Na_Ta.push_back(0.0);
        // channel parameter e_Na_Ta
        e_Na_Ta.push_back(50.0);

        
    }
}

void nest::i_Na_TaMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_Na_Ta channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_Na_Ta" ) ){
        if(getValue< double >( channel_params, "gbar_Na_Ta" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_Na_Ta_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_Na_Ta.push_back(0);
        // state variable h_Na_Ta
        h_Na_Ta.push_back(0.81757448);
        // state variable m_Na_Ta
        m_Na_Ta.push_back(0.00307019);
        // i_Na_Ta channel parameter 
        if( channel_params->known( "h_Na_Ta" ) )
            h_Na_Ta[neuron_i_Na_Ta_channel_count-1] = getValue< double >( channel_params, "h_Na_Ta" );
        // i_Na_Ta channel parameter 
        if( channel_params->known( "m_Na_Ta" ) )
            m_Na_Ta[neuron_i_Na_Ta_channel_count-1] = getValue< double >( channel_params, "m_Na_Ta" );
        
        // i_Na_Ta channel ODE state 
        if( channel_params->known( "h_Na_Ta" ) )
            h_Na_Ta[neuron_i_Na_Ta_channel_count-1] = getValue< double >( channel_params, "h_Na_Ta" );
        // i_Na_Ta channel ODE state 
        if( channel_params->known( "m_Na_Ta" ) )
            m_Na_Ta[neuron_i_Na_Ta_channel_count-1] = getValue< double >( channel_params, "m_Na_Ta" );
        

        
        // channel parameter gbar_Na_Ta
        gbar_Na_Ta.push_back(0.0);
        // channel parameter e_Na_Ta
        e_Na_Ta.push_back(50.0);
        // i_Na_Ta channel parameter 
        if( channel_params->known( "gbar_Na_Ta" ) )
            gbar_Na_Ta[neuron_i_Na_Ta_channel_count-1] = getValue< double >( channel_params, "gbar_Na_Ta" );
        // i_Na_Ta channel parameter 
        if( channel_params->known( "e_Na_Ta" ) )
            e_Na_Ta[neuron_i_Na_Ta_channel_count-1] = getValue< double >( channel_params, "e_Na_Ta" );
        
    }
}

void
nest::i_Na_TaMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Na_Ta_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("h_Na_Ta") + std::to_string(compartment_idx))] = &h_Na_Ta[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("h_Na_Ta") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Na_Ta_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("m_Na_Ta") + std::to_string(compartment_idx))] = &m_Na_Ta[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("m_Na_Ta") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Na_Ta_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_Na_Ta") + std::to_string(compartment_idx))] = &i_tot_i_Na_Ta[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_Na_Ta") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_Na_TaMultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_Na_Ta_channel_count, 0.);
    std::vector< double > i_val(neuron_i_Na_Ta_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_Na_Ta_channel_count, 0.);

         std::vector< double > __h(neuron_i_Na_Ta_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__h_Na_Ta__h_Na_Ta(neuron_i_Na_Ta_channel_count, 0);
        std::vector< double > __P__m_Na_Ta__m_Na_Ta(neuron_i_Na_Ta_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_Na_Ta_channel_count; i++){
            __P__h_Na_Ta__h_Na_Ta[i] = std::exp((-__h[i]) / tau_h_Na_Ta(v_comp[i]));
            h_Na_Ta[i] = __P__h_Na_Ta__h_Na_Ta[i] * (h_Na_Ta[i] - h_inf_Na_Ta(v_comp[i])) + h_inf_Na_Ta(v_comp[i]);
            __P__m_Na_Ta__m_Na_Ta[i] = std::exp((-__h[i]) / tau_m_Na_Ta(v_comp[i]));
            m_Na_Ta[i] = __P__m_Na_Ta__m_Na_Ta[i] * (m_Na_Ta[i] - m_inf_Na_Ta(v_comp[i])) + m_inf_Na_Ta(v_comp[i]);

            // compute the conductance of the i_Na_Ta channel
            this->i_tot_i_Na_Ta[i] = gbar_Na_Ta[i] * (h_Na_Ta[i] * pow(m_Na_Ta[i], 3)) * (e_Na_Ta[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_Na_Ta[i]) * h_Na_Ta[i] * pow(m_Na_Ta[i], 3);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_Na_Ta[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions Na_Ta
double nest::i_Na_TaMultichannelTestModel::h_inf_Na_Ta ( double v_comp) const
{  
  double val;
  val = (0.015 * v_comp * std::exp(0.166666666666667 * v_comp) - 2.50525511853685e-07 * v_comp + 0.99 * std::exp(0.166666666666667 * v_comp) - 1.65346837823432e-05) / (898.112125727967 * v_comp * std::exp(0.333333333333333 * v_comp) - 2.50525511853685e-07 * v_comp + 59275.4002980458 * std::exp(0.333333333333333 * v_comp) - 1.65346837823432e-05);
  return val;
}


inline 
//
double nest::i_Na_TaMultichannelTestModel::tau_h_Na_Ta ( double v_comp) const
{  
  double val;
  val = ((-0.677966101694915) * std::exp(0.166666666666667 * v_comp) + 20296.3192254908 * std::exp(0.333333333333333 * v_comp) + 5.66159348821887e-06) / (898.112125727967 * v_comp * std::exp(0.333333333333333 * v_comp) - 2.50525511853685e-07 * v_comp + 59275.4002980458 * std::exp(0.333333333333333 * v_comp) - 1.65346837823432e-05);
  return val;
}


inline 
//
double nest::i_Na_TaMultichannelTestModel::m_inf_Na_Ta ( double v_comp) const
{  
  double val;
  val = 0.182 * (v_comp + 38.0) * (563.030236835951 * std::exp(0.166666666666667 * v_comp) - 1.0) * std::exp(0.166666666666667 * v_comp) / ((0.00177610354573438 - 1.0 * std::exp(0.166666666666667 * v_comp)) * ((-0.124) * v_comp - 4.712) + (0.182 * v_comp + 6.9159999999999995) * (563.030236835951 * std::exp(0.166666666666667 * v_comp) - 1.0) * std::exp(0.166666666666667 * v_comp));
  return val;
}


inline 
//
double nest::i_Na_TaMultichannelTestModel::tau_m_Na_Ta ( double v_comp) const
{  
  double val;
  val = (0.677966101694915 * std::exp(0.166666666666667 * v_comp) - 190.857707402017 * std::exp(0.333333333333333 * v_comp) - 0.000602068998554027) / (0.058 * v_comp * std::exp(0.166666666666667 * v_comp) - 102.471503104143 * v_comp * std::exp(0.333333333333333 * v_comp) + 0.000220236839671063 * v_comp + 2.204 * std::exp(0.166666666666667 * v_comp) - 3893.91711795744 * std::exp(0.333333333333333 * v_comp) + 0.00836899990750039);
  return val;
}

void nest::i_Na_TaMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_Na_Ta_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_Na_Ta[chan_id];
    }
}

std::vector< double > nest::i_Na_TaMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_Na_Ta_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_Na_Ta_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_Na_Ta channel end ///////////////////////////////////////////////////////////


// i_NaTa_t channel //////////////////////////////////////////////////////////////////
void nest::i_NaTa_tMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_NaTa_t_channel_count++;
        i_tot_i_NaTa_t.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable h_NaTa_t
        h_NaTa_t.push_back(0.81757448);
        // state variable m_NaTa_t
        m_NaTa_t.push_back(0.00307019);

        
        // channel parameter gbar_NaTa_t
        gbar_NaTa_t.push_back(0.0);
        // channel parameter e_NaTa_t
        e_NaTa_t.push_back(50.0);

        
    }
}

void nest::i_NaTa_tMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_NaTa_t channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_NaTa_t" ) ){
        if(getValue< double >( channel_params, "gbar_NaTa_t" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_NaTa_t_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_NaTa_t.push_back(0);
        // state variable h_NaTa_t
        h_NaTa_t.push_back(0.81757448);
        // state variable m_NaTa_t
        m_NaTa_t.push_back(0.00307019);
        // i_NaTa_t channel parameter 
        if( channel_params->known( "h_NaTa_t" ) )
            h_NaTa_t[neuron_i_NaTa_t_channel_count-1] = getValue< double >( channel_params, "h_NaTa_t" );
        // i_NaTa_t channel parameter 
        if( channel_params->known( "m_NaTa_t" ) )
            m_NaTa_t[neuron_i_NaTa_t_channel_count-1] = getValue< double >( channel_params, "m_NaTa_t" );
        
        // i_NaTa_t channel ODE state 
        if( channel_params->known( "h_NaTa_t" ) )
            h_NaTa_t[neuron_i_NaTa_t_channel_count-1] = getValue< double >( channel_params, "h_NaTa_t" );
        // i_NaTa_t channel ODE state 
        if( channel_params->known( "m_NaTa_t" ) )
            m_NaTa_t[neuron_i_NaTa_t_channel_count-1] = getValue< double >( channel_params, "m_NaTa_t" );
        

        
        // channel parameter gbar_NaTa_t
        gbar_NaTa_t.push_back(0.0);
        // channel parameter e_NaTa_t
        e_NaTa_t.push_back(50.0);
        // i_NaTa_t channel parameter 
        if( channel_params->known( "gbar_NaTa_t" ) )
            gbar_NaTa_t[neuron_i_NaTa_t_channel_count-1] = getValue< double >( channel_params, "gbar_NaTa_t" );
        // i_NaTa_t channel parameter 
        if( channel_params->known( "e_NaTa_t" ) )
            e_NaTa_t[neuron_i_NaTa_t_channel_count-1] = getValue< double >( channel_params, "e_NaTa_t" );
        
    }
}

void
nest::i_NaTa_tMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_NaTa_t_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("h_NaTa_t") + std::to_string(compartment_idx))] = &h_NaTa_t[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("h_NaTa_t") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_NaTa_t_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("m_NaTa_t") + std::to_string(compartment_idx))] = &m_NaTa_t[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("m_NaTa_t") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_NaTa_t_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_NaTa_t") + std::to_string(compartment_idx))] = &i_tot_i_NaTa_t[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_NaTa_t") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_NaTa_tMultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_NaTa_t_channel_count, 0.);
    std::vector< double > i_val(neuron_i_NaTa_t_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_NaTa_t_channel_count, 0.);

         std::vector< double > __h(neuron_i_NaTa_t_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__h_NaTa_t__h_NaTa_t(neuron_i_NaTa_t_channel_count, 0);
        std::vector< double > __P__m_NaTa_t__m_NaTa_t(neuron_i_NaTa_t_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_NaTa_t_channel_count; i++){
            __P__h_NaTa_t__h_NaTa_t[i] = std::exp((-__h[i]) / tau_h_NaTa_t(v_comp[i]));
            h_NaTa_t[i] = __P__h_NaTa_t__h_NaTa_t[i] * (h_NaTa_t[i] - h_inf_NaTa_t(v_comp[i])) + h_inf_NaTa_t(v_comp[i]);
            __P__m_NaTa_t__m_NaTa_t[i] = std::exp((-__h[i]) / tau_m_NaTa_t(v_comp[i]));
            m_NaTa_t[i] = __P__m_NaTa_t__m_NaTa_t[i] * (m_NaTa_t[i] - m_inf_NaTa_t(v_comp[i])) + m_inf_NaTa_t(v_comp[i]);

            // compute the conductance of the i_NaTa_t channel
            this->i_tot_i_NaTa_t[i] = gbar_NaTa_t[i] * (h_NaTa_t[i] * pow(m_NaTa_t[i], 3)) * (e_NaTa_t[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_NaTa_t[i]) * h_NaTa_t[i] * pow(m_NaTa_t[i], 3);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_NaTa_t[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions NaTa_t
double nest::i_NaTa_tMultichannelTestModel::h_inf_NaTa_t ( double v_comp) const
{  
  double val;
  if (v_comp >= (-66.000006) && v_comp < (-65.999994))
  {  
    val = (-0.0416666666666667) * v_comp - 2.25;
  }
  else
  {  
    val = (1.0 * std::exp(0.166666666666667 * v_comp) - 1.67017007902457e-05) / (59874.1417151978 * std::exp(0.333333333333333 * v_comp) - 1.67017007902457e-05);
  }
  return val;
}


inline 
//
double nest::i_NaTa_tMultichannelTestModel::tau_h_NaTa_t ( double v_comp) const
{  
  double val;
  if (v_comp >= (-66.000006) && v_comp < (-65.999994))
  {  
    val = 1.88140072945764;
  }
  else
  {  
    val = ((-45.1536175069833) * std::exp(0.166666666666667 * v_comp) + 1351767.04678348 * std::exp(0.333333333333333 * v_comp) + 0.000377071104599416) / (59874.1417151978 * v_comp * std::exp(0.333333333333333 * v_comp) - 1.67017007902457e-05 * v_comp + 3951693.35320306 * std::exp(0.333333333333333 * v_comp) - 0.00110231225215621);
  }
  return val;
}


inline 
//
double nest::i_NaTa_tMultichannelTestModel::m_inf_NaTa_t ( double v_comp) const
{  
  double val;
  if (v_comp > (-38.000006) && v_comp < (-37.999994))
  {  
    val = (0.091 * v_comp + 4.55) / (0.029 * v_comp + 2.938);
  }
  else
  {  
    val = 0.182 * (v_comp + 38.0) * (563.030236835951 * std::exp(0.166666666666667 * v_comp) - 1) * std::exp(0.166666666666667 * v_comp) / ((0.124 * v_comp + 4.712) * (std::exp(0.166666666666667 * v_comp) - 0.00177610354573438) + (v_comp + 38.0) * (102.47150310414307 * std::exp(0.166666666666667 * v_comp) - 0.182) * std::exp(0.166666666666667 * v_comp));
  }
  return val;
}


inline 
//
double nest::i_NaTa_tMultichannelTestModel::tau_m_NaTa_t ( double v_comp) const
{  
  double val;
  if (v_comp > (-38.000006) && v_comp < (-37.999994))
  {  
    val = 0.338652131302374 / (0.029 * v_comp + 2.938);
  }
  else
  {  
    val = 0.338652131302374 * (std::exp(0.166666666666667 * v_comp) - 0.00177610354573438) * (563.030236835951 * std::exp(0.166666666666667 * v_comp) - 1) / ((0.124 * v_comp + 4.712) * (std::exp(0.166666666666667 * v_comp) - 0.00177610354573438) + (v_comp + 38.0) * (102.47150310414307 * std::exp(0.166666666666667 * v_comp) - 0.182) * std::exp(0.166666666666667 * v_comp));
  }
  return val;
}

void nest::i_NaTa_tMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_NaTa_t_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_NaTa_t[chan_id];
    }
}

std::vector< double > nest::i_NaTa_tMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_NaTa_t_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_NaTa_t_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_NaTa_t channel end ///////////////////////////////////////////////////////////


// i_Kv3_1 channel //////////////////////////////////////////////////////////////////
void nest::i_Kv3_1MultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_Kv3_1_channel_count++;
        i_tot_i_Kv3_1.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable m_Kv3_1
        m_Kv3_1.push_back(6.379e-05);

        
        // channel parameter gbar_Kv3_1
        gbar_Kv3_1.push_back(0.0);
        // channel parameter e_Kv3_1
        e_Kv3_1.push_back((-85.0));

        
    }
}

void nest::i_Kv3_1MultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_Kv3_1 channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_Kv3_1" ) ){
        if(getValue< double >( channel_params, "gbar_Kv3_1" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_Kv3_1_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_Kv3_1.push_back(0);
        // state variable m_Kv3_1
        m_Kv3_1.push_back(6.379e-05);
        // i_Kv3_1 channel parameter 
        if( channel_params->known( "m_Kv3_1" ) )
            m_Kv3_1[neuron_i_Kv3_1_channel_count-1] = getValue< double >( channel_params, "m_Kv3_1" );
        
        // i_Kv3_1 channel ODE state 
        if( channel_params->known( "m_Kv3_1" ) )
            m_Kv3_1[neuron_i_Kv3_1_channel_count-1] = getValue< double >( channel_params, "m_Kv3_1" );
        

        
        // channel parameter gbar_Kv3_1
        gbar_Kv3_1.push_back(0.0);
        // channel parameter e_Kv3_1
        e_Kv3_1.push_back((-85.0));
        // i_Kv3_1 channel parameter 
        if( channel_params->known( "gbar_Kv3_1" ) )
            gbar_Kv3_1[neuron_i_Kv3_1_channel_count-1] = getValue< double >( channel_params, "gbar_Kv3_1" );
        // i_Kv3_1 channel parameter 
        if( channel_params->known( "e_Kv3_1" ) )
            e_Kv3_1[neuron_i_Kv3_1_channel_count-1] = getValue< double >( channel_params, "e_Kv3_1" );
        
    }
}

void
nest::i_Kv3_1MultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Kv3_1_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("m_Kv3_1") + std::to_string(compartment_idx))] = &m_Kv3_1[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("m_Kv3_1") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Kv3_1_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_Kv3_1") + std::to_string(compartment_idx))] = &i_tot_i_Kv3_1[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_Kv3_1") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_Kv3_1MultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_Kv3_1_channel_count, 0.);
    std::vector< double > i_val(neuron_i_Kv3_1_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_Kv3_1_channel_count, 0.);

         std::vector< double > __h(neuron_i_Kv3_1_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__m_Kv3_1__m_Kv3_1(neuron_i_Kv3_1_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_Kv3_1_channel_count; i++){
            __P__m_Kv3_1__m_Kv3_1[i] = std::exp((-__h[i]) / tau_m_Kv3_1(v_comp[i]));
            m_Kv3_1[i] = __P__m_Kv3_1__m_Kv3_1[i] * (m_Kv3_1[i] - m_inf_Kv3_1(v_comp[i])) + m_inf_Kv3_1(v_comp[i]);

            // compute the conductance of the i_Kv3_1 channel
            this->i_tot_i_Kv3_1[i] = gbar_Kv3_1[i] * (m_Kv3_1[i]) * (e_Kv3_1[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_Kv3_1[i]) * m_Kv3_1[i];
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_Kv3_1[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions Kv3_1
double nest::i_Kv3_1MultichannelTestModel::m_inf_Kv3_1 ( double v_comp) const
{  
  double val;
  val = 1.0 * std::exp(0.103092783505155 * v_comp) / (1.0 * std::exp(0.103092783505155 * v_comp) + 6.874610940966);
  return val;
}


inline 
//
double nest::i_Kv3_1MultichannelTestModel::tau_m_Kv3_1 ( double v_comp) const
{  
  double val;
  val = 4.0 * std::exp(0.0226551880380607 * v_comp) / (1.0 * std::exp(0.0226551880380607 * v_comp) + 0.348253173014273);
  return val;
}

void nest::i_Kv3_1MultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_Kv3_1_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_Kv3_1[chan_id];
    }
}

std::vector< double > nest::i_Kv3_1MultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_Kv3_1_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_Kv3_1_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_Kv3_1 channel end ///////////////////////////////////////////////////////////


// i_Ca_LVAst channel //////////////////////////////////////////////////////////////////
void nest::i_Ca_LVAstMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_Ca_LVAst_channel_count++;
        i_tot_i_Ca_LVAst.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable h_Ca_LVAst
        h_Ca_LVAst.push_back(0.08756384);
        // state variable m_Ca_LVAst
        m_Ca_LVAst.push_back(0.00291975);

        
        // channel parameter gbar_Ca_LVAst
        gbar_Ca_LVAst.push_back(0.0);
        // channel parameter e_Ca_LVAst
        e_Ca_LVAst.push_back(50.0);

        
    }
}

void nest::i_Ca_LVAstMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_Ca_LVAst channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_Ca_LVAst" ) ){
        if(getValue< double >( channel_params, "gbar_Ca_LVAst" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_Ca_LVAst_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_Ca_LVAst.push_back(0);
        // state variable h_Ca_LVAst
        h_Ca_LVAst.push_back(0.08756384);
        // state variable m_Ca_LVAst
        m_Ca_LVAst.push_back(0.00291975);
        // i_Ca_LVAst channel parameter 
        if( channel_params->known( "h_Ca_LVAst" ) )
            h_Ca_LVAst[neuron_i_Ca_LVAst_channel_count-1] = getValue< double >( channel_params, "h_Ca_LVAst" );
        // i_Ca_LVAst channel parameter 
        if( channel_params->known( "m_Ca_LVAst" ) )
            m_Ca_LVAst[neuron_i_Ca_LVAst_channel_count-1] = getValue< double >( channel_params, "m_Ca_LVAst" );
        
        // i_Ca_LVAst channel ODE state 
        if( channel_params->known( "h_Ca_LVAst" ) )
            h_Ca_LVAst[neuron_i_Ca_LVAst_channel_count-1] = getValue< double >( channel_params, "h_Ca_LVAst" );
        // i_Ca_LVAst channel ODE state 
        if( channel_params->known( "m_Ca_LVAst" ) )
            m_Ca_LVAst[neuron_i_Ca_LVAst_channel_count-1] = getValue< double >( channel_params, "m_Ca_LVAst" );
        

        
        // channel parameter gbar_Ca_LVAst
        gbar_Ca_LVAst.push_back(0.0);
        // channel parameter e_Ca_LVAst
        e_Ca_LVAst.push_back(50.0);
        // i_Ca_LVAst channel parameter 
        if( channel_params->known( "gbar_Ca_LVAst" ) )
            gbar_Ca_LVAst[neuron_i_Ca_LVAst_channel_count-1] = getValue< double >( channel_params, "gbar_Ca_LVAst" );
        // i_Ca_LVAst channel parameter 
        if( channel_params->known( "e_Ca_LVAst" ) )
            e_Ca_LVAst[neuron_i_Ca_LVAst_channel_count-1] = getValue< double >( channel_params, "e_Ca_LVAst" );
        
    }
}

void
nest::i_Ca_LVAstMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Ca_LVAst_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("h_Ca_LVAst") + std::to_string(compartment_idx))] = &h_Ca_LVAst[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("h_Ca_LVAst") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Ca_LVAst_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("m_Ca_LVAst") + std::to_string(compartment_idx))] = &m_Ca_LVAst[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("m_Ca_LVAst") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Ca_LVAst_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_Ca_LVAst") + std::to_string(compartment_idx))] = &i_tot_i_Ca_LVAst[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_Ca_LVAst") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_Ca_LVAstMultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_Ca_LVAst_channel_count, 0.);
    std::vector< double > i_val(neuron_i_Ca_LVAst_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_Ca_LVAst_channel_count, 0.);

         std::vector< double > __h(neuron_i_Ca_LVAst_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__h_Ca_LVAst__h_Ca_LVAst(neuron_i_Ca_LVAst_channel_count, 0);
        std::vector< double > __P__m_Ca_LVAst__m_Ca_LVAst(neuron_i_Ca_LVAst_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_Ca_LVAst_channel_count; i++){
            __P__h_Ca_LVAst__h_Ca_LVAst[i] = std::exp((-__h[i]) / tau_h_Ca_LVAst(v_comp[i]));
            h_Ca_LVAst[i] = __P__h_Ca_LVAst__h_Ca_LVAst[i] * (h_Ca_LVAst[i] - h_inf_Ca_LVAst(v_comp[i])) + h_inf_Ca_LVAst(v_comp[i]);
            __P__m_Ca_LVAst__m_Ca_LVAst[i] = std::exp((-__h[i]) / tau_m_Ca_LVAst(v_comp[i]));
            m_Ca_LVAst[i] = __P__m_Ca_LVAst__m_Ca_LVAst[i] * (m_Ca_LVAst[i] - m_inf_Ca_LVAst(v_comp[i])) + m_inf_Ca_LVAst(v_comp[i]);

            // compute the conductance of the i_Ca_LVAst channel
            this->i_tot_i_Ca_LVAst[i] = gbar_Ca_LVAst[i] * (h_Ca_LVAst[i] * pow(m_Ca_LVAst[i], 2)) * (e_Ca_LVAst[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_Ca_LVAst[i]) * h_Ca_LVAst[i] * pow(m_Ca_LVAst[i], 2);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_Ca_LVAst[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions Ca_LVAst
double nest::i_Ca_LVAstMultichannelTestModel::h_inf_Ca_LVAst ( double v_comp) const
{  
  double val;
  val = 1.0 / (1280165.59676428 * std::exp(0.15625 * v_comp) + 1);
  return val;
}


inline 
//
double nest::i_Ca_LVAstMultichannelTestModel::tau_h_Ca_LVAst ( double v_comp) const
{  
  double val;
  val = (8568.15374958056 * std::exp((1.0 / 7.0) * v_comp) + 23.7056491911662) / (1265.03762380433 * std::exp((1.0 / 7.0) * v_comp) + 1);
  return val;
}


inline 
//
double nest::i_Ca_LVAstMultichannelTestModel::m_inf_Ca_LVAst ( double v_comp) const
{  
  double val;
  val = 1.0 / (1 + 0.00127263380133981 * std::exp((-1.0) / 6.0 * v_comp));
  return val;
}


inline 
//
double nest::i_Ca_LVAstMultichannelTestModel::tau_m_Ca_LVAst ( double v_comp) const
{  
  double val;
  val = (1856.88578179326 * std::exp((1.0 / 5.0) * v_comp) + 8.46630328255936) / (1096.63315842846 * std::exp((1.0 / 5.0) * v_comp) + 1);
  return val;
}

void nest::i_Ca_LVAstMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_Ca_LVAst_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_Ca_LVAst[chan_id];
    }
}

std::vector< double > nest::i_Ca_LVAstMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_Ca_LVAst_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_Ca_LVAst_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_Ca_LVAst channel end ///////////////////////////////////////////////////////////


// i_Ca_HVA channel //////////////////////////////////////////////////////////////////
void nest::i_Ca_HVAMultichannelTestModel::new_channel(std::size_t comp_ass)
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        

    if(channel_contributing){
        neuron_i_Ca_HVA_channel_count++;
        i_tot_i_Ca_HVA.push_back(0);
        compartment_association.push_back(comp_ass);
        // state variable h_Ca_HVA
        h_Ca_HVA.push_back(0.69823671);
        // state variable m_Ca_HVA
        m_Ca_HVA.push_back(9.18e-06);

        
        // channel parameter gbar_Ca_HVA
        gbar_Ca_HVA.push_back(0.0);
        // channel parameter e_Ca_HVA
        e_Ca_HVA.push_back(50.0);

        
    }
}

void nest::i_Ca_HVAMultichannelTestModel::new_channel(std::size_t comp_ass, const DictionaryDatum& channel_params)
// update i_Ca_HVA channel parameters
{
    //Check whether the channel will contribute at all based on initial key-parameters. If not then don't add the channel.
    bool channel_contributing = true;
    if( channel_params->known( "gbar_Ca_HVA" ) ){
        if(getValue< double >( channel_params, "gbar_Ca_HVA" ) <= 1e-9){
            channel_contributing = false;
        }
    }else{
        
        
        if(0.0 <= 1e-9){
            channel_contributing = false;
        }
        
        
    }

    if(channel_contributing){
        neuron_i_Ca_HVA_channel_count++;
        compartment_association.push_back(comp_ass);
        i_tot_i_Ca_HVA.push_back(0);
        // state variable h_Ca_HVA
        h_Ca_HVA.push_back(0.69823671);
        // state variable m_Ca_HVA
        m_Ca_HVA.push_back(9.18e-06);
        // i_Ca_HVA channel parameter 
        if( channel_params->known( "h_Ca_HVA" ) )
            h_Ca_HVA[neuron_i_Ca_HVA_channel_count-1] = getValue< double >( channel_params, "h_Ca_HVA" );
        // i_Ca_HVA channel parameter 
        if( channel_params->known( "m_Ca_HVA" ) )
            m_Ca_HVA[neuron_i_Ca_HVA_channel_count-1] = getValue< double >( channel_params, "m_Ca_HVA" );
        
        // i_Ca_HVA channel ODE state 
        if( channel_params->known( "h_Ca_HVA" ) )
            h_Ca_HVA[neuron_i_Ca_HVA_channel_count-1] = getValue< double >( channel_params, "h_Ca_HVA" );
        // i_Ca_HVA channel ODE state 
        if( channel_params->known( "m_Ca_HVA" ) )
            m_Ca_HVA[neuron_i_Ca_HVA_channel_count-1] = getValue< double >( channel_params, "m_Ca_HVA" );
        

        
        // channel parameter gbar_Ca_HVA
        gbar_Ca_HVA.push_back(0.0);
        // channel parameter e_Ca_HVA
        e_Ca_HVA.push_back(50.0);
        // i_Ca_HVA channel parameter 
        if( channel_params->known( "gbar_Ca_HVA" ) )
            gbar_Ca_HVA[neuron_i_Ca_HVA_channel_count-1] = getValue< double >( channel_params, "gbar_Ca_HVA" );
        // i_Ca_HVA channel parameter 
        if( channel_params->known( "e_Ca_HVA" ) )
            e_Ca_HVA[neuron_i_Ca_HVA_channel_count-1] = getValue< double >( channel_params, "e_Ca_HVA" );
        
    }
}

void
nest::i_Ca_HVAMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Ca_HVA_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("h_Ca_HVA") + std::to_string(compartment_idx))] = &h_Ca_HVA[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("h_Ca_HVA") + std::to_string(compartment_idx))] = &zero_recordable;
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Ca_HVA_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("m_Ca_HVA") + std::to_string(compartment_idx))] = &m_Ca_HVA[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("m_Ca_HVA") + std::to_string(compartment_idx))] = &zero_recordable;
  
  found_rec = false;
  for(size_t chan_id = 0; chan_id < neuron_i_Ca_HVA_channel_count; chan_id++){
      if(compartment_association[chan_id] == compartment_idx){
        ( *recordables )[ Name( std::string("i_tot_i_Ca_HVA") + std::to_string(compartment_idx))] = &i_tot_i_Ca_HVA[chan_id];
        found_rec = true;
      }
  }
  if(!found_rec) ( *recordables )[ Name( std::string("i_tot_i_Ca_HVA") + std::to_string(compartment_idx))] = &zero_recordable;
}

std::pair< std::vector< double >, std::vector< double > > nest::i_Ca_HVAMultichannelTestModel::f_numstep(std::vector< double > v_comp)
{
    std::vector< double > g_val(neuron_i_Ca_HVA_channel_count, 0.);
    std::vector< double > i_val(neuron_i_Ca_HVA_channel_count, 0.);

        std::vector< double > d_i_tot_dv(neuron_i_Ca_HVA_channel_count, 0.);

         std::vector< double > __h(neuron_i_Ca_HVA_channel_count, Time::get_resolution().get_ms()); 
        std::vector< double > __P__h_Ca_HVA__h_Ca_HVA(neuron_i_Ca_HVA_channel_count, 0);
        std::vector< double > __P__m_Ca_HVA__m_Ca_HVA(neuron_i_Ca_HVA_channel_count, 0);
        #pragma omp simd
        for(std::size_t i = 0; i < neuron_i_Ca_HVA_channel_count; i++){
            __P__h_Ca_HVA__h_Ca_HVA[i] = std::exp((-__h[i]) / tau_h_Ca_HVA(v_comp[i]));
            h_Ca_HVA[i] = __P__h_Ca_HVA__h_Ca_HVA[i] * (h_Ca_HVA[i] - h_inf_Ca_HVA(v_comp[i])) + h_inf_Ca_HVA(v_comp[i]);
            __P__m_Ca_HVA__m_Ca_HVA[i] = std::exp((-__h[i]) / tau_m_Ca_HVA(v_comp[i]));
            m_Ca_HVA[i] = __P__m_Ca_HVA__m_Ca_HVA[i] * (m_Ca_HVA[i] - m_inf_Ca_HVA(v_comp[i])) + m_inf_Ca_HVA(v_comp[i]);

            // compute the conductance of the i_Ca_HVA channel
            this->i_tot_i_Ca_HVA[i] = gbar_Ca_HVA[i] * (h_Ca_HVA[i] * pow(m_Ca_HVA[i], 2)) * (e_Ca_HVA[i] - v_comp[i]);

            // derivative
            d_i_tot_dv[i] = (-gbar_Ca_HVA[i]) * h_Ca_HVA[i] * pow(m_Ca_HVA[i], 2);
            g_val[i] = - d_i_tot_dv[i];
            i_val[i] = this->i_tot_i_Ca_HVA[i] - d_i_tot_dv[i] * v_comp[i];
        }
    return std::make_pair(g_val, i_val);

}

inline 
//  functions Ca_HVA
double nest::i_Ca_HVAMultichannelTestModel::h_inf_Ca_HVA ( double v_comp) const
{  
  double val;
  val = 0.000457 * (std::exp((-1.0) / 28.0 * v_comp - 15.0 / 28.0) + 1) * std::exp((1.0 / 28.0) * v_comp + 15.0 / 28.0) / ((0.0065 * std::exp((1.0 / 50.0) * v_comp + 13.0 / 50.0) + 0.000457) * std::exp((1.0 / 28.0) * v_comp + 15.0 / 28.0) + 0.000457);
  return val;
}


inline 
//
double nest::i_Ca_HVAMultichannelTestModel::tau_h_Ca_HVA ( double v_comp) const
{  
  double val;
  val = 1.0 / (0.000457 * std::exp((-1.0) / 50.0 * v_comp - 13.0 / 50.0) + 0.0065 / (std::exp((-1.0) / 28.0 * v_comp - 15.0 / 28.0) + 1));
  return val;
}


inline 
//
double nest::i_Ca_HVAMultichannelTestModel::m_inf_Ca_HVA ( double v_comp) const
{  
  double val;
  val = (0.055 * v_comp + 1.485) * std::exp(0.321981424148607 * v_comp) / ((0.055 * v_comp + 1.485) * std::exp(0.321981424148607 * v_comp) + 0.0114057221149848 * std::exp(0.263157894736842 * v_comp) - 9.36151644263754e-06);
  return val;
}


inline 
//
double nest::i_Ca_HVAMultichannelTestModel::tau_m_Ca_HVA ( double v_comp) const
{  
  double val;
  val = (1.0 * std::exp(0.263157894736842 * v_comp) - 0.000820773673798209) * std::exp((1.0 / 17.0) * v_comp) / ((0.055 * v_comp + 1.485) * std::exp(0.321981424148607 * v_comp) + 0.0114057221149848 * std::exp(0.263157894736842 * v_comp) - 9.36151644263754e-06);
  return val;
}

void nest::i_Ca_HVAMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t chan_id = 0; chan_id < neuron_i_Ca_HVA_channel_count; chan_id++){
        compartment_to_current[this->compartment_association[chan_id]] += this->i_tot_i_Ca_HVA[chan_id];
    }
}

std::vector< double > nest::i_Ca_HVAMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_Ca_HVA_channel_count, 0.0);
    for(std::size_t chan_id = 0; chan_id < this->neuron_i_Ca_HVA_channel_count; chan_id++){
        distributed_vector[chan_id] = shared_vector[compartment_association[chan_id]];
    }
    return distributed_vector;
}

// i_Ca_HVA channel end ///////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

//////////////////////////////// concentrations

// c_ca concentration /////////////////////////////////////////////////////

void nest::c_caMultichannelTestModel::new_concentration(std::size_t comp_ass)
{
    //Check whether the concentration will contribute at all based on initial key-parameters. If not then don't add the concentration.
    bool concentration_contributing = true;

    if(concentration_contributing){
        neuron_c_ca_concentration_count++;
        c_ca.push_back(0);
        compartment_association.push_back(comp_ass);

        
        // channel parameter gamma_ca
        gamma_ca.push_back(1e-15);
        // channel parameter tau_ca
        tau_ca.push_back(100.0);
        // channel parameter inf_ca
        inf_ca.push_back(0.0001);

        
    }
}

void nest::c_caMultichannelTestModel::new_concentration(std::size_t comp_ass, const DictionaryDatum& concentration_params)
{
    //Check whether the concentration will contribute at all based on initial key-parameters. If not then don't add the concentration.
    bool concentration_contributing = true;

    if(concentration_contributing){
        neuron_c_ca_concentration_count++;
        c_ca.push_back(0);
        compartment_association.push_back(comp_ass);
        
        // c_ca concentration ODE state 
        if( concentration_params->known( "c_ca" ) )
            c_ca[neuron_c_ca_concentration_count-1] = getValue< double >( concentration_params, "c_ca" );
        

        
        // channel parameter gamma_ca
        gamma_ca.push_back(1e-15);
        // channel parameter tau_ca
        tau_ca.push_back(100.0);
        // channel parameter inf_ca
        inf_ca.push_back(0.0001);
        // c_ca concentration parameter 
        if( concentration_params->known( "gamma_ca" ) )
            gamma_ca[neuron_c_ca_concentration_count-1] = getValue< double >( concentration_params, "gamma_ca" );
        // c_ca concentration parameter 
        if( concentration_params->known( "tau_ca" ) )
            tau_ca[neuron_c_ca_concentration_count-1] = getValue< double >( concentration_params, "tau_ca" );
        // c_ca concentration parameter 
        if( concentration_params->known( "inf_ca" ) )
            inf_ca[neuron_c_ca_concentration_count-1] = getValue< double >( concentration_params, "inf_ca" );
        
    }
}

void
nest::c_caMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables,
                                               const long compartment_idx)
{
  // add state variables to recordables map
  bool found_rec = false;
  
    found_rec = false;
    for(size_t conc_id = 0; conc_id < neuron_c_ca_concentration_count; conc_id++){
      if(compartment_association[conc_id] == compartment_idx){
        ( *recordables )[ Name( std::string("c_ca") + std::to_string(compartment_idx))] = &c_ca[conc_id];
        found_rec = true;
      }
    }
    if(!found_rec) ( *recordables )[ Name( std::string("c_ca") + std::to_string(compartment_idx))] = &zero_recordable;
}

void nest::c_caMultichannelTestModel::f_numstep(std::vector< double > v_comp
                        , std::vector< double > i_Ca_HVA, std::vector< double > i_Ca_LVAst)
{
        std::vector< double > __h(neuron_c_ca_concentration_count, Time::get_resolution().get_ms());
        std::vector< double > __P__c_ca__c_ca(neuron_c_ca_concentration_count, 0);

        #pragma omp simd
        for(std::size_t i = 0; i < neuron_c_ca_concentration_count; i++){
            __P__c_ca__c_ca[i] = std::exp((-__h[i]) / tau_ca[i]);
            c_ca[i] = __P__c_ca__c_ca[i] * (c_ca[i] - gamma_ca[i] * tau_ca[i] * (i_Ca_HVA[i] + i_Ca_LVAst[i]) - inf_ca[i]) + gamma_ca[i] * tau_ca[i] * (i_Ca_HVA[i] + i_Ca_LVAst[i]) + inf_ca[i];
        }
}

void nest::c_caMultichannelTestModel::get_concentrations_per_compartment(std::vector< double >& compartment_to_concentration){
    for(std::size_t comp_id = 0; comp_id < compartment_to_concentration.size(); comp_id++){
        compartment_to_concentration[comp_id] = 0;
    }
    for(std::size_t conc_id = 0; conc_id < neuron_c_ca_concentration_count; conc_id++){
        compartment_to_concentration[this->compartment_association[conc_id]] += this->c_ca[conc_id];
    }
}

std::vector< double > nest::c_caMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_c_ca_concentration_count, 0.0);
    for(std::size_t conc_id = 0; conc_id < this->neuron_c_ca_concentration_count; conc_id++){
        distributed_vector[conc_id] = shared_vector[compartment_association[conc_id]];
    }
    return distributed_vector;
}

// c_ca concentration end ///////////////////////////////////////////////////////////



////////////////////////////////////// synapses
// i_AMPA synapse ////////////////////////////////////////////////////////////////

void nest::i_AMPAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index)
{
    neuron_i_AMPA_synapse_count++;
    i_tot_i_AMPA.push_back(0);
    compartment_association.push_back(comp_ass);
    syn_idx.push_back(syn_index);

    
    // synapse parameter e_AMPA
    e_AMPA.push_back(0);
    // synapse parameter tau_r_AMPA
    tau_r_AMPA.push_back(0.2);
    // synapse parameter tau_d_AMPA
    tau_d_AMPA.push_back(3.0);

    // set propagators to ode toolbox returned value
    __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA.push_back(0);
    __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA__d.push_back(0);
    __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA.push_back(0);
    __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_AMPA__X__spikes_AMPA.push_back(0);
    g_AMPA__X__spikes_AMPA__d.push_back(0);

    // user declared internals in order they were declared
    tp_AMPA.push_back(0);
    g_norm_AMPA.push_back(0);
}

void nest::i_AMPAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params)
// update  synapse parameters
{
    neuron_i_AMPA_synapse_count++;
    compartment_association.push_back(comp_ass);
    i_tot_i_AMPA.push_back(0);
    syn_idx.push_back(syn_index);
    
    

    
    // synapse parameter e_AMPA
    e_AMPA.push_back(0);
    // synapse parameter tau_r_AMPA
    tau_r_AMPA.push_back(0.2);
    // synapse parameter tau_d_AMPA
    tau_d_AMPA.push_back(3.0);
    if( synapse_params->known( "e_AMPA" ) )
        e_AMPA[neuron_i_AMPA_synapse_count-1] = getValue< double >( synapse_params, "e_AMPA" );
    if( synapse_params->known( "tau_r_AMPA" ) )
        tau_r_AMPA[neuron_i_AMPA_synapse_count-1] = getValue< double >( synapse_params, "tau_r_AMPA" );
    if( synapse_params->known( "tau_d_AMPA" ) )
        tau_d_AMPA[neuron_i_AMPA_synapse_count-1] = getValue< double >( synapse_params, "tau_d_AMPA" );
    

    // set propagators to ode toolbox returned value
    __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA.push_back(0);
    __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA__d.push_back(0);
    __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA.push_back(0);
    __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_AMPA__X__spikes_AMPA.push_back(0);
    g_AMPA__X__spikes_AMPA__d.push_back(0);

    // user declared internals in order they were declared
    tp_AMPA.push_back(0);
    g_norm_AMPA.push_back(0);
}

void
nest::i_AMPAMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables, const long compartment_idx)
{
  for(size_t syns_id = 0; syns_id < neuron_i_AMPA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "g_AMPA" + std::to_string(syns_id) )] = &g_AMPA__X__spikes_AMPA[syns_id];
      }
  }
    for(size_t syns_id = 0; syns_id < neuron_i_AMPA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "i_tot_i_AMPA" + std::to_string(syns_id) )] = &i_tot_i_AMPA[syns_id];
      }
    }
}
void nest::i_AMPAMultichannelTestModel::pre_run_hook()
{

    std::vector< double > __h(neuron_i_AMPA_synapse_count, Time::get_resolution().get_ms());

  for(std::size_t i = 0; i < neuron_i_AMPA_synapse_count; i++){
    // set propagators to ode toolbox returned value
    __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA[i] = 1.0 * tau_d_AMPA[i] * std::exp((-__h[i]) / tau_d_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]) - 1.0 * tau_r_AMPA[i] * std::exp((-__h[i]) / tau_r_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]);
    __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA__d[i] = (-1.0) * tau_d_AMPA[i] * tau_r_AMPA[i] * std::exp((-__h[i]) / tau_r_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]) + 1.0 * tau_d_AMPA[i] * tau_r_AMPA[i] * std::exp((-__h[i]) / tau_d_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]);
    __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA[i] = 1.0 * std::exp((-__h[i]) / tau_r_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]) - 1.0 * std::exp((-__h[i]) / tau_d_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]);
    __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA__d[i] = 1.0 * tau_d_AMPA[i] * std::exp((-__h[i]) / tau_r_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]) - 1.0 * tau_r_AMPA[i] * std::exp((-__h[i]) / tau_d_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]);

    // initial values for kernel state variables, set to zero
    g_AMPA__X__spikes_AMPA[i] = 0;
    g_AMPA__X__spikes_AMPA__d[i] = 0;

    // user declared internals in order they were declared
    tp_AMPA[i] = (tau_r_AMPA[i] * tau_d_AMPA[i]) / (tau_d_AMPA[i] - tau_r_AMPA[i]) * std::log(tau_d_AMPA[i] / tau_r_AMPA[i]);
    g_norm_AMPA[i] = 1.0 / ((-std::exp((-tp_AMPA[i]) / tau_r_AMPA[i])) + std::exp((-tp_AMPA[i]) / tau_d_AMPA[i]));

  spikes_AMPA_[i]->clear();
  }
}

std::pair< std::vector< double >, std::vector< double > > nest::i_AMPAMultichannelTestModel::f_numstep( std::vector< double > v_comp, const long lag )
{
    std::vector< double > g_val(neuron_i_AMPA_synapse_count, 0.);
    std::vector< double > i_val(neuron_i_AMPA_synapse_count, 0.);
    std::vector< double > d_i_tot_dv(neuron_i_AMPA_synapse_count, 0.);

    

    std::vector < double > s_val(neuron_i_AMPA_synapse_count, 0);

  for(std::size_t i = 0; i < neuron_i_AMPA_synapse_count; i++){
      // get spikes
      s_val[i] = spikes_AMPA_[i]->get_value( lag ); //  * g_norm_;
  }

      //update ODE state variable
  #pragma omp simd
  for(std::size_t i = 0; i < neuron_i_AMPA_synapse_count; i++){

      // update kernel state variable / compute synaptic conductance
      g_AMPA__X__spikes_AMPA[i] = __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA[i] * g_AMPA__X__spikes_AMPA[i] + __P__g_AMPA__X__spikes_AMPA__g_AMPA__X__spikes_AMPA__d[i] * g_AMPA__X__spikes_AMPA__d[i];
      g_AMPA__X__spikes_AMPA[i] += s_val[i] * 0;
      g_AMPA__X__spikes_AMPA__d[i] = __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA[i] * g_AMPA__X__spikes_AMPA[i] + __P__g_AMPA__X__spikes_AMPA__d__g_AMPA__X__spikes_AMPA__d[i] * g_AMPA__X__spikes_AMPA__d[i];
      g_AMPA__X__spikes_AMPA__d[i] += s_val[i] * g_norm_AMPA[i] * (1 / tau_r_AMPA[i] - 1 / tau_d_AMPA[i]);

      // total current
      // this expression should be the transformed inline expression

      this->i_tot_i_AMPA[i] = g_AMPA__X__spikes_AMPA[i] * (e_AMPA[i] - v_comp[i]);

      // derivative of that expression
      // voltage derivative of total current
      // compute derivative with respect to current with sympy
      d_i_tot_dv[i] = (-g_AMPA__X__spikes_AMPA[i]);

      // for numerical integration
      g_val[i] = - d_i_tot_dv[i];
      i_val[i] = this->i_tot_i_AMPA[i] - d_i_tot_dv[i] * v_comp[i];
  }

  return std::make_pair(g_val, i_val);

}

void nest::i_AMPAMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t syn_id = 0; syn_id < neuron_i_AMPA_synapse_count; syn_id++){
        compartment_to_current[this->compartment_association[syn_id]] += this->i_tot_i_AMPA[syn_id];
    }
}

std::vector< double > nest::i_AMPAMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_AMPA_synapse_count, 0.0);
    for(std::size_t syn_id = 0; syn_id < this->neuron_i_AMPA_synapse_count; syn_id++){
        distributed_vector[syn_id] = shared_vector[compartment_association[syn_id]];
    }
    return distributed_vector;
}

// i_AMPA synapse end ///////////////////////////////////////////////////////////
// i_GABA synapse ////////////////////////////////////////////////////////////////

void nest::i_GABAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index)
{
    neuron_i_GABA_synapse_count++;
    i_tot_i_GABA.push_back(0);
    compartment_association.push_back(comp_ass);
    syn_idx.push_back(syn_index);

    
    // synapse parameter e_GABA
    e_GABA.push_back((-80.0));
    // synapse parameter tau_r_GABA
    tau_r_GABA.push_back(0.2);
    // synapse parameter tau_d_GABA
    tau_d_GABA.push_back(10.0);

    // set propagators to ode toolbox returned value
    __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA.push_back(0);
    __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA__d.push_back(0);
    __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA.push_back(0);
    __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_GABA__X__spikes_GABA.push_back(0);
    g_GABA__X__spikes_GABA__d.push_back(0);

    // user declared internals in order they were declared
    tp_GABA.push_back(0);
    g_norm_GABA.push_back(0);
}

void nest::i_GABAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params)
// update  synapse parameters
{
    neuron_i_GABA_synapse_count++;
    compartment_association.push_back(comp_ass);
    i_tot_i_GABA.push_back(0);
    syn_idx.push_back(syn_index);
    
    

    
    // synapse parameter e_GABA
    e_GABA.push_back((-80.0));
    // synapse parameter tau_r_GABA
    tau_r_GABA.push_back(0.2);
    // synapse parameter tau_d_GABA
    tau_d_GABA.push_back(10.0);
    if( synapse_params->known( "e_GABA" ) )
        e_GABA[neuron_i_GABA_synapse_count-1] = getValue< double >( synapse_params, "e_GABA" );
    if( synapse_params->known( "tau_r_GABA" ) )
        tau_r_GABA[neuron_i_GABA_synapse_count-1] = getValue< double >( synapse_params, "tau_r_GABA" );
    if( synapse_params->known( "tau_d_GABA" ) )
        tau_d_GABA[neuron_i_GABA_synapse_count-1] = getValue< double >( synapse_params, "tau_d_GABA" );
    

    // set propagators to ode toolbox returned value
    __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA.push_back(0);
    __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA__d.push_back(0);
    __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA.push_back(0);
    __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_GABA__X__spikes_GABA.push_back(0);
    g_GABA__X__spikes_GABA__d.push_back(0);

    // user declared internals in order they were declared
    tp_GABA.push_back(0);
    g_norm_GABA.push_back(0);
}

void
nest::i_GABAMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables, const long compartment_idx)
{
  for(size_t syns_id = 0; syns_id < neuron_i_GABA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "g_GABA" + std::to_string(syns_id) )] = &g_GABA__X__spikes_GABA[syns_id];
      }
  }
    for(size_t syns_id = 0; syns_id < neuron_i_GABA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "i_tot_i_GABA" + std::to_string(syns_id) )] = &i_tot_i_GABA[syns_id];
      }
    }
}
void nest::i_GABAMultichannelTestModel::pre_run_hook()
{

    std::vector< double > __h(neuron_i_GABA_synapse_count, Time::get_resolution().get_ms());

  for(std::size_t i = 0; i < neuron_i_GABA_synapse_count; i++){
    // set propagators to ode toolbox returned value
    __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA[i] = 1.0 * tau_d_GABA[i] * std::exp((-__h[i]) / tau_d_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]) - 1.0 * tau_r_GABA[i] * std::exp((-__h[i]) / tau_r_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]);
    __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA__d[i] = (-1.0) * tau_d_GABA[i] * tau_r_GABA[i] * std::exp((-__h[i]) / tau_r_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]) + 1.0 * tau_d_GABA[i] * tau_r_GABA[i] * std::exp((-__h[i]) / tau_d_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]);
    __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA[i] = 1.0 * std::exp((-__h[i]) / tau_r_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]) - 1.0 * std::exp((-__h[i]) / tau_d_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]);
    __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA__d[i] = 1.0 * tau_d_GABA[i] * std::exp((-__h[i]) / tau_r_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]) - 1.0 * tau_r_GABA[i] * std::exp((-__h[i]) / tau_d_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]);

    // initial values for kernel state variables, set to zero
    g_GABA__X__spikes_GABA[i] = 0;
    g_GABA__X__spikes_GABA__d[i] = 0;

    // user declared internals in order they were declared
    tp_GABA[i] = (tau_r_GABA[i] * tau_d_GABA[i]) / (tau_d_GABA[i] - tau_r_GABA[i]) * std::log(tau_d_GABA[i] / tau_r_GABA[i]);
    g_norm_GABA[i] = 1.0 / ((-std::exp((-tp_GABA[i]) / tau_r_GABA[i])) + std::exp((-tp_GABA[i]) / tau_d_GABA[i]));

  spikes_GABA_[i]->clear();
  }
}

std::pair< std::vector< double >, std::vector< double > > nest::i_GABAMultichannelTestModel::f_numstep( std::vector< double > v_comp, const long lag )
{
    std::vector< double > g_val(neuron_i_GABA_synapse_count, 0.);
    std::vector< double > i_val(neuron_i_GABA_synapse_count, 0.);
    std::vector< double > d_i_tot_dv(neuron_i_GABA_synapse_count, 0.);

    

    std::vector < double > s_val(neuron_i_GABA_synapse_count, 0);

  for(std::size_t i = 0; i < neuron_i_GABA_synapse_count; i++){
      // get spikes
      s_val[i] = spikes_GABA_[i]->get_value( lag ); //  * g_norm_;
  }

      //update ODE state variable
  #pragma omp simd
  for(std::size_t i = 0; i < neuron_i_GABA_synapse_count; i++){

      // update kernel state variable / compute synaptic conductance
      g_GABA__X__spikes_GABA[i] = __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA[i] * g_GABA__X__spikes_GABA[i] + __P__g_GABA__X__spikes_GABA__g_GABA__X__spikes_GABA__d[i] * g_GABA__X__spikes_GABA__d[i];
      g_GABA__X__spikes_GABA[i] += s_val[i] * 0;
      g_GABA__X__spikes_GABA__d[i] = __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA[i] * g_GABA__X__spikes_GABA[i] + __P__g_GABA__X__spikes_GABA__d__g_GABA__X__spikes_GABA__d[i] * g_GABA__X__spikes_GABA__d[i];
      g_GABA__X__spikes_GABA__d[i] += s_val[i] * g_norm_GABA[i] * (1 / tau_r_GABA[i] - 1 / tau_d_GABA[i]);

      // total current
      // this expression should be the transformed inline expression

      this->i_tot_i_GABA[i] = g_GABA__X__spikes_GABA[i] * (e_GABA[i] - v_comp[i]);

      // derivative of that expression
      // voltage derivative of total current
      // compute derivative with respect to current with sympy
      d_i_tot_dv[i] = (-g_GABA__X__spikes_GABA[i]);

      // for numerical integration
      g_val[i] = - d_i_tot_dv[i];
      i_val[i] = this->i_tot_i_GABA[i] - d_i_tot_dv[i] * v_comp[i];
  }

  return std::make_pair(g_val, i_val);

}

void nest::i_GABAMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t syn_id = 0; syn_id < neuron_i_GABA_synapse_count; syn_id++){
        compartment_to_current[this->compartment_association[syn_id]] += this->i_tot_i_GABA[syn_id];
    }
}

std::vector< double > nest::i_GABAMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_GABA_synapse_count, 0.0);
    for(std::size_t syn_id = 0; syn_id < this->neuron_i_GABA_synapse_count; syn_id++){
        distributed_vector[syn_id] = shared_vector[compartment_association[syn_id]];
    }
    return distributed_vector;
}

// i_GABA synapse end ///////////////////////////////////////////////////////////
// i_NMDA synapse ////////////////////////////////////////////////////////////////

void nest::i_NMDAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index)
{
    neuron_i_NMDA_synapse_count++;
    i_tot_i_NMDA.push_back(0);
    compartment_association.push_back(comp_ass);
    syn_idx.push_back(syn_index);

    
    // synapse parameter e_NMDA
    e_NMDA.push_back(0);
    // synapse parameter tau_r_NMDA
    tau_r_NMDA.push_back(0.2);
    // synapse parameter tau_d_NMDA
    tau_d_NMDA.push_back(43.0);

    // set propagators to ode toolbox returned value
    __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA.push_back(0);
    __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA__d.push_back(0);
    __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA.push_back(0);
    __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_NMDA__X__spikes_NMDA.push_back(0);
    g_NMDA__X__spikes_NMDA__d.push_back(0);

    // user declared internals in order they were declared
    tp_NMDA.push_back(0);
    g_norm_NMDA.push_back(0);
}

void nest::i_NMDAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params)
// update  synapse parameters
{
    neuron_i_NMDA_synapse_count++;
    compartment_association.push_back(comp_ass);
    i_tot_i_NMDA.push_back(0);
    syn_idx.push_back(syn_index);
    
    

    
    // synapse parameter e_NMDA
    e_NMDA.push_back(0);
    // synapse parameter tau_r_NMDA
    tau_r_NMDA.push_back(0.2);
    // synapse parameter tau_d_NMDA
    tau_d_NMDA.push_back(43.0);
    if( synapse_params->known( "e_NMDA" ) )
        e_NMDA[neuron_i_NMDA_synapse_count-1] = getValue< double >( synapse_params, "e_NMDA" );
    if( synapse_params->known( "tau_r_NMDA" ) )
        tau_r_NMDA[neuron_i_NMDA_synapse_count-1] = getValue< double >( synapse_params, "tau_r_NMDA" );
    if( synapse_params->known( "tau_d_NMDA" ) )
        tau_d_NMDA[neuron_i_NMDA_synapse_count-1] = getValue< double >( synapse_params, "tau_d_NMDA" );
    

    // set propagators to ode toolbox returned value
    __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA.push_back(0);
    __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA__d.push_back(0);
    __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA.push_back(0);
    __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_NMDA__X__spikes_NMDA.push_back(0);
    g_NMDA__X__spikes_NMDA__d.push_back(0);

    // user declared internals in order they were declared
    tp_NMDA.push_back(0);
    g_norm_NMDA.push_back(0);
}

void
nest::i_NMDAMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables, const long compartment_idx)
{
  for(size_t syns_id = 0; syns_id < neuron_i_NMDA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "g_NMDA" + std::to_string(syns_id) )] = &g_NMDA__X__spikes_NMDA[syns_id];
      }
  }
    for(size_t syns_id = 0; syns_id < neuron_i_NMDA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "i_tot_i_NMDA" + std::to_string(syns_id) )] = &i_tot_i_NMDA[syns_id];
      }
    }
}
void nest::i_NMDAMultichannelTestModel::pre_run_hook()
{

    std::vector< double > __h(neuron_i_NMDA_synapse_count, Time::get_resolution().get_ms());

  for(std::size_t i = 0; i < neuron_i_NMDA_synapse_count; i++){
    // set propagators to ode toolbox returned value
    __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA[i] = 1.0 * tau_d_NMDA[i] * std::exp((-__h[i]) / tau_d_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]) - 1.0 * tau_r_NMDA[i] * std::exp((-__h[i]) / tau_r_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]);
    __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA__d[i] = (-1.0) * tau_d_NMDA[i] * tau_r_NMDA[i] * std::exp((-__h[i]) / tau_r_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]) + 1.0 * tau_d_NMDA[i] * tau_r_NMDA[i] * std::exp((-__h[i]) / tau_d_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]);
    __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA[i] = 1.0 * std::exp((-__h[i]) / tau_r_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]) - 1.0 * std::exp((-__h[i]) / tau_d_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]);
    __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA__d[i] = 1.0 * tau_d_NMDA[i] * std::exp((-__h[i]) / tau_r_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]) - 1.0 * tau_r_NMDA[i] * std::exp((-__h[i]) / tau_d_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]);

    // initial values for kernel state variables, set to zero
    g_NMDA__X__spikes_NMDA[i] = 0;
    g_NMDA__X__spikes_NMDA__d[i] = 0;

    // user declared internals in order they were declared
    tp_NMDA[i] = (tau_r_NMDA[i] * tau_d_NMDA[i]) / (tau_d_NMDA[i] - tau_r_NMDA[i]) * std::log(tau_d_NMDA[i] / tau_r_NMDA[i]);
    g_norm_NMDA[i] = 1.0 / ((-std::exp((-tp_NMDA[i]) / tau_r_NMDA[i])) + std::exp((-tp_NMDA[i]) / tau_d_NMDA[i]));

  spikes_NMDA_[i]->clear();
  }
}

std::pair< std::vector< double >, std::vector< double > > nest::i_NMDAMultichannelTestModel::f_numstep( std::vector< double > v_comp, const long lag )
{
    std::vector< double > g_val(neuron_i_NMDA_synapse_count, 0.);
    std::vector< double > i_val(neuron_i_NMDA_synapse_count, 0.);
    std::vector< double > d_i_tot_dv(neuron_i_NMDA_synapse_count, 0.);

    

    std::vector < double > s_val(neuron_i_NMDA_synapse_count, 0);

  for(std::size_t i = 0; i < neuron_i_NMDA_synapse_count; i++){
      // get spikes
      s_val[i] = spikes_NMDA_[i]->get_value( lag ); //  * g_norm_;
  }

      //update ODE state variable
  #pragma omp simd
  for(std::size_t i = 0; i < neuron_i_NMDA_synapse_count; i++){

      // update kernel state variable / compute synaptic conductance
      g_NMDA__X__spikes_NMDA[i] = __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA[i] * g_NMDA__X__spikes_NMDA[i] + __P__g_NMDA__X__spikes_NMDA__g_NMDA__X__spikes_NMDA__d[i] * g_NMDA__X__spikes_NMDA__d[i];
      g_NMDA__X__spikes_NMDA[i] += s_val[i] * 0;
      g_NMDA__X__spikes_NMDA__d[i] = __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA[i] * g_NMDA__X__spikes_NMDA[i] + __P__g_NMDA__X__spikes_NMDA__d__g_NMDA__X__spikes_NMDA__d[i] * g_NMDA__X__spikes_NMDA__d[i];
      g_NMDA__X__spikes_NMDA__d[i] += s_val[i] * g_norm_NMDA[i] * (1 / tau_r_NMDA[i] - 1 / tau_d_NMDA[i]);

      // total current
      // this expression should be the transformed inline expression

      this->i_tot_i_NMDA[i] = g_NMDA__X__spikes_NMDA[i] * (e_NMDA[i] - v_comp[i]) / (1.0 + 0.3 * std::exp((-0.1) * v_comp[i]));

      // derivative of that expression
      // voltage derivative of total current
      // compute derivative with respect to current with sympy
      d_i_tot_dv[i] = (-g_NMDA__X__spikes_NMDA[i]) / (1.0 + 0.3 * std::exp((-0.1) * v_comp[i])) + 0.03 * g_NMDA__X__spikes_NMDA[i] * (e_NMDA[i] - v_comp[i]) * std::exp((-0.1) * v_comp[i]) / pow((1.0 + 0.3 * std::exp((-0.1) * v_comp[i])), 2);

      // for numerical integration
      g_val[i] = - d_i_tot_dv[i];
      i_val[i] = this->i_tot_i_NMDA[i] - d_i_tot_dv[i] * v_comp[i];
  }

  return std::make_pair(g_val, i_val);

}

void nest::i_NMDAMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t syn_id = 0; syn_id < neuron_i_NMDA_synapse_count; syn_id++){
        compartment_to_current[this->compartment_association[syn_id]] += this->i_tot_i_NMDA[syn_id];
    }
}

std::vector< double > nest::i_NMDAMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_NMDA_synapse_count, 0.0);
    for(std::size_t syn_id = 0; syn_id < this->neuron_i_NMDA_synapse_count; syn_id++){
        distributed_vector[syn_id] = shared_vector[compartment_association[syn_id]];
    }
    return distributed_vector;
}

// i_NMDA synapse end ///////////////////////////////////////////////////////////
// i_AMPA_NMDA synapse ////////////////////////////////////////////////////////////////

void nest::i_AMPA_NMDAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index)
{
    neuron_i_AMPA_NMDA_synapse_count++;
    i_tot_i_AMPA_NMDA.push_back(0);
    compartment_association.push_back(comp_ass);
    syn_idx.push_back(syn_index);

    
    // synapse parameter e_AN_AMPA
    e_AN_AMPA.push_back(0);
    // synapse parameter tau_r_AN_AMPA
    tau_r_AN_AMPA.push_back(0.2);
    // synapse parameter tau_d_AN_AMPA
    tau_d_AN_AMPA.push_back(3.0);
    // synapse parameter e_AN_NMDA
    e_AN_NMDA.push_back(0);
    // synapse parameter tau_r_AN_NMDA
    tau_r_AN_NMDA.push_back(0.2);
    // synapse parameter tau_d_AN_NMDA
    tau_d_AN_NMDA.push_back(43.0);
    // synapse parameter NMDA_ratio
    NMDA_ratio.push_back(2.0);

    // set propagators to ode toolbox returned value
    __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN.push_back(0);
    __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN__d.push_back(0);
    __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN.push_back(0);
    __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN__d.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN__d.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_AN_NMDA__X__spikes_AN.push_back(0);
    g_AN_NMDA__X__spikes_AN__d.push_back(0);
    g_AN_AMPA__X__spikes_AN.push_back(0);
    g_AN_AMPA__X__spikes_AN__d.push_back(0);

    // user declared internals in order they were declared
    tp_AN_AMPA.push_back(0);
    g_norm_AN_AMPA.push_back(0);
    tp_AN_NMDA.push_back(0);
    g_norm_AN_NMDA.push_back(0);
}

void nest::i_AMPA_NMDAMultichannelTestModel::new_synapse(std::size_t comp_ass, const long syn_index, const DictionaryDatum& synapse_params)
// update  synapse parameters
{
    neuron_i_AMPA_NMDA_synapse_count++;
    compartment_association.push_back(comp_ass);
    i_tot_i_AMPA_NMDA.push_back(0);
    syn_idx.push_back(syn_index);
    
    

    
    // synapse parameter e_AN_AMPA
    e_AN_AMPA.push_back(0);
    // synapse parameter tau_r_AN_AMPA
    tau_r_AN_AMPA.push_back(0.2);
    // synapse parameter tau_d_AN_AMPA
    tau_d_AN_AMPA.push_back(3.0);
    // synapse parameter e_AN_NMDA
    e_AN_NMDA.push_back(0);
    // synapse parameter tau_r_AN_NMDA
    tau_r_AN_NMDA.push_back(0.2);
    // synapse parameter tau_d_AN_NMDA
    tau_d_AN_NMDA.push_back(43.0);
    // synapse parameter NMDA_ratio
    NMDA_ratio.push_back(2.0);
    if( synapse_params->known( "e_AN_AMPA" ) )
        e_AN_AMPA[neuron_i_AMPA_NMDA_synapse_count-1] = getValue< double >( synapse_params, "e_AN_AMPA" );
    if( synapse_params->known( "tau_r_AN_AMPA" ) )
        tau_r_AN_AMPA[neuron_i_AMPA_NMDA_synapse_count-1] = getValue< double >( synapse_params, "tau_r_AN_AMPA" );
    if( synapse_params->known( "tau_d_AN_AMPA" ) )
        tau_d_AN_AMPA[neuron_i_AMPA_NMDA_synapse_count-1] = getValue< double >( synapse_params, "tau_d_AN_AMPA" );
    if( synapse_params->known( "e_AN_NMDA" ) )
        e_AN_NMDA[neuron_i_AMPA_NMDA_synapse_count-1] = getValue< double >( synapse_params, "e_AN_NMDA" );
    if( synapse_params->known( "tau_r_AN_NMDA" ) )
        tau_r_AN_NMDA[neuron_i_AMPA_NMDA_synapse_count-1] = getValue< double >( synapse_params, "tau_r_AN_NMDA" );
    if( synapse_params->known( "tau_d_AN_NMDA" ) )
        tau_d_AN_NMDA[neuron_i_AMPA_NMDA_synapse_count-1] = getValue< double >( synapse_params, "tau_d_AN_NMDA" );
    if( synapse_params->known( "NMDA_ratio" ) )
        NMDA_ratio[neuron_i_AMPA_NMDA_synapse_count-1] = getValue< double >( synapse_params, "NMDA_ratio" );
    

    // set propagators to ode toolbox returned value
    __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN.push_back(0);
    __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN__d.push_back(0);
    __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN.push_back(0);
    __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN__d.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN__d.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN.push_back(0);
    __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN__d.push_back(0);

    // initial values for kernel state variables, set to zero
    g_AN_NMDA__X__spikes_AN.push_back(0);
    g_AN_NMDA__X__spikes_AN__d.push_back(0);
    g_AN_AMPA__X__spikes_AN.push_back(0);
    g_AN_AMPA__X__spikes_AN__d.push_back(0);

    // user declared internals in order they were declared
    tp_AN_AMPA.push_back(0);
    g_norm_AN_AMPA.push_back(0);
    tp_AN_NMDA.push_back(0);
    g_norm_AN_NMDA.push_back(0);
}

void
nest::i_AMPA_NMDAMultichannelTestModel::append_recordables(std::map< Name, double* >* recordables, const long compartment_idx)
{
  for(size_t syns_id = 0; syns_id < neuron_i_AMPA_NMDA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "g_AN_NMDA" + std::to_string(syns_id) )] = &g_AN_NMDA__X__spikes_AN[syns_id];
      }
  }
  for(size_t syns_id = 0; syns_id < neuron_i_AMPA_NMDA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "g_AN_AMPA" + std::to_string(syns_id) )] = &g_AN_AMPA__X__spikes_AN[syns_id];
      }
  }
    for(size_t syns_id = 0; syns_id < neuron_i_AMPA_NMDA_synapse_count; syns_id++){
      if(compartment_association[syns_id] == compartment_idx){
        ( *recordables )[ Name( "i_tot_i_AMPA_NMDA" + std::to_string(syns_id) )] = &i_tot_i_AMPA_NMDA[syns_id];
      }
    }
}
void nest::i_AMPA_NMDAMultichannelTestModel::pre_run_hook()
{

    std::vector< double > __h(neuron_i_AMPA_NMDA_synapse_count, Time::get_resolution().get_ms());

  for(std::size_t i = 0; i < neuron_i_AMPA_NMDA_synapse_count; i++){
    // set propagators to ode toolbox returned value
    __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN[i] = 1.0 * tau_d_AN_NMDA[i] * std::exp((-__h[i]) / tau_d_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]) - 1.0 * tau_r_AN_NMDA[i] * std::exp((-__h[i]) / tau_r_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]);
    __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN__d[i] = (-1.0) * tau_d_AN_NMDA[i] * tau_r_AN_NMDA[i] * std::exp((-__h[i]) / tau_r_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]) + 1.0 * tau_d_AN_NMDA[i] * tau_r_AN_NMDA[i] * std::exp((-__h[i]) / tau_d_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]);
    __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN[i] = 1.0 * std::exp((-__h[i]) / tau_r_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]) - 1.0 * std::exp((-__h[i]) / tau_d_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]);
    __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN__d[i] = 1.0 * tau_d_AN_NMDA[i] * std::exp((-__h[i]) / tau_r_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]) - 1.0 * tau_r_AN_NMDA[i] * std::exp((-__h[i]) / tau_d_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]);
    __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN[i] = 1.0 * tau_d_AN_AMPA[i] * std::exp((-__h[i]) / tau_d_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]) - 1.0 * tau_r_AN_AMPA[i] * std::exp((-__h[i]) / tau_r_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]);
    __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN__d[i] = (-1.0) * tau_d_AN_AMPA[i] * tau_r_AN_AMPA[i] * std::exp((-__h[i]) / tau_r_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]) + 1.0 * tau_d_AN_AMPA[i] * tau_r_AN_AMPA[i] * std::exp((-__h[i]) / tau_d_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]);
    __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN[i] = 1.0 * std::exp((-__h[i]) / tau_r_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]) - 1.0 * std::exp((-__h[i]) / tau_d_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]);
    __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN__d[i] = 1.0 * tau_d_AN_AMPA[i] * std::exp((-__h[i]) / tau_r_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]) - 1.0 * tau_r_AN_AMPA[i] * std::exp((-__h[i]) / tau_d_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]);

    // initial values for kernel state variables, set to zero
    g_AN_NMDA__X__spikes_AN[i] = 0;
    g_AN_NMDA__X__spikes_AN__d[i] = 0;
    g_AN_AMPA__X__spikes_AN[i] = 0;
    g_AN_AMPA__X__spikes_AN__d[i] = 0;

    // user declared internals in order they were declared
    tp_AN_AMPA[i] = (tau_r_AN_AMPA[i] * tau_d_AN_AMPA[i]) / (tau_d_AN_AMPA[i] - tau_r_AN_AMPA[i]) * std::log(tau_d_AN_AMPA[i] / tau_r_AN_AMPA[i]);
    g_norm_AN_AMPA[i] = 1.0 / ((-std::exp((-tp_AN_AMPA[i]) / tau_r_AN_AMPA[i])) + std::exp((-tp_AN_AMPA[i]) / tau_d_AN_AMPA[i]));
    tp_AN_NMDA[i] = (tau_r_AN_NMDA[i] * tau_d_AN_NMDA[i]) / (tau_d_AN_NMDA[i] - tau_r_AN_NMDA[i]) * std::log(tau_d_AN_NMDA[i] / tau_r_AN_NMDA[i]);
    g_norm_AN_NMDA[i] = 1.0 / ((-std::exp((-tp_AN_NMDA[i]) / tau_r_AN_NMDA[i])) + std::exp((-tp_AN_NMDA[i]) / tau_d_AN_NMDA[i]));

  spikes_AN_[i]->clear();
  }
}

std::pair< std::vector< double >, std::vector< double > > nest::i_AMPA_NMDAMultichannelTestModel::f_numstep( std::vector< double > v_comp, const long lag )
{
    std::vector< double > g_val(neuron_i_AMPA_NMDA_synapse_count, 0.);
    std::vector< double > i_val(neuron_i_AMPA_NMDA_synapse_count, 0.);
    std::vector< double > d_i_tot_dv(neuron_i_AMPA_NMDA_synapse_count, 0.);

    

    std::vector < double > s_val(neuron_i_AMPA_NMDA_synapse_count, 0);

  for(std::size_t i = 0; i < neuron_i_AMPA_NMDA_synapse_count; i++){
      // get spikes
      s_val[i] = spikes_AN_[i]->get_value( lag ); //  * g_norm_;
  }

      //update ODE state variable
  #pragma omp simd
  for(std::size_t i = 0; i < neuron_i_AMPA_NMDA_synapse_count; i++){

      // update kernel state variable / compute synaptic conductance
      g_AN_NMDA__X__spikes_AN[i] = __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN[i] * g_AN_NMDA__X__spikes_AN[i] + __P__g_AN_NMDA__X__spikes_AN__g_AN_NMDA__X__spikes_AN__d[i] * g_AN_NMDA__X__spikes_AN__d[i];
      g_AN_NMDA__X__spikes_AN[i] += s_val[i] * 0;
      g_AN_NMDA__X__spikes_AN__d[i] = __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN[i] * g_AN_NMDA__X__spikes_AN[i] + __P__g_AN_NMDA__X__spikes_AN__d__g_AN_NMDA__X__spikes_AN__d[i] * g_AN_NMDA__X__spikes_AN__d[i];
      g_AN_NMDA__X__spikes_AN__d[i] += s_val[i] * g_norm_AN_NMDA[i] * (1 / tau_r_AN_NMDA[i] - 1 / tau_d_AN_NMDA[i]);
      g_AN_AMPA__X__spikes_AN[i] = __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN[i] * g_AN_AMPA__X__spikes_AN[i] + __P__g_AN_AMPA__X__spikes_AN__g_AN_AMPA__X__spikes_AN__d[i] * g_AN_AMPA__X__spikes_AN__d[i];
      g_AN_AMPA__X__spikes_AN[i] += s_val[i] * 0;
      g_AN_AMPA__X__spikes_AN__d[i] = __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN[i] * g_AN_AMPA__X__spikes_AN[i] + __P__g_AN_AMPA__X__spikes_AN__d__g_AN_AMPA__X__spikes_AN__d[i] * g_AN_AMPA__X__spikes_AN__d[i];
      g_AN_AMPA__X__spikes_AN__d[i] += s_val[i] * g_norm_AN_AMPA[i] * (1 / tau_r_AN_AMPA[i] - 1 / tau_d_AN_AMPA[i]);

      // total current
      // this expression should be the transformed inline expression

      this->i_tot_i_AMPA_NMDA[i] = g_AN_AMPA__X__spikes_AN[i] * (e_AN_AMPA[i] - v_comp[i]) + NMDA_ratio[i] * g_AN_NMDA__X__spikes_AN[i] * (e_AN_NMDA[i] - v_comp[i]) / (1.0 + 0.3 * std::exp((-0.1) * v_comp[i]));

      // derivative of that expression
      // voltage derivative of total current
      // compute derivative with respect to current with sympy
      d_i_tot_dv[i] = (-NMDA_ratio[i]) * g_AN_NMDA__X__spikes_AN[i] / (1.0 + 0.3 * std::exp((-0.1) * v_comp[i])) + 0.03 * NMDA_ratio[i] * g_AN_NMDA__X__spikes_AN[i] * (e_AN_NMDA[i] - v_comp[i]) * std::exp((-0.1) * v_comp[i]) / pow((1.0 + 0.3 * std::exp((-0.1) * v_comp[i])), 2) - g_AN_AMPA__X__spikes_AN[i];

      // for numerical integration
      g_val[i] = - d_i_tot_dv[i];
      i_val[i] = this->i_tot_i_AMPA_NMDA[i] - d_i_tot_dv[i] * v_comp[i];
  }

  return std::make_pair(g_val, i_val);

}

void nest::i_AMPA_NMDAMultichannelTestModel::get_currents_per_compartment(std::vector< double >& compartment_to_current){
    for(std::size_t comp_id = 0; comp_id < compartment_to_current.size(); comp_id++){
        compartment_to_current[comp_id] = 0;
    }
    for(std::size_t syn_id = 0; syn_id < neuron_i_AMPA_NMDA_synapse_count; syn_id++){
        compartment_to_current[this->compartment_association[syn_id]] += this->i_tot_i_AMPA_NMDA[syn_id];
    }
}

std::vector< double > nest::i_AMPA_NMDAMultichannelTestModel::distribute_shared_vector(std::vector< double > shared_vector){
    std::vector< double > distributed_vector(this->neuron_i_AMPA_NMDA_synapse_count, 0.0);
    for(std::size_t syn_id = 0; syn_id < this->neuron_i_AMPA_NMDA_synapse_count; syn_id++){
        distributed_vector[syn_id] = shared_vector[compartment_association[syn_id]];
    }
    return distributed_vector;
}

// i_AMPA_NMDA synapse end ///////////////////////////////////////////////////////////


////////////////////////////////////// continuous inputs





