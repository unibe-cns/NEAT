/*
 *  cm_tree.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "cm_tree_multichannel_test_model.h"

nest::CompartmentMultichannelTestModel::CompartmentMultichannelTestModel( const long compartment_index, const long parent_index)
  : xx_( 0.0 )
  , yy_( 0.0 )
  , comp_index( compartment_index )
  , p_index( parent_index )
  , parent( nullptr )
  , v_comp( new double(0.0) )
  , ca( 1.0 )
  , gc( 0.01 )
  , gl( 0.1 )
  , el( -70. )
  , gg0( nullptr )
  , ca__div__dt( nullptr )
  , gl__times__el( nullptr )
  , ff( nullptr )
  , gg( nullptr )
  , hh( 0.0 )
  , n_passed( 0 )
{
  *v_comp = el;
}

nest::CompartmentMultichannelTestModel::CompartmentMultichannelTestModel( const long compartment_index, const long parent_index,
    double* v_comp_ref, double* ca__div__dt_ref, double* gl__times__el_ref, double* gg0_ref, double* gg_ref, double* ff_ref )
  : xx_( 0.0 )
  , yy_( 0.0 )
  , comp_index( compartment_index )
  , p_index( parent_index )
  , parent( nullptr )
  , v_comp( v_comp_ref )
  , ca( 1.0 )
  , gc( 0.01 )
  , gl( 0.1 )
  , el( -70. )
  , gg0( gg0_ref )
  , ca__div__dt( ca__div__dt_ref )
  , gl__times__el( gl__times__el_ref )
  , ff( ff_ref )
  , gg( gg_ref )
  , hh( 0.0 )
  , n_passed( 0 )
{
  *v_comp = el;
}
nest::CompartmentMultichannelTestModel::CompartmentMultichannelTestModel( const long compartment_index,
  const long parent_index,
  const DictionaryDatum& compartment_params,
  double* v_comp_ref, double* ca__div__dt_ref, double* gl__times__el_ref, double* gg0_ref, double* gg_ref, double* ff_ref)
  : xx_( 0.0 )
  , yy_( 0.0 )
  , comp_index( compartment_index )
  , p_index( parent_index )
  , parent( nullptr )
  , v_comp( v_comp_ref )
  , ca( 1.0 )
  , gc( 0.01 )
  , gl( 0.1 )
  , el( -70. )
  , gg0( gg0_ref )
  , ca__div__dt( ca__div__dt_ref )
  , gl__times__el( gl__times__el_ref )
  , ff( ff_ref )
  , gg( gg_ref )
  , hh( 0.0 )
  , n_passed( 0 )
{

  double v_comp_update;
  updateValue< double >( compartment_params, names::C_m, ca );
  updateValue< double >( compartment_params, names::g_C, gc );
  updateValue< double >( compartment_params, names::g_L, gl );
  updateValue< double >( compartment_params, names::e_L, el );
  updateValue< double >( compartment_params, "v_comp", v_comp_update);

  *v_comp = v_comp_update;
}

void
nest::CompartmentMultichannelTestModel::pre_run_hook()
{

  const double dt = Time::get_resolution().get_ms();
  // used in vectorized passive loop
  *ca__div__dt = ca / dt;
  *gl__times__el = gl * el;
  *gg0 = *ca__div__dt + gl;
}

std::map< Name, double* >
nest::CompartmentMultichannelTestModel::get_recordables()
{
  std::map< Name, double* > recordables;

  recordables[ Name( "v_comp" + std::to_string( comp_index ) ) ] = v_comp;

  return recordables;
}

// for matrix construction
void
nest::CompartmentMultichannelTestModel::construct_matrix_coupling_elements()
{
  if ( parent != nullptr )
  {
    *gg += gc;
    // matrix off diagonal element
    hh = -gc;
  }

  for ( auto child_it = children.begin(); child_it != children.end(); ++child_it )
  {
    *gg += ( *child_it ).gc;
  }
}


nest::CompTreeMultichannelTestModel::CompTreeMultichannelTestModel()
  : root_( -1, -1 )
  , size_( 0 )
{
  compartments_.resize( 0 );
  leafs_.resize( 0 );
}

/**
 * Add a compartment to the tree structure via the python interface
 * root shoud have -1 as parent index. Add root compartment first.
 * Assumes parent of compartment is already added
 */
void
nest::CompTreeMultichannelTestModel::add_compartment( const long parent_index )
{
  v_comp_vec.push_back(0.0);
  ca__div__dt_vec.push_back(0.0);
  gl__times__el_vec.push_back(0.0);
  gg0_vec.push_back(0.0);
  gg_vec.push_back(0.0);
  ff_vec.push_back(0.0);
  size_t comp_index = v_comp_vec.size()-1;

  CompartmentMultichannelTestModel* compartment = new CompartmentMultichannelTestModel(
    size_, parent_index, &(v_comp_vec[comp_index]), &(ca__div__dt_vec[comp_index]),
    &(gl__times__el_vec[comp_index]), &(gg0_vec[comp_index]), &(gg_vec[comp_index]), &(ff_vec[comp_index])
  );

  neuron_currents.add_compartment();
  add_compartment( compartment, parent_index );
}

void
nest::CompTreeMultichannelTestModel::add_compartment( const long parent_index, const DictionaryDatum& compartment_params )
{
  //Check whether all passed parameters exist within the used neuron model:
  Dictionary* comp_param_copy = new Dictionary(*compartment_params);

  comp_param_copy->remove(names::C_m);
  comp_param_copy->remove(names::g_C);
  comp_param_copy->remove(names::g_L);
  comp_param_copy->remove(names::e_L);
  comp_param_copy->remove("v_comp");
  if( comp_param_copy->known( "gbar_h" ) ) comp_param_copy->remove("gbar_h");
  if( comp_param_copy->known( "e_h" ) ) comp_param_copy->remove("e_h");
  if( comp_param_copy->known( "hf_h" ) ) comp_param_copy->remove("hf_h");
  if( comp_param_copy->known( "hs_h" ) ) comp_param_copy->remove("hs_h");
  if( comp_param_copy->known( "hf_h" ) ) comp_param_copy->remove("hf_h");
  if( comp_param_copy->known( "hs_h" ) ) comp_param_copy->remove("hs_h");
  if( comp_param_copy->known( "gbar_TestChannel2" ) ) comp_param_copy->remove("gbar_TestChannel2");
  if( comp_param_copy->known( "e_TestChannel2" ) ) comp_param_copy->remove("e_TestChannel2");
  if( comp_param_copy->known( "a00_TestChannel2" ) ) comp_param_copy->remove("a00_TestChannel2");
  if( comp_param_copy->known( "a01_TestChannel2" ) ) comp_param_copy->remove("a01_TestChannel2");
  if( comp_param_copy->known( "a10_TestChannel2" ) ) comp_param_copy->remove("a10_TestChannel2");
  if( comp_param_copy->known( "a11_TestChannel2" ) ) comp_param_copy->remove("a11_TestChannel2");
  if( comp_param_copy->known( "a00_TestChannel2" ) ) comp_param_copy->remove("a00_TestChannel2");
  if( comp_param_copy->known( "a01_TestChannel2" ) ) comp_param_copy->remove("a01_TestChannel2");
  if( comp_param_copy->known( "a10_TestChannel2" ) ) comp_param_copy->remove("a10_TestChannel2");
  if( comp_param_copy->known( "a11_TestChannel2" ) ) comp_param_copy->remove("a11_TestChannel2");
  if( comp_param_copy->known( "gbar_TestChannel" ) ) comp_param_copy->remove("gbar_TestChannel");
  if( comp_param_copy->known( "e_TestChannel" ) ) comp_param_copy->remove("e_TestChannel");
  if( comp_param_copy->known( "a00_TestChannel" ) ) comp_param_copy->remove("a00_TestChannel");
  if( comp_param_copy->known( "a01_TestChannel" ) ) comp_param_copy->remove("a01_TestChannel");
  if( comp_param_copy->known( "a02_TestChannel" ) ) comp_param_copy->remove("a02_TestChannel");
  if( comp_param_copy->known( "a10_TestChannel" ) ) comp_param_copy->remove("a10_TestChannel");
  if( comp_param_copy->known( "a11_TestChannel" ) ) comp_param_copy->remove("a11_TestChannel");
  if( comp_param_copy->known( "a12_TestChannel" ) ) comp_param_copy->remove("a12_TestChannel");
  if( comp_param_copy->known( "a00_TestChannel" ) ) comp_param_copy->remove("a00_TestChannel");
  if( comp_param_copy->known( "a01_TestChannel" ) ) comp_param_copy->remove("a01_TestChannel");
  if( comp_param_copy->known( "a02_TestChannel" ) ) comp_param_copy->remove("a02_TestChannel");
  if( comp_param_copy->known( "a10_TestChannel" ) ) comp_param_copy->remove("a10_TestChannel");
  if( comp_param_copy->known( "a11_TestChannel" ) ) comp_param_copy->remove("a11_TestChannel");
  if( comp_param_copy->known( "a12_TestChannel" ) ) comp_param_copy->remove("a12_TestChannel");
  if( comp_param_copy->known( "gbar_SKv3_1" ) ) comp_param_copy->remove("gbar_SKv3_1");
  if( comp_param_copy->known( "e_SKv3_1" ) ) comp_param_copy->remove("e_SKv3_1");
  if( comp_param_copy->known( "z_SKv3_1" ) ) comp_param_copy->remove("z_SKv3_1");
  if( comp_param_copy->known( "z_SKv3_1" ) ) comp_param_copy->remove("z_SKv3_1");
  if( comp_param_copy->known( "gbar_SK_E2" ) ) comp_param_copy->remove("gbar_SK_E2");
  if( comp_param_copy->known( "e_SK_E2" ) ) comp_param_copy->remove("e_SK_E2");
  if( comp_param_copy->known( "z_SK_E2" ) ) comp_param_copy->remove("z_SK_E2");
  if( comp_param_copy->known( "z_SK_E2" ) ) comp_param_copy->remove("z_SK_E2");
  if( comp_param_copy->known( "gbar_SK" ) ) comp_param_copy->remove("gbar_SK");
  if( comp_param_copy->known( "e_SK" ) ) comp_param_copy->remove("e_SK");
  if( comp_param_copy->known( "z_SK" ) ) comp_param_copy->remove("z_SK");
  if( comp_param_copy->known( "z_SK" ) ) comp_param_copy->remove("z_SK");
  if( comp_param_copy->known( "gbar_PiecewiseChannel" ) ) comp_param_copy->remove("gbar_PiecewiseChannel");
  if( comp_param_copy->known( "e_PiecewiseChannel" ) ) comp_param_copy->remove("e_PiecewiseChannel");
  if( comp_param_copy->known( "a_PiecewiseChannel" ) ) comp_param_copy->remove("a_PiecewiseChannel");
  if( comp_param_copy->known( "b_PiecewiseChannel" ) ) comp_param_copy->remove("b_PiecewiseChannel");
  if( comp_param_copy->known( "a_PiecewiseChannel" ) ) comp_param_copy->remove("a_PiecewiseChannel");
  if( comp_param_copy->known( "b_PiecewiseChannel" ) ) comp_param_copy->remove("b_PiecewiseChannel");
  if( comp_param_copy->known( "gbar_Na_Ta" ) ) comp_param_copy->remove("gbar_Na_Ta");
  if( comp_param_copy->known( "e_Na_Ta" ) ) comp_param_copy->remove("e_Na_Ta");
  if( comp_param_copy->known( "h_Na_Ta" ) ) comp_param_copy->remove("h_Na_Ta");
  if( comp_param_copy->known( "m_Na_Ta" ) ) comp_param_copy->remove("m_Na_Ta");
  if( comp_param_copy->known( "h_Na_Ta" ) ) comp_param_copy->remove("h_Na_Ta");
  if( comp_param_copy->known( "m_Na_Ta" ) ) comp_param_copy->remove("m_Na_Ta");
  if( comp_param_copy->known( "gbar_NaTa_t" ) ) comp_param_copy->remove("gbar_NaTa_t");
  if( comp_param_copy->known( "e_NaTa_t" ) ) comp_param_copy->remove("e_NaTa_t");
  if( comp_param_copy->known( "h_NaTa_t" ) ) comp_param_copy->remove("h_NaTa_t");
  if( comp_param_copy->known( "m_NaTa_t" ) ) comp_param_copy->remove("m_NaTa_t");
  if( comp_param_copy->known( "h_NaTa_t" ) ) comp_param_copy->remove("h_NaTa_t");
  if( comp_param_copy->known( "m_NaTa_t" ) ) comp_param_copy->remove("m_NaTa_t");
  if( comp_param_copy->known( "gbar_Kv3_1" ) ) comp_param_copy->remove("gbar_Kv3_1");
  if( comp_param_copy->known( "e_Kv3_1" ) ) comp_param_copy->remove("e_Kv3_1");
  if( comp_param_copy->known( "m_Kv3_1" ) ) comp_param_copy->remove("m_Kv3_1");
  if( comp_param_copy->known( "m_Kv3_1" ) ) comp_param_copy->remove("m_Kv3_1");
  if( comp_param_copy->known( "gbar_Ca_LVAst" ) ) comp_param_copy->remove("gbar_Ca_LVAst");
  if( comp_param_copy->known( "e_Ca_LVAst" ) ) comp_param_copy->remove("e_Ca_LVAst");
  if( comp_param_copy->known( "h_Ca_LVAst" ) ) comp_param_copy->remove("h_Ca_LVAst");
  if( comp_param_copy->known( "m_Ca_LVAst" ) ) comp_param_copy->remove("m_Ca_LVAst");
  if( comp_param_copy->known( "h_Ca_LVAst" ) ) comp_param_copy->remove("h_Ca_LVAst");
  if( comp_param_copy->known( "m_Ca_LVAst" ) ) comp_param_copy->remove("m_Ca_LVAst");
  if( comp_param_copy->known( "gbar_Ca_HVA" ) ) comp_param_copy->remove("gbar_Ca_HVA");
  if( comp_param_copy->known( "e_Ca_HVA" ) ) comp_param_copy->remove("e_Ca_HVA");
  if( comp_param_copy->known( "h_Ca_HVA" ) ) comp_param_copy->remove("h_Ca_HVA");
  if( comp_param_copy->known( "m_Ca_HVA" ) ) comp_param_copy->remove("m_Ca_HVA");
  if( comp_param_copy->known( "h_Ca_HVA" ) ) comp_param_copy->remove("h_Ca_HVA");
  if( comp_param_copy->known( "m_Ca_HVA" ) ) comp_param_copy->remove("m_Ca_HVA");
  if( comp_param_copy->known( "gamma_ca" ) ) comp_param_copy->remove("gamma_ca");
  if( comp_param_copy->known( "tau_ca" ) ) comp_param_copy->remove("tau_ca");
  if( comp_param_copy->known( "inf_ca" ) ) comp_param_copy->remove("inf_ca");
  if( comp_param_copy->known( "c_ca" ) ) comp_param_copy->remove("c_ca");
  if( comp_param_copy->known( "e_AMPA" ) ) comp_param_copy->remove("e_AMPA");
  if( comp_param_copy->known( "tau_r_AMPA" ) ) comp_param_copy->remove("tau_r_AMPA");
  if( comp_param_copy->known( "tau_d_AMPA" ) ) comp_param_copy->remove("tau_d_AMPA");
  if( comp_param_copy->known( "e_GABA" ) ) comp_param_copy->remove("e_GABA");
  if( comp_param_copy->known( "tau_r_GABA" ) ) comp_param_copy->remove("tau_r_GABA");
  if( comp_param_copy->known( "tau_d_GABA" ) ) comp_param_copy->remove("tau_d_GABA");
  if( comp_param_copy->known( "e_NMDA" ) ) comp_param_copy->remove("e_NMDA");
  if( comp_param_copy->known( "tau_r_NMDA" ) ) comp_param_copy->remove("tau_r_NMDA");
  if( comp_param_copy->known( "tau_d_NMDA" ) ) comp_param_copy->remove("tau_d_NMDA");
  if( comp_param_copy->known( "e_AN_AMPA" ) ) comp_param_copy->remove("e_AN_AMPA");
  if( comp_param_copy->known( "tau_r_AN_AMPA" ) ) comp_param_copy->remove("tau_r_AN_AMPA");
  if( comp_param_copy->known( "tau_d_AN_AMPA" ) ) comp_param_copy->remove("tau_d_AN_AMPA");
  if( comp_param_copy->known( "e_AN_NMDA" ) ) comp_param_copy->remove("e_AN_NMDA");
  if( comp_param_copy->known( "tau_r_AN_NMDA" ) ) comp_param_copy->remove("tau_r_AN_NMDA");
  if( comp_param_copy->known( "tau_d_AN_NMDA" ) ) comp_param_copy->remove("tau_d_AN_NMDA");
  if( comp_param_copy->known( "NMDA_ratio" ) ) comp_param_copy->remove("NMDA_ratio");

  if(!comp_param_copy->empty()){
    std::string msg = "Following parameters are invalid: ";
    for(auto& param : *comp_param_copy){
        msg += param.first.toString();
        msg += "\n";
    }
    throw BadParameter(msg);
  }

  v_comp_vec.push_back(0.0);
  ca__div__dt_vec.push_back(0.0);
  gl__times__el_vec.push_back(0.0);
  gg0_vec.push_back(0.0);
  gg_vec.push_back(0.0);
  ff_vec.push_back(0.0);
  size_t comp_index = v_comp_vec.size()-1;

  CompartmentMultichannelTestModel* compartment = new CompartmentMultichannelTestModel(
    size_, parent_index, compartment_params, &(v_comp_vec[comp_index]), &(ca__div__dt_vec[comp_index]),
    &(gl__times__el_vec[comp_index]), &(gg0_vec[comp_index]), &(gg_vec[comp_index]), &(ff_vec[comp_index])
  );

  neuron_currents.add_compartment(compartment_params);
  add_compartment( compartment, parent_index );
}

void
nest::CompTreeMultichannelTestModel::add_compartment( CompartmentMultichannelTestModel* compartment, const long parent_index )
{
  size_++;

  if ( parent_index >= 0 )
  {
    /**
     * we do not raise an UnknownCompartment exception from within
     * get_compartment(), because we want to print a more informative
     * exception message
     */
    CompartmentMultichannelTestModel* parent = get_compartment( parent_index, get_root(), 0 );
    if ( parent == nullptr )
    {
      std::string msg = "does not exist in tree, but was specified as a parent compartment";
      throw UnknownCompartment( parent_index, msg );
    }

    parent->children.push_back( *compartment );
  }
  else
  {
    // we raise an error if the root already exists
    if ( root_.comp_index >= 0 )
    {
      std::string msg = ", the root, has already been instantiated";
      throw UnknownCompartment( root_.comp_index, msg );
    }
    root_ = *compartment;
  }

  compartment_indices_.push_back( compartment->comp_index );

  set_compartments();
}

/**
 * Get the compartment corresponding to the provided index in the tree.
 *
 * This function gets the compartments by a recursive search through the tree.
 *
 * The overloaded functions looks only in the subtree of the provided compartment,
 * and also has the option to throw an error if no compartment corresponding to
 * `compartment_index` is found in the tree
 */
nest::CompartmentMultichannelTestModel*
nest::CompTreeMultichannelTestModel::get_compartment( const long compartment_index ) const
{
  return get_compartment( compartment_index, get_root(), 1 );
}

nest::CompartmentMultichannelTestModel*
nest::CompTreeMultichannelTestModel::get_compartment( const long compartment_index, CompartmentMultichannelTestModel* compartment, const long raise_flag ) const
{
  CompartmentMultichannelTestModel* r_compartment = nullptr;

  if ( compartment->comp_index == compartment_index )
  {
    r_compartment = compartment;
  }
  else
  {
    auto child_it = compartment->children.begin();
    while ( ( not r_compartment ) && child_it != compartment->children.end() )
    {
      r_compartment = get_compartment( compartment_index, &( *child_it ), 0 );
      ++child_it;
    }
  }

  if ( ( not r_compartment ) && raise_flag )
  {
    std::string msg = "does not exist in tree";
    throw UnknownCompartment( compartment_index, msg );
  }

  return r_compartment;
}

/**
 * Get the compartment corresponding to the provided index in the tree. Optimized
 * trough the use of a pointer vector containing all compartments. Calling this
 * function before CompTreeMultichannelTestModel::init_pointers() is called will result in a segmentation
 * fault
 */
nest::CompartmentMultichannelTestModel*
nest::CompTreeMultichannelTestModel::get_compartment_opt( const long compartment_idx ) const
{
  return compartments_[ compartment_idx ];
}

/**
 * Initialize all tree structure pointers
 */
void
nest::CompTreeMultichannelTestModel::init_pointers()
{
  set_parents();
  set_compartments();
  set_compartment_variables();
  set_leafs();
}

/**
 * For each compartments, sets its pointer towards its parent compartment
 */
void
nest::CompTreeMultichannelTestModel::set_parents()
{
  for ( auto compartment_idx_it = compartment_indices_.begin(); compartment_idx_it != compartment_indices_.end();
        ++compartment_idx_it )
  {
    CompartmentMultichannelTestModel* comp_ptr = get_compartment( *compartment_idx_it );
    // will be nullptr if root
    CompartmentMultichannelTestModel* parent_ptr = get_compartment( comp_ptr->p_index, &root_, 0 );
    comp_ptr->parent = parent_ptr;
  }
}

/**
 * Creates a vector of compartment pointers, organized in the order in which they were
 * added by `add_compartment()`
 */
void
nest::CompTreeMultichannelTestModel::set_compartments()
{
  compartments_.clear();

  for ( auto compartment_idx_it = compartment_indices_.begin(); compartment_idx_it != compartment_indices_.end();
        ++compartment_idx_it )
  {
    compartments_.push_back( get_compartment( *compartment_idx_it ) );
  }
}

/**
 * Set pointer variables within a compartment
 */
void
nest::CompTreeMultichannelTestModel::set_compartment_variables()
{
  //reset compartment pointers due to unsafe pointers in vectors when resizing during compartment creation
  for( size_t i = 0; i < v_comp_vec.size(); i++){
    compartments_[i]->v_comp = &(v_comp_vec[i]);
    compartments_[i]->ca__div__dt = &(ca__div__dt_vec[i]);
    compartments_[i]->gl__times__el = &(gl__times__el_vec[i]);
    compartments_[i]->gg0 = &(gg0_vec[i]);
    compartments_[i]->gg = &(gg_vec[i]);
    compartments_[i]->ff = &(ff_vec[i]);
  }
}

/**
 * Creates a vector of compartment pointers of compartments that are also leafs of the tree.
 */
void
nest::CompTreeMultichannelTestModel::set_leafs()
{
  leafs_.clear();
  for ( auto compartment_it = compartments_.begin(); compartment_it != compartments_.end(); ++compartment_it )
  {
    if ( int( ( *compartment_it )->children.size() ) == 0 )
    {
      leafs_.push_back( *compartment_it );
    }
  }
}

/**
 * Initializes pointers for the spike buffers for all synapse receptors
 */
void
nest::CompTreeMultichannelTestModel::set_syn_buffers( std::vector< RingBuffer >& syn_buffers )
{
  neuron_currents.set_buffers( syn_buffers );
}

/**
 * Returns a map of variable names and pointers to the recordables
 */
std::map< Name, double* >
nest::CompTreeMultichannelTestModel::get_recordables()
{
  std::map< Name, double* > recordables;

  /**
   * add recordables for all compartments, suffixed by compartment_idx,
   * to "recordables"
   */
  for ( auto compartment_it = compartments_.begin(); compartment_it != compartments_.end(); ++compartment_it )
  {
    long comp_index = (*compartment_it)->comp_index;
    std::map< Name, double* > recordables_comp = neuron_currents.get_recordables( comp_index );
    recordables.insert( recordables_comp.begin(), recordables_comp.end() );

    recordables_comp = ( *compartment_it )->get_recordables();
    recordables.insert( recordables_comp.begin(), recordables_comp.end() );
  }
  return recordables;
}

/**
 * Initialize state variables
 */
void
nest::CompTreeMultichannelTestModel::pre_run_hook()
{
  if ( root_.comp_index < 0 )
  {
    std::string msg = "does not exist in tree, meaning that no compartments have been added";
    throw UnknownCompartment( 0, msg );
  }

  // initialize the compartments
  for ( auto compartment_it = compartments_.begin(); compartment_it != compartments_.end(); ++compartment_it )
  {
    ( *compartment_it )->pre_run_hook();
  }
    neuron_currents.pre_run_hook();
}

/**
 * Returns vector of voltage values, indices correspond to compartments in `compartments_`
 */
std::vector< double >
nest::CompTreeMultichannelTestModel::get_voltage() const
{
  return v_comp_vec;
}

/**
 * Return voltage of single compartment voltage, indicated by the compartment_index
 */
double
nest::CompTreeMultichannelTestModel::get_compartment_voltage( const long compartment_index )
{
  return *(compartments_[ compartment_index ]->v_comp);
}

/**
 * Construct the matrix equation to be solved to advance the model one timestep
 */
void
nest::CompTreeMultichannelTestModel::construct_matrix( const long lag )
{
  // compute all channel currents, receptor currents, and input currents
  std::vector< std::pair< double, double > > comps_gi = neuron_currents.f_numstep( v_comp_vec, lag );

  #pragma omp simd
  for( size_t i = 0; i < v_comp_vec.size(); i++ ){
    // passive current left hand side
    gg_vec[i] = gg0_vec[i];
    // passive currents right hand side
    ff_vec[i] = ca__div__dt_vec[i] * v_comp_vec[i] + gl__times__el_vec[i];

    // add all currents to compartment
    gg_vec[i] += comps_gi[i].first;
    ff_vec[i] += comps_gi[i].second;
  }
  for ( auto compartment_it = compartments_.begin(); compartment_it != compartments_.end(); ++compartment_it )
  {
    ( *compartment_it )->construct_matrix_coupling_elements();
  }
}

/**
 * Solve matrix with O(n) algorithm
 */
void
nest::CompTreeMultichannelTestModel::solve_matrix()
{
  std::vector< CompartmentMultichannelTestModel* >::iterator leaf_it = leafs_.begin();

  // start the down sweep (puts to zero the sub diagonal matrix elements)
  solve_matrix_downsweep( leafs_[ 0 ], leaf_it );

  // do up sweep to set voltages
  solve_matrix_upsweep( &root_, 0.0 );
}

void
nest::CompTreeMultichannelTestModel::solve_matrix_downsweep( CompartmentMultichannelTestModel* compartment, std::vector< CompartmentMultichannelTestModel* >::iterator leaf_it )
{
  // compute the input output transformation at compartment
  std::pair< double, double > output = compartment->io();

  // move on to the parent layer
  if ( compartment->parent != nullptr )
  {
    CompartmentMultichannelTestModel* parent = compartment->parent;
    // gather input from child layers
    parent->gather_input( output );
    // move on to next compartments
    ++parent->n_passed;
    if ( parent->n_passed == int( parent->children.size() ) )
    {
      parent->n_passed = 0;
      // move on to next compartment
      solve_matrix_downsweep( parent, leaf_it );
    }
    else
    {
      // start at next leaf
      ++leaf_it;
      if ( leaf_it != leafs_.end() )
      {
        solve_matrix_downsweep( *leaf_it, leaf_it );
      }
    }
  }
}

void
nest::CompTreeMultichannelTestModel::solve_matrix_upsweep( CompartmentMultichannelTestModel* compartment, double vv )
{
  // compute compartment voltage
  vv = compartment->calc_v( vv );
  // move on to child compartments
  for ( auto child_it = compartment->children.begin(); child_it != compartment->children.end(); ++child_it )
  {
    solve_matrix_upsweep( &( *child_it ), vv );
  }
}

/**
 * Print the tree graph
 */
void
nest::CompTreeMultichannelTestModel::print_tree() const
{
  // loop over all compartments
  std::printf( ">>> CM tree with %d compartments <<<\n", int( compartments_.size() ) );
  for ( int ii = 0; ii < int( compartments_.size() ); ++ii )
  {
    CompartmentMultichannelTestModel* compartment = compartments_[ ii ];
    std::cout << "    CompartmentMultichannelTestModel " << compartment->comp_index << ": ";
    std::cout << "C_m = " << compartment->ca << " nF, ";
    std::cout << "g_L = " << compartment->gl << " uS, ";
    std::cout << "e_L = " << compartment->el << " mV, ";
    if ( compartment->parent != nullptr )
    {
      std::cout << "Parent " << compartment->parent->comp_index << " --> ";
      std::cout << "g_c = " << compartment->gc << " uS, ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}