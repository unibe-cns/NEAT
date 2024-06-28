**************
Abstract Trees
**************

.. automodule:: neat.trees
.. currentmodule:: neat


Basic tree
==========

.. autoclass:: neat.STree

.. autosummary::
   :toctree: generated/

   STree.__getitem__
   STree.__len__
   STree.__iter__
   STree.__str__
   STree.__copy__
   STree.__repr__
   STree.__hash__
   STree.unique_hash
   STree.check_ordered
   STree.get_nodes
   STree.nodes
   STree.gather_nodes
   STree.get_leafs
   STree.leafs
   STree.is_leaf
   STree.root
   STree.is_root
   STree.add_node_with_parent_from_index
   STree.add_node_with_parent
   STree.soft_remove_node
   STree.remove_node
   STree.remove_single_node
   STree.insert_node
   STree.reset_indices
   STree.get_sub_tree
   STree.depth_of_node
   STree.degree_of_node
   STree.order_of_node
   STree.path_to_root
   STree.path_between_nodes
   STree.path_between_nodes_depth_first
   STree.get_nodes_in_subtree
   STree.sister_leafs
   STree.get_two_variable_expansion_points_to_root
   STree.get_two_variable_expansion_points_from_root
   STree.get_get_two_variable_expansion_pointss
   STree.get_nearest_neighbours


.. autoclass:: neat.SNode


Compartment Tree
================

.. autoclass:: neat.CompartmentTree

.. autosummary::
   :toctree: generated/

   CompartmentTree.add_channel_current
   CompartmentTree.set_expansion_points
   CompartmentTree.set_e_eq
   CompartmentTree.get_e_eq
   CompartmentTree.fit_e_leak
   CompartmentTree.get_equivalent_locs
   CompartmentTree.calc_impedance_matrix
   CompartmentTree.calc_conductance_matrix
   CompartmentTree.calc_system_matrix
   CompartmentTree.calc_eigenvalues
   CompartmentTree.compute_gmc
   CompartmentTree.compute_g_channels
   CompartmentTree.compute_g_single_channel
   CompartmentTree.compute_concMechGamma
   CompartmentTree.compute_concMechTau
   CompartmentTree.compute_c
   CompartmentTree.reset_fit_data
   CompartmentTree.run_fit
   CompartmentTree.compute_fake_geometry
   CompartmentTree.plot_dendrogram

.. autoclass:: neat.CompartmentNode


Neural Evaluation Tree
======================

.. autoclass:: neat.NET

.. autosummary::
   :toctree: generated/

    NET.get_loc_idxs
    NET.get_leaf_loc_node
    NET.set_new_loc_idxs
    NET.get_reduced_tree
    NET.calc_total_impedance
    NET.calc_i_z
    NET.calc_i_z_matrix
    NET.calc_impedance_matrix
    NET.calc_impedance_matrix
    NET.calc_compartmentalization
    NET.plot_dendrogram


.. autoclass:: neat.NETNode


.. autoclass:: neat.Kernel

.. autosummary::
   :toctree: generated/

   Kernel.k_bar
   Kernel.t
   Kernel.ft


Simulate reduced compartmental models
======================================

.. autoclass:: neat.simulations.neuron.neuronmodel.NeuronCompartmentTree

.. autosummary::
   :toctree: generated/

.. autofunction:: neat.simulations.neuron.neuronmodel.createReducedNeuronModel

.. autosummary::
   :toctree: generated/


*******************
Morphological Trees
*******************


Morphology Tree
===============


.. autoclass:: neat.MorphTree

Read a morphology from an SWC file

.. autosummary::
   :toctree: generated/

   MorphTree.read_swc_tree_from_file
   MorphTree.determine_soma_type

.. autosummary::
   :toctree: generated/

   MorphTree.__getitem__
   MorphTree.__iter__

Get specific nodes or sets of nodes from the tree.

.. autosummary::
   :toctree: generated/

   MorphTree.root
   MorphTree.get_nodes
   MorphTree.nodes
   MorphTree.get_leafs
   MorphTree.leafs
   MorphTree.get_nodes_in_basal_subtree
   MorphTree.get_nodes_in_apical_subtree
   MorphTree.get_nodes_in_axonal_subtree
   MorphTree.convert_node_arg_to_nodes

Relating to the computational tree.

.. autosummary::
   :toctree: generated/

   MorphTree.setTreetype
   MorphTree.treetype
   MorphTree.read_swc_tree_from_file
   MorphTree.set_comp_tree
   MorphTree._evaluate_comp_criteria
   MorphTree.remove_comp_tree

Storing locations, interacting with stored locations and distributing
locations

.. autosummary::
   :toctree: generated/

   MorphTree.convert_loc_arg_to_locs
   MorphTree.store_locs
   MorphTree.add_loc
   MorphTree.clear_locs
   MorphTree.remove_locs
   MorphTree._try_name
   MorphTree.get_locs
   MorphTree.get_node_indices
   MorphTree.get_x_coords
   MorphTree.get_loc_idxs_on_node
   MorphTree.get_loc_idxs_on_nodes
   MorphTree.get_loc_idxs_on_path
   MorphTree.get_nearest_loc_idxs
   MorphTree.get_nearest_neighbour_loc_idxs
   MorphTree.get_leaf_loc_idxs
   MorphTree.distances_to_soma
   MorphTree.distances_to_bifurcation
   MorphTree.distribute_locs_on_nodes
   MorphTree.distribute_locs_uniform
   MorphTree.distribute_locs_random
   MorphTree.extend_with_bifurcation_locs
   MorphTree.unique_locs
   MorphTree.path_length

Plotting on a 1D axis.

.. autosummary::
   :toctree: generated/

   MorphTree.make_x_axis
   MorphTree.set_node_colors
   MorphTree.get_x_values
   MorphTree.plot_1d
   MorphTree.plot_true_d2s
   MorphTree.color_x_axis

Plotting the morphology in 2D.

.. autosummary::
   :toctree: generated/

   MorphTree.plot_2d_morphology
   MorphTree.plot_morphology_interactive

Creating new trees from the existing tree.

.. autosummary::
   :toctree: generated/

   MorphTree.create_new_tree
   MorphTree.create_compartment_tree
   MorphTree.__copy__


.. autoclass:: neat.MorphNode

.. autosummary::
   :toctree: generated/

    MorphNode.set_p3d
    MorphNode.child_nodes

.. autoclass:: neat.MorphLoc


Physiology Tree
===============

.. autoclass:: neat.PhysTree

.. autosummary::
   :toctree: generated/

   PhysTree.as_passive_membrane
   PhysTree.set_v_ep
   PhysTree.set_physiology
   PhysTree.set_leak_current
   PhysTree.add_channel_current
   PhysTree.get_channels_in_tree
   PhysTree.fit_leak_current
   PhysTree._evaluate_comp_criteria

.. autoclass:: neat.PhysNode


Separation of Variables Tree
============================

.. autoclass:: neat.SOVTree

.. autosummary::
   :toctree: generated/

   SOVTree.calc_sov_equations
   SOVTree.get_mode_importance
   SOVTree.get_important_modes
   SOVTree.calc_impedance_matrix
   SOVTree.construct_net
   SOVTree.compute_lin_terms

.. autoclass:: neat.SOVNode


Greens Tree
===========

.. autoclass:: neat.GreensTree

.. autosummary::
   :toctree: generated/

   GreensTree.remove_expansion_points
   GreensTree.set_impedance
   GreensTree.calc_zf
   GreensTree.calc_impedance_matrix

.. autoclass:: neat.GreensNode

.. autosummary::
   :toctree: generated/

    GreensNode.set_expansion_point


Simulate NEURON models
======================

.. autoclass:: neat.simulations.neuron.neuronmodel.NeuronSimTree

.. autosummary::
   :toctree: generated/

   NeuronSimTree.init_model
   NeuronSimTree.delete_model
   NeuronSimTree.add_shunt
   NeuronSimTree.add_double_exp_current
   NeuronSimTree.add_exp_synapse
   NeuronSimTree.add_double_exp_synapse
   NeuronSimTree.add_nmda_synapse
   NeuronSimTree.add_double_exp_nmda_synapse
   NeuronSimTree.add_i_clamp
   NeuronSimTree.add_sin_clamp
   NeuronSimTree.add_ou_clamp
   NeuronSimTree.add_ou_conductance
   NeuronSimTree.add_ou_reversal
   NeuronSimTree.add_v_clamp
   NeuronSimTree.set_spiketrain
   NeuronSimTree.run
   NeuronSimTree.calc_e_eq


Compute equilibrium potentials and concentrations
=================================================

.. autoclass:: neat.EquilibriumTree

.. autosummary::
   :toctree: generated/

   EquilibriumTree.calc_e_eq
   EquilibriumTree.set_e_eq


*************
Other Classes
*************

Fitting reduced models
======================

.. autoclass:: neat.CompartmentFitter

To implement the default methodology.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.set_ctree
   CompartmentFitter.fit_model

To check the faithfullness of the passive reduction, the following functions
implement vizualisation of impedance kernels.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.check_passive
   CompartmentFitter.get_kernels
   CompartmentFitter.plot_kernels

Individual fit functions.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.create_tree_gf
   CompartmentFitter.create_tree_sov
   CompartmentFitter.fit_leak_only
   CompartmentFitter.fit_passive
   CompartmentFitter.eval_channel
   CompartmentFitter.fit_channels
   CompartmentFitter.fit_concentration
   CompartmentFitter.fit_capacitance
   CompartmentFitter.fit_syn_rescale


Defining ion channels
=====================

.. autoclass:: neat.IonChannel

.. autosummary::
   :toctree: generated/

   IonChannel.set_default_params
   IonChannel.compute_p_open
   IonChannel.compute_derivatives
   IonChannel.compute_derivativesConc
   IonChannel.compute_varinf
   IonChannel.compute_tauinf
   IonChannel.compute_linear
   IonChannel.compute_linear_conc
   IonChannel.compute_lin_sum
   IonChannel.compute_lin_conc


Neural evaluation tree simulator
================================

.. autoclass:: neat.netsim.NETSim


Compute Fourrier transforms
===========================

.. autoclass:: neat.FourrierTools

.. autosummary::
   :toctree: generated/

   FourrierTools.__call__
   FourrierTools.ft
   FourrierTools.ftInv





