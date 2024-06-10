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
   STree.checkOrdered
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
   STree.upBifurcationNode
   STree.downBifurcationNode
   STree.get_bifurcation_nodes
   STree.get_nearest_neighbours


.. autoclass:: neat.SNode


Compartment Tree
================

.. autoclass:: neat.CompartmentTree

.. autosummary::
   :toctree: generated/

   CompartmentTree.addCurrent
   CompartmentTree.setExpansionPoints
   CompartmentTree.setEEq
   CompartmentTree.getEEq
   CompartmentTree.fitEL
   CompartmentTree.getEquivalentLocs
   CompartmentTree.calcImpedanceMatrix
   CompartmentTree.calcConductanceMatrix
   CompartmentTree.calcSystemMatrix
   CompartmentTree.calcEigenvalues
   CompartmentTree.computeGMC
   CompartmentTree.computeGChanFromImpedance
   CompartmentTree.computeGSingleChanFromImpedance
   CompartmentTree.computeConcMechGamma
   CompartmentTree.computeConcMechTau
   CompartmentTree.computeC
   CompartmentTree.resetFitData
   CompartmentTree.runFit
   CompartmentTree.computeFakeGeometry
   CompartmentTree.plotDendrogram

.. autoclass:: neat.CompartmentNode


Neural Evaluation Tree
======================

.. autoclass:: neat.NET

.. autosummary::
   :toctree: generated/

    NET.getLocInds
    NET.getLeafLocNode
    NET.setNewLocInds
    NET.getReducedTree
    NET.calcTotalImpedance
    NET.calcIZ
    NET.calcIZMatrix
    NET.calcImpedanceMatrix
    NET.calcImpMat
    NET.getCompartmentalization
    NET.plotDendrogram


.. autoclass:: neat.NETNode


.. autoclass:: neat.Kernel

.. autosummary::
   :toctree: generated/

   Kernel.k_bar
   Kernel.t
   Kernel.ft


Simulate reduced compartmental models
======================================

.. autoclass:: neat.tools.simtools.neuron.neuronmodel.NeuronCompartmentTree

.. autosummary::
   :toctree: generated/

.. autofunction:: neat.tools.simtools.neuron.neuronmodel.createReducedNeuronModel

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

   PhysTree.asPassiveMembrane
   PhysTree.setVEP
   PhysTree.setPhysiology
   PhysTree.setLeakCurrent
   PhysTree.addCurrent
   PhysTree.getChannelsInTree
   PhysTree.fitLeakCurrent
   PhysTree._evaluate_comp_criteria

.. autoclass:: neat.PhysNode


Separation of Variables Tree
============================

.. autoclass:: neat.SOVTree

.. autosummary::
   :toctree: generated/

   SOVTree.calcSOVEquations
   SOVTree.getModeImportance
   SOVTree.getImportantModes
   SOVTree.calcImpedanceMatrix
   SOVTree.constructNET
   SOVTree.computeLinTerms

.. autoclass:: neat.SOVNode


Greens Tree
===========

.. autoclass:: neat.GreensTree

.. autosummary::
   :toctree: generated/

   GreensTree.removeExpansionPoints
   GreensTree.setImpedance
   GreensTree.calcZF
   GreensTree.calcImpedanceMatrix

.. autoclass:: neat.GreensNode

.. autosummary::
   :toctree: generated/

    GreensNode.setExpansionPoint


Simulate NEURON models
======================

.. autoclass:: neat.tools.simtools.neuron.neuronmodel.NeuronSimTree

.. autosummary::
   :toctree: generated/

   NeuronSimTree.init_model
   NeuronSimTree.deleteModel
   NeuronSimTree.addShunt
   NeuronSimTree.addDoubleExpCurrent
   NeuronSimTree.addExpSynapse
   NeuronSimTree.addDoubleExpSynapse
   NeuronSimTree.addNMDASynapse
   NeuronSimTree.addDoubleExpNMDASynapse
   NeuronSimTree.addIClamp
   NeuronSimTree.addSinClamp
   NeuronSimTree.addOUClamp
   NeuronSimTree.addOUconductance
   NeuronSimTree.addOUReversal
   NeuronSimTree.addVClamp
   NeuronSimTree.setSpikeTrain
   NeuronSimTree.run
   NeuronSimTree.calcEEq


Compute equilibrium potentials and concentrations
=================================================

.. autoclass:: neat.EquilibriumTree

.. autosummary::
   :toctree: generated/

   EquilibriumTree.calcEEq
   EquilibriumTree.setEEq


*************
Other Classes
*************

Fitting reduced models
======================

.. autoclass:: neat.CompartmentFitter

To implement the default methodology.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.setCTree
   CompartmentFitter.fitModel

To check the faithfullness of the passive reduction, the following functions
implement vizualisation of impedance kernels.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.checkPassive
   CompartmentFitter.getKernels
   CompartmentFitter.plotKernels

Individual fit functions.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.createTreeGF
   CompartmentFitter.createTreeSOV
   CompartmentFitter.fitPassiveLeak
   CompartmentFitter.fitPassive
   CompartmentFitter.evalChannel
   CompartmentFitter.fitChannels
   CompartmentFitter.fitConcentration
   CompartmentFitter.fitCapacitance
   CompartmentFitter.fitSynRescale


Defining ion channels
=====================

.. autoclass:: neat.IonChannel

.. autosummary::
   :toctree: generated/

   IonChannel.setDefaultParams
   IonChannel.computePOpen
   IonChannel.computeDerivatives
   IonChannel.computeDerivativesConc
   IonChannel.computeVarinf
   IonChannel.computeTauinf
   IonChannel.computeLinear
   IonChannel.computeLinearConc
   IonChannel.computeLinSum
   IonChannel.computeLinConc


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





