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
   STree.checkOrdered
   STree.getNodes
   STree.nodes
   STree.gatherNodes
   STree.getLeafs
   STree.leafs
   STree.isLeaf
   STree.root
   STree.isRoot
   STree.addNodeWithParentFromIndex
   STree.addNodeWithParent
   STree.softRemoveNode
   STree.removeNode
   STree.removeSingleNode
   STree.insertNode
   STree.resetIndices
   STree.getSubTree
   STree.depthOfNode
   STree.degreeOfNode
   STree.orderOfNode
   STree.pathToRoot
   STree.pathBetweenNodes
   STree.pathBetweenNodesDepthFirst
   STree.getNodesInSubtree
   STree.sisterLeafs
   STree.upBifurcationNode
   STree.downBifurcationNode
   STree.getBifurcationNodes
   STree.getNearestNeighbours
   STree.__copy__


.. note:: This is a subset of the methods of STree, which I chose to show how we can document public as well
          private methods of STree while leaving the others in the dark.

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

   MorphTree.readSWCTreeFromFile
   MorphTree.determineSomaType

.. autosummary::
   :toctree: generated/

   MorphTree.__getitem__
   MorphTree.__iter__

Get specific nodes or sets of nodes from the tree.

.. autosummary::
   :toctree: generated/

   MorphTree.root
   MorphTree.getNodes
   MorphTree.nodes
   MorphTree.getLeafs
   MorphTree.leafs
   MorphTree.getNodesInBasalSubtree
   MorphTree.getNodesInApicalSubtree
   MorphTree.getNodesInAxonalSubtree
   MorphTree._convertNodeArgToNodes

Relating to the computational tree.

.. autosummary::
   :toctree: generated/

   MorphTree.setTreetype
   MorphTree.treetype
   MorphTree.readSWCTreeFromFile
   MorphTree.setCompTree
   MorphTree._evaluateCompCriteria
   MorphTree.removeCompTree

Storing locations, interacting with stored locations and distributing
locations

.. autosummary::
   :toctree: generated/

   MorphTree._convertLocArgToLocs
   MorphTree.storeLocs
   MorphTree.addLoc
   MorphTree.clearLocs
   MorphTree.removeLocs
   MorphTree._tryName
   MorphTree.getLocs
   MorphTree.getNodeIndices
   MorphTree.getXCoords
   MorphTree.getLocindsOnNode
   MorphTree.getLocindsOnNodes
   MorphTree.getLocindsOnPath
   MorphTree.getNearestLocinds
   MorphTree.getNearestNeighbourLocinds
   MorphTree.getLeafLocinds
   MorphTree.distancesToSoma
   MorphTree.distancesToBifurcation
   MorphTree.distributeLocsOnNodes
   MorphTree.distributeLocsUniform
   MorphTree.distributeLocsRandom
   MorphTree.extendWithBifurcationLocs
   MorphTree.uniqueLocs
   MorphTree.pathLength

Plotting on a 1D axis.

.. autosummary::
   :toctree: generated/

   MorphTree.makeXAxis
   MorphTree.setNodeColors
   MorphTree.getXValues
   MorphTree.plot1D
   MorphTree.plotTrueD2S
   MorphTree.colorXAxis

Plotting the morphology in 2D.

.. autosummary::
   :toctree: generated/

   MorphTree.plot2DMorphology
   MorphTree.plotMorphologyInteractive

Creating new trees from the existing tree.

.. autosummary::
   :toctree: generated/

   MorphTree.createNewTree
   MorphTree.createCompartmentTree
   MorphTree.__copy__


.. autoclass:: neat.MorphNode

.. autosummary::
   :toctree: generated/

    MorphNode.setP3D
    MorphNode.child_nodes

.. autoclass:: neat.MorphLoc


Physiology Tree
===============

.. autoclass:: neat.PhysTree

.. autosummary::
   :toctree: generated/

   PhysTree.asPassiveMembrane
   PhysTree.setEEq
   PhysTree.setPhysiology
   PhysTree.setLeakCurrent
   PhysTree.addCurrent
   PhysTree.getChannelsInTree
   PhysTree.fitLeakCurrent
   PhysTree._evaluateCompCriteria

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

   NeuronSimTree.initModel
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

Individual fit functions.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.createTreeGF
   CompartmentFitter.createTreeSOV
   CompartmentFitter.fitPassiveLeak
   CompartmentFitter.fitPassive
   CompartmentFitter.evalChannel
   CompartmentFitter.fitChannels
   CompartmentFitter.fitCapacitance
   CompartmentFitter.setEEq
   CompartmentFitter.getEEq
   CompartmentFitter.fitEEq
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





