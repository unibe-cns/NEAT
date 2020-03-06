**********
Trees
**********

.. automodule:: neat.trees
.. currentmodule:: neat

Abstract Tree
=============

.. autoclass:: neat.STree

.. autosummary::
   :toctree: generated/

   STree.__init__
   STree._findNode
   STree.__len__
   STree.checkOrdered
   STree.getNodes
   STree.gatherNodes
   STree._gatherNodes
   
   
.. note:: This is a subset of the methods of STree, which I chose to show how we can document public as well
          private methods of STree while leaving the others in the dark.

.. autoclass:: neat.SNode


Morphological Tree
==================


.. autoclass:: neat.MorphTree

.. autoclass:: neat.MorphNode

.. autoclass:: neat.MorphLoc

Phys Tree
=========

.. autoclass:: neat.PhysTree

.. autoclass:: neat.PhysNode


Separation of Variables Tree
============================

.. autoclass:: neat.SOVTree

.. autoclass:: neat.SOVNode


Greens Tree
============

.. autoclass:: neat.GreensTree

.. autoclass:: neat.GreensNode


Compartment Tree
================

.. autoclass:: neat.CompartmentTree

.. autoclass:: neat.CompartmentNode


Neural Evaluation Tree
======================

.. autoclass:: neat.NET

.. autoclass:: neat.NETNode

