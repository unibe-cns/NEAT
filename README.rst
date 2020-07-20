NEAT (NEural Analysis Toolkit)
==============================

Introduction
------------

NEAT is a python library for the study, simulation and simplification of
morphological neuron models. NEAT accepts morphologies in the *de facto*
standard .swc format [Cannon1998]_, and implements high-level tools to interact
with and analyze the morphologies.

NEAT also allows for the convenient definition of morphological neuron models.
These models can be simulated, through an interface with the NEURON simulator
[Carnevale2004]_, or can be analyzed with two classical methods: *(i)* the
separation of variables method [Major1993]_ to obtain impedance kernels as
a superposition of exponentials and *(ii)* Koch's method to compute impedances
with linearized ion channels analytically in the frequency domain [Koch1985]_.
Furthermore, NEAT implements the neural evaluation tree framework [Wybo2019]_
and an associated C++ simulator, to analyze subunit independence.

Finally, NEAT implements a new and powerful method to simplify morphological
neuron models into compartmental models with few compartments [Wybo2020]_. For
these models, NEAT also provides a NEURON interface so that they can be
simulated directly, and will soon also provide a NEST interface [Gewaltig2007]_.

Documentation
-------------

Documentation is available `here <https://neatdend.readthedocs.io>`_

Installation
------------

**Install**

Note: The following instructions are for Linux and Max OSX systems and only use
command line tools. Please follow the appropriate manuals for Windows systems or
tools with graphical interfaces.

You can install the latest release via pip:

   .. code-block:: shell

      pip install neatdend

The adventurous can install the most recent development version directly from our master branch (don't use this in production unless there are good reasons!):

.. code-block:: shell

   git clone git@github.com:unibe-cns/NEAT.git
   cd NEAT
   pip install .

**Post-Install**

To use NEAT with `NEURON <https://neuron.yale.edu/neuron/>`_, make sure NEURON
is properly installed with its Python interface, and compile and install the
default NEURON mechanisms by running

.. code-block:: shell

    compilechannels default

Test the installation

.. code-block:: shell

    pytest

References
----------

.. [Cannon1998] Cannon et al. (1998) *An online archive of reconstructed hippocampal neurons*, J. Neurosci. methods.
.. [Carnevale2004] Carnevale, Nicholas T. and Hines, Michael L. (2004) *The NEURON book*
.. [Koch1985] Koch, C. and Poggio, T. (1985) *A simple algorithm for solving the cable equation in dendritic trees of arbitrary geometry*, Journal of neuroscience methods, 12(4), pp. 303–315.
.. [Major1993] Major et al. (1993) *Solutions for transients in arbitrarily branching cables: I. Voltage recording with a somatic shunt*, Biophysical journal, 65(1), pp. 423–49.
.. [Martelli03] A. Martelli (2003) *Python in a Nutshell*, O’Reilly Media Inc.
.. [Wybo2019] Wybo, Willem A.M. et al. (2019) *Electrical Compartmentalization in Neurons*, Cell Reports, 26(7), pp. 1759--1773 shunt.*, Biophysical journal, 65(1), pp. 423–49.
.. [Wybo2020] Wybo, Willem A.M. et al. (2020) TBA.
.. [Gewaltig2007] Gewaltig, Marc-Oliver and Diesmann, Markus. (2007) *NEST (NEural Simulation Tool)*, Scholarpedia, 2(4), pp. 1430
