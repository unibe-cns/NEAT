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

Installation
------------

Note: The following instructions are for Linux and Max OSX systems and only use
command line tools. Please follow the appropriate manuals for Windows systems or
tools with graphical interfaces.

Install with pip:
::
    pip install neat-dend

Install using `setup.py` (requires `git <https://git-scm.com>`_):
::
    git clone https://github.com/unibe-cns/NEAT
    cd NEAT
    python setup.py install

To test the installation (requires `pytest`)
::
    sh run_tests.sh

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