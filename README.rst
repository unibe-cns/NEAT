NEAT (NEural Analysis Toolkit)
==============================

Introduction
------------

NEAT is a python library that allows for the study of neuronal integration in morphological neuron models using (semi-) analytical techniques. For now, NEAT accepts morphologies in the *de facto* standard .swc format [#f1]_.

NEAT implements:

* The separation of variables method to compute the Green's function associated with a morphology [#f2]_, [#f3]_, [#f4]_, [#f5]_.
* Green's function calculation with linearized ion channels [#f6]_.
* The neural evaluation tree framework [#f7]_.
* Various tools to plot, analyze and interact with neuronal morphologies.
* A c++ simulator that simulates the NET framework.

Installation
------------

Note: The following instructions are for Linux and Max OSX systems and only use command line tools. Please follow the appropriate manuals for Windows systems or tools with graphical interfaces.

Check out the git repository and install using :code:`setup.py`
::
    git clone https://github.com/WillemWybo/NEAT
    cd NEAT
    python setup.py install

To test the installation (requires :code:`pytest`)
::
    sh run_tests.sh


References

.. [#f1] Cannon et al. *An online archive of reconstructed hippocampal neurons.*, J. Neurosci. methods.
.. [#f2] Major, G., Evans, J. D. and Jack, J. B. (1993) *Solutions for transients in arbitrarily branching cables: I. Voltage recording with a somatic shunt.*, Biophysical journal, 65(1), pp. 423–49.
.. [#f3] Major, G., Evans, J. D. and Jack, J. B. (1993) *Solutions for transients in arbitrarily branching cables II. Voltage clamp theory*, Biophysical journal, 65(1), pp. 469–491.
.. [#f4] Major, G. (1993) *Solutions for transients in arbitrarily branching cables: III. Voltage clamp problems.*, Biophysical journal, 65(1), pp. 469–491.
.. [#f5] Major, G. and Evans, J. D. (1994) *Solutions for transients in arbitrarily branching cables: IV. Nonuniform electrical parameters.*, Biophysical journal, 66(3 Pt 1), pp. 615–33.
.. [#f6] Koch, C. and Poggio, T. (1985) *A simple algorithm for solving the cable equation in dendritic trees of arbitrary geometry.*, Journal of neuroscience methods, 12(4), pp. 303–315.
.. [#f7] Wybo, W. A. M., Torben-nielsen, B. and Gewaltig, M. (2018) *Dynamic compartmentalization in neurons enables branch-specific learning.*, bioRxiv, 10.1101/24. doi: 10.1101/244772.