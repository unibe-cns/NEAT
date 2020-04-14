Install
=======

Note: The following instructions are for Linux and Max OSX systems and only use
command line tools. Please follow the appropriate manuals for Windows systems or
tools with graphical interfaces.


Install using `setup.py` (requires `git <https://git-scm.com>`_):
::
    git clone https://github.com/unibe-cns/NEAT
    cd NEAT
    python setup.py install

Post-Install
============

To use NEAT with `NEURON <https://neuron.yale.edu/neuron/>`_, make sure NEURON
is properly installed with its Python interface, and compile and install the
default NEURON mechanisms by running
::
    compilechannels default

To test the installation (requires `pytest`)
::
    pytest

