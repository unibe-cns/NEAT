Installation
============

Install
-------

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

Post-Install
------------

To use NEAT with `NEURON <https://neuron.yale.edu/neuron/>`_, make sure NEURON
is properly installed with its Python interface, and compile and install the
default NEURON mechanisms by running
::

    compilechannels default

To test the installation (requires `pytest`)
::

    pytest

