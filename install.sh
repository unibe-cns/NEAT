#!/bin/bash

# create mech directory if it doesn't exist and make sure it is empy
mkdir -p neat/tools/simtools/neuron/mech
rm neat/tools/simtools/neuron/mech/*
# write the IonChannel mod files and copy other input mod files for NEURON
python3 -m neat.channels.writemodfiles
cp -R neat/tools/simtools/neuron/mech_storage/ neat/tools/simtools/neuron/mech/
# write the IonChannel cpp code for netsim
python3 -m neat.channels.writecppcode
# run install script
python3 setup.py install --prefix="~/.local"
# copy ion channel mech to installation directory file
cp -R neat/tools/simtools/neuron/mech ~/.local/lib/python3.7/site-packages/neat/tools/simtools/neuron/
# compile the ion channels to neuron
cd ~/.local/lib/python3.7/site-packages/neat/tools/simtools/neuron
rm -r x86_64/
nrnivmodl mech/
