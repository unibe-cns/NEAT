#!/bin/bash

# write the IonChannel cpp code for netsim before calling native python installer
python3 -m neat.channels.writecppcode
python3 setup.py install --prefix="~/.local"