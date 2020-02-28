#!/usr/bin/env python3
import os, glob, sys, inspect, shutil, subprocess

import neat
from neat import IonChannel

path_to_dir = sys.argv[1]
try:
    path_neat = sys.argv[2]
except IndexError:
    path_neat = neat.__path__[0]
path_for_mod_files = os.path.join(path_neat, 'tools/simtools/neuron/mech/')
path_for_compilation = os.path.join(path_neat, 'tools/simtools/neuron/')
print('--- writing channels from \n' + path_to_dir + ' to \n' + path_for_mod_files)


def allBaseClasses(cls):
    """
    Return list get all base classes from a given class
    """
    return [cls.__base__] + allBaseClasses(cls.__base__) if cls is not None else []


for channel_module in glob.glob(path_to_dir + '*.py'):
    # import channel modules
    # convert names from glob to something susceptible to python import
    channel_module = channel_module.replace('/', '.')
    channel_module = channel_module.replace('.py', '')
    print('Reading channels from:', channel_module)
    exec('import ' + channel_module + ' as chans')

    for name, obj in inspect.getmembers(chans):
        # if an object is a class and inheriting from IonChannel, write its mod-file
        if inspect.isclass(obj) and IonChannel in allBaseClasses(obj):
            chan = obj()
            print(' - write .mod file for:', chan.__class__.__name__)
            chan.writeModFile(path_for_mod_files)


# change to directory where 'mech/' folder is located and compile the mechanisms
os.chdir(path_for_compilation)
subprocess.call(["rm", "-r", "x86_64/"])
subprocess.call(["nrnivmodl", "mech/"])



