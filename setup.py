#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

from distutils.core import setup
from distutils.extension import Extension

import os, subprocess, shutil, sys

import numpy
from Cython.Distutils import build_ext

from __version__ import version as pversion

dependencies = ['numpy>=1.14.1',
                'matplotlib>=2.1.2',
                'cython>=0.27.3',
                'scipy>=1.0.0',
                'sympy>=1.1.1',
                'scikit-learn>=0.19.1']

ext = Extension("netsim",
                ["neat/tools/simtools/net/netsim.pyx",
                 "neat/tools/simtools/net/NETC.cc",
                 "neat/tools/simtools/net/Synapses.cc",
                 "neat/tools/simtools/net/Ionchannels.cc",
                 "neat/tools/simtools/net/Tools.cc"],
                language="c++",
                extra_compile_args=["-w", "-O3", "-std=gnu++11"],
                include_dirs=[numpy.get_include()])

# setup(
#     name='neat',
#     version=pversion,
#     packages=['neat',
#               'neat.trees',
#               'neat.tools',
#               'neat.tools.fittools',
#               'neat.tools.plottools',
#               'neat.channels'],
#     ext_modules=[ext],
#     cmdclass={'build_ext': build_ext},
#     include_package_data=True,
#     author='Willem Wybo',
#     classifiers=['Development Status :: 3 - Alpha',
#                  'Programming Language :: Python :: 3.7'],
#     install_requires=dependencies,
# )

s_ = setup(
    name='neat',
    version=pversion,
    packages=['neat',
              'neat.trees',
              'neat.tools',
              'neat.tools.fittools',
              'neat.tools.plottools',
              'neat.tools.simtools.neuron',
              'neat.channels',
              'neat.channels.channelcollection'],
    # package_dir={'neat': 'neat/tools'},
    # package_data={'neat': ['neat/tools/simtools/neuron/mech/*.mod']},
    ext_package='neat',
    # ext_modules=cythonize([ext]),
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext},
    include_package_data=True,
    author='Willem Wybo',
    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.7'],
    install_requires=dependencies,
)

# set paths required for installation of neat/channels/compilechannels.py script
installation_path = s_.command_obj['install'].install_lib
current_path = os.path.dirname(sys.argv[0])
channel_path = os.path.join(installation_path, 'neat/channels/')
compile_file = os.path.join(channel_path, "compilechannels.py")

# copy standard mechanism storage to installation path
mech_path = os.path.join(installation_path, 'neat/tools/simtools/neuron/mech')
storage_path = os.path.join(current_path, 'neat/tools/simtools/neuron/mech_storage')
subprocess.call(["cp", "-rf", storage_path, mech_path])

# install the script, remove symlink if it already exists
subprocess.call(["chmod", "+x", compile_file])
link_name = "/usr/local/bin/compilechannels"
try:
    os.symlink(compile_file, link_name)
except FileExistsError:
    os.remove(link_name)
    os.symlink(compile_file, link_name)
# install the default ion channels
os.chdir(channel_path)
subprocess.call(["compilechannels",
                 "channelcollection/"])




