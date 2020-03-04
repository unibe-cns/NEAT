#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.develop import develop
from setuptools.command.install import install

import os, subprocess, shutil, sys

import numpy

from __version__ import version as pversion


"""
Define post install commands via command classes.
From https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
"""

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # develop install
        develop.run(self)
        # execute post installation commands
        compile_default_ion_channels()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # regular install
        install.run(self)
        # execute post installation commands
        compile_default_ion_channels()


def compile_default_ion_channels():
    subprocess.call(['python', 'neat/channels/compilechannels.py', 'neat/channels/channelcollection/'])


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
    ext_package='neat',
    ext_modules=[ext],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    include_package_data=True,
    author='Willem Wybo',
    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.7'],
    install_requires=dependencies,
)
