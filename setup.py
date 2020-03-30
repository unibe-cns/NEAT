#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

import os, subprocess, shutil, sys

import numpy

from __version__ import version as pversion


class BuildExtCommand(build_ext):

    def run(self):
        # execute pre build ext commands
        write_ionchannel_header_and_cpp_file()
        # build ext
        build_ext.run(self)
        # execute post build ext commands


def write_ionchannel_header_and_cpp_file():
    """
    Writes Ionchannel.h and Ionchannel.cpp files that are required for
    the netsim.pyx extension.

    """
    # we can import before installation as we are in the root
    # directory
    import neat.channels.writecppcode as writecppcode
    writecppcode.write()


def read_requirements():
    with open('./requirements.txt') as fp:
        requirements = fp.read()
    return requirements


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
    scripts=['neat/channels/compilechannels'],
    packages=['neat',
              'neat.trees',
              'neat.tools',
              'neat.tools.fittools',
              'neat.tools.plottools',
              'neat.tools.simtools.neuron',
              'neat.channels',
              'neat.channels.channelcollection'],
    package_data={
        "neat.tools.simtools.neuron": ["mech/*.mod"],
    },
    ext_package='neat',
    ext_modules=[ext],
    cmdclass={
        'build_ext': BuildExtCommand,
    },
    author='Willem Wybo',
    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.7'],
    install_requires=read_requirements(),
)
