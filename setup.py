#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

import re
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import os, subprocess, shutil, sys


def read_version():
    with open("./neat/__version__.py") as f:
        line = f.read()
        match = re.findall(r"[0-9]+\.[0-9]+\.[0-9]+", line)
        return match[0]


def read_requirements():
    with open('./requirements/requirements.txt') as fp:
        requirements = fp.read()
    return requirements


class DelayedIncludeDirs(list):
    """Delay importing of numpy until extension is built. This allows pip
    to install numpy if it's not available.

    """
    def __init__(self):
        super().__init__()
        # WARNING: for some reason we need to have a non-empty list otherwise
        # __iter__ is never called; this dummy value is never used!
        self.append('dummy value')

    def __iter__(self):
        import numpy
        return iter([numpy.get_include(), "neat/tools/simtools/net/*.h"])


ext = Extension(name="netsim",
                sources=["neat/tools/simtools/net/netsim.pyx",
                         "neat/tools/simtools/net/Ionchannels.cc",
                         "neat/tools/simtools/net/netsim.pyx",
                         "neat/tools/simtools/net/NETC.cc",
                         "neat/tools/simtools/net/Synapses.cc",
                         "neat/tools/simtools/net/Tools.cc"
                         ],
                language="c++",
                extra_compile_args=["-w", "-O3", "-std=gnu++11"],
                include_dirs=DelayedIncludeDirs(),
                )

s_ = setup(
    name='neatdend',
    version=read_version(),
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
        "neat.tools.simtools.neuron": ["mech_storage/*.mod"],
    },
    ext_package='neat',
    ext_modules=cythonize([ext], language_level=3),
    include_package_data=True,
    author='Willem Wybo, Jakob Jordan, Benjamin Ellenberger',
    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.7',
                 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],
    license='GPLv3',
    url='https://github.com/unibe-cns/NEAT',
    long_description=open('README.rst').read(),
    long_description_content_type="text/x-rst",
    install_requires=read_requirements(),
)
