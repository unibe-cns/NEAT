#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

import os
import codecs
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def read_version():
    for line in read("neat/__version__.py").splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


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
        return iter([numpy.get_include(), "neat/simulations/net/*.h"])


ext = Extension(name="netsim",
                sources=["neat/simulations/net/netsim.pyx",
                         "neat/simulations/net/Ionchannels.cc",
                         "neat/simulations/net/netsim.pyx",
                         "neat/simulations/net/NETC.cc",
                         "neat/simulations/net/Synapses.cc",
                         "neat/simulations/net/Tools.cc"
                         ],
                language="c++",
                extra_compile_args=["-w", "-O3", "-std=gnu++11"],
                include_dirs=DelayedIncludeDirs(),
                )

s_ = setup(
    name='neatdend',
    version=read_version(),
    scripts=['neat/actions/neatmodels'],
    packages=['neat',
              'neat.trees',
              'neat.actions',
              'neat.tools',
              'neat.tools.fittools',
              'neat.tools.plottools',
              'neat.simulations.neuron',
              'neat.simulations.nest',
              'neat.modelreduction',
              'neat.channels',
              'neat.channels.channelcollection'],
    package_data={
        "neat.simulations.neuron": ["mech_storage/*.mod"],
        "neat.simulations.nest": ["default_syns.nestml"]
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
