#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

import re
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

import os, subprocess, shutil, sys


def read_version():
    with open("./neat/__version__.py") as f:
        line = f.read()
        match = re.findall(r"[0-9]+\.[0-9]+\.[0-9]+", line)
        return match[0]


class DevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # develop install
        develop.run(self)


class InstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # regular install
        install.run(self)


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


class DelayedIncludeDirs:
    """Delay importing of numpy until extension is built. This allows pip
    to install numpy if it's not available.

    """
    def __iter__(self):
        import numpy
        return iter([numpy.get_include(), "neat/tools/simtools/net/*.h"])


ext = Extension("netsim",
                ["neat/tools/simtools/net/netsim.pyx",
                 "neat/tools/simtools/net/NETC.cc",
                 "neat/tools/simtools/net/Synapses.cc",
                 "neat/tools/simtools/net/Ionchannels.cc",
                 "neat/tools/simtools/net/Tools.cc"],
                language="c++",
                extra_compile_args=["-w", "-O3", "-std=gnu++11"],
                include_dirs=DelayedIncludeDirs())
ext.cython_directives = {'language_level': "3"}

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
    ext_modules=[ext],
    cmdclass={
        'develop': DevelopCommand,
        'install': InstallCommand,
        'build_ext': BuildExtCommand,
    },
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
