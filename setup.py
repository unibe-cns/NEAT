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
Define pre- and post-install commands via command classes.
From https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
"""

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # execute pre installation commands
        write_ionchannel_header_and_cpp_file()
        copy_default_neuron_mechanisms()
        # develop install
        develop.run(self)
        # execute post installation commands
        compile_default_ion_channels()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # execute pre installation commands
        write_ionchannel_header_and_cpp_file()
        copy_default_neuron_mechanisms()
        # regular install
        install.run(self)
        # execute post installation commands
        compile_default_ion_channels()


def write_ionchannel_header_and_cpp_file():
    """
    Writes Ionchannel.h and Ionchannel.cpp files that are required for
    the netsim.pyx extension.

    """
    # we can import even before installation as we are in the root
    # directory
    import neat.channels.writecppcode as writecppcode
    writecppcode.write()


def copy_default_neuron_mechanisms():
    """
    Copies the default neuron mechanisms from
        neat/tools/simtools/neuron/mech_storage/
    to
        neat/tools/simtools/neuron/mech/
    and creates an `__init__.py` file in that directory.

    The mech/ directory is generated in a clean state by the installer and
    aggregates all neuron .mod files. Hence, it should never be used for
    permanent storage.

    """
    mech_path = 'neat/tools/simtools/neuron/mech/'
    if os.path.exists(mech_path):
        shutil.rmtree(mech_path)
    shutil.copytree('neat/tools/simtools/neuron/mech_storage/', mech_path)
    f = open('neat/tools/simtools/neuron/mech/__init__.py', 'wb')
    f.close()


def compile_default_ion_channels():
    """
    Compiles the default ion channels found in channelcollection for
    use with NEURON.
    """
    cwd = os.getcwd()
    os.chdir('neat/channels/')
    subprocess.call(['compilechannels', 'channelcollection/'])
    os.chdir(cwd)


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
              'neat.tools.simtools.neuron.mech',
              'neat.channels',
              'neat.channels.channelcollection'],
    ext_package='neat',
    ext_modules=[ext],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    include_package_data=True,
    package_data={'': ['*.mod']},
    author='Willem Wybo',
    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 3.7'],
    install_requires=read_requirements(),
)
