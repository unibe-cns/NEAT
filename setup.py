#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

from __version__ import version as pversion

import numpy

dependencies = ['numpy>=1.14.1',
                'matplotlib>=2.1.2',
                'cython>=0.27.3',
                'scipy>=1.0.0',
                'sympy>=1.1.1',
                'sklearn>=0.19.1']

ext = Extension("netsim",
                ["neat/tools/simtools/netsim.pyx",
                 "neat/tools/simtools/NETC.cc",
                 "neat/tools/simtools/Synapses.cc",
                 "neat/tools/simtools/Ionchannels.cc",
                 "neat/tools/simtools/Tools.cc"],
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

setup(
    name='neat',
    version=pversion,
    packages=['neat',
              'neat.trees',
              'neat.tools',
              'neat.tools.fittools',
              'neat.tools.plottools',
              'neat.channels'],
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
