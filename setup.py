#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NEAT (NEural Analysis Toolkit)

Author: W. Wybo
"""

import numpy
from setuptools import setup, extension
from Cython.Build import cythonize

ext = extension.Extension(
    name="netsim",
    sources=["src/neat/simulations/net/netsim.pyx",
                "src/neat/simulations/net/Ionchannels.cc",
                "src/neat/simulations/net/netsim.pyx",
                "src/neat/simulations/net/NETC.cc",
                "src/neat/simulations/net/Synapses.cc",
                "src/neat/simulations/net/Tools.cc"
                ],
    language="c++",
    extra_compile_args=["-w", "-O3", "-std=gnu++11"],
    include_dirs=[numpy.get_include()],
)

s_ = setup(
    ext_package='neat',
    ext_modules=cythonize([ext], language_level=3),
)
