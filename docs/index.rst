.. _contents:

Overview of NEAT_
=================

.. _NEAT: https://github.com/unibe-cns/NEAT

NEAT is a python library for the study, simulation and simplification of
morphological neuron models. NEAT accepts morphologies in the *de facto*
standard .swc format [Cannon1998]_, and implements high-level tools to interact
with and analyze the morphologies.

NEAT also allows for the convenient definition of morphological neuron models.
These models can be simulated, through an interface with the NEURON simulator
[Carnevale2004]_, or can be analyzed with two classical methods: *(i)* the
separation of variables method [Major1993]_ to obtain impedance kernels as
a superposition of exponentials and *(ii)* Koch's method to compute impedances
with linearized ion channels analytically in the frequency domain [Koch1985]_.
Furthermore, NEAT implements the neural evaluation tree framework [Wybo2019]_
and an associated C++ simulator, to analyze subunit independence.

Finally, NEAT implements a new and powerful method to simplify morphological
neuron models into compartmental models with few compartments [Wybo2020]_. For
these models, NEAT also provides a NEURON interface so that they can be
simulated directly, and will soon also provide a NEST interface [Gewaltig2007]_.

Audience
--------

NEAT is of interest to neuroscientist who aim to understand dendritic
computation, and to explore dendritic computation at the network level.

Python
------

Python is a powerful programming language that allows simple and flexible
representations neural morphologies. Python has a vibrant and growing ecosystem
of packages that NEAT uses to provide more features such as numerical linear
algebra and drawing. In order to make the most out of NEAT you will want to know
how to write basic programs in Python. Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_ and the text
by Alex Martelli [Martelli03]_.

Free software
-------------

NEAT is free software; you can redistribute it and/or modify it under the
terms of the :doc:`GNU General Public License`.  We welcome contributions.
Join us on `GitHub <https://github.com/unibe-cns/NEAT>`_.

History
-------

NEAT was born in April 2018. The original version was designed and written by
Willem Wybo, based on code by Benjamin Torben-Nielsen. With help of Jakob
Jordan and Benjamin Ellenberger, NEAT became an installable python package with
documentation website.

Contributors are listed in :doc:`credits. <credits>`

Documentation
-------------

.. only:: html

    :Release: |version|
    :Date: |today|

.. toctree::
   :maxdepth: 1

   install
   tutorial
   reference/index
   developer/index
   news
   license
   credits
   citing
   bibliography
   auto_examples/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`
