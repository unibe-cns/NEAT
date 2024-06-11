# This is a hack to allow running headless e.g. Jenkins
import os
import warnings

havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    havedisplay = (exitval == 0)

if not havedisplay:
    import matplotlib
    matplotlib.use('Agg')

from .trees.stree import STree
from .trees.stree import SNode

from .trees.morphtree import MorphTree
from .trees.morphtree import MorphNode
from .trees.morphtree import MorphLoc

from .trees.phystree import PhysTree
from .trees.phystree import PhysNode

from .trees.sovtree import SOVTree
from .trees.sovtree import SOVNode, SomaSOVNode

from .trees.greenstree import GreensTree, GreensTreeTime
from .trees.greenstree import GreensNode, SomaGreensNode

from .trees.netree import NET
from .trees.netree import NETNode
from .trees.netree import Kernel

from .trees.compartmenttree import CompartmentTree
from .trees.compartmenttree import CompartmentNode

try:
    from .simulations.neuron.neuronmodel import load_neuron_model
    from .simulations.neuron.neuronmodel import NeuronSimTree
    from .simulations.neuron.neuronmodel import NeuronSimNode
    from .simulations.neuron.neuronmodel import NeuronCompartmentTree
except ModuleNotFoundError:
    warnings.warn('NEURON not available', UserWarning)

try:
    from .simulations.nest.nestmodel import NestCompartmentTree
    from .simulations.nest.nestmodel import NestCompartmentNode
    from .simulations.nest.nestmodel import load_nest_model
except ModuleNotFoundError:
    warnings.warn('NEST not available', UserWarning)

from .tools.kernelextraction import FourrierTools

from .channels.ionchannels import IonChannel
from .channels.concmechs import ExpConcMech

from .modelreduction.compartmentfitter import CompartmentFitter
from .modelreduction.cachetrees import EquilibriumTree

from .__version__ import __version__
