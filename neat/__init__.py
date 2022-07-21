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

from .trees.greenstree import GreensTree
from .trees.greenstree import GreensNode, SomaGreensNode

from .trees.netree import NET
from .trees.netree import NETNode
from .trees.netree import Kernel

from .trees.compartmenttree import CompartmentTree
from .trees.compartmenttree import CompartmentNode

try:
    from .tools.simtools.neuron.neuronmodel import loadNeuron
    from .tools.simtools.neuron.neuronmodel import NeuronSimTree
    from .tools.simtools.neuron.neuronmodel import NeuronSimNode
    from .tools.simtools.neuron.neuronmodel import NeuronCompartmentTree
    from .tools.simtools.neuron.neuronmodel import createReducedNeuronModel

except ModuleNotFoundError:
    warnings.warn('NEURON not available', UserWarning)

try:
    from .tools.simtools.nest.nestmodel import createNestModel
except ModuleNotFoundError:
    warnings.warn('NEST not available', UserWarning)

from .tools.kernelextraction import FourrierTools

from .channels.ionchannels import IonChannel

from .tools.fittools.compartmentfitter import CompartmentFitter

from .__version__ import __version__
