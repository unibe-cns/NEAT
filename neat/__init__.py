import warnings

# This is a hack to allow running headless e.g. Jenkins
import os
if not os.environ.get('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')

from neat.trees.stree import STree
from neat.trees.stree import SNode

from neat.trees.morphtree import MorphTree
from neat.trees.morphtree import MorphNode
from neat.trees.morphtree import MorphLoc

from neat.trees.phystree import PhysTree
from neat.trees.phystree import PhysNode

from neat.trees.sovtree import SOVTree
from neat.trees.sovtree import SOVNode

from neat.trees.greenstree import GreensTree
from neat.trees.greenstree import GreensNode

from neat.trees.netree import NET
from neat.trees.netree import NETNode
from neat.trees.netree import Kernel

from neat.trees.compartmenttree import CompartmentTree
from neat.trees.compartmenttree import CompartmentNode

try:
    from neat.tools.simtools.neuron.neuronmodel import NeuronSimTree
    from neat.tools.simtools.neuron.neuronmodel import NeuronSimNode
    from neat.tools.simtools.neuron.neuronmodel import NeuronCompartmentTree
except ModuleNotFoundError:
    warnings.warn('NEURON not available', UserWarning)

from neat.channels.ionchannels import IonChannel