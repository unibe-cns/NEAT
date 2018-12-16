
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