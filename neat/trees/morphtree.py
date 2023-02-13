"""
File contains:

    - `neat.MorphLoc`
    - `neat.MorphNode`
    - `neat.MorphTree`

Authors: B. Torben-Nielsen (legacy code) and W. Wybo
"""

import numpy as np

import matplotlib.patheffects as patheffects
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
import copy
from functools import reduce

from .stree import SNode, STree
from .compartmenttree import CompartmentNode, CompartmentTree


def originalTreetypeDecorator(fun):
    """
    Decorator that provides the safety that the treetype is set to
    'original' inside the functions it decorates
    """
    # wrapper to access self
    def wrapped(self, *args, **kwargs):
        current_treetype = self.treetype
        self.treetype = 'original'
        res = fun(self, *args, **kwargs)
        self.treetype = current_treetype
        return res
    wrapped.__doc__ = fun.__doc__
    return wrapped

def computationalTreetypeDecorator(fun):
    """
    Decorator that provides the safety that the treetype is set to
    'computational' inside the functions it decorates. This decorator also
    checks if a computational tree has been defined.

    Raises
    ------
        AttributeError
            If this function is called and no computational tree has been
            defined
    """
    # wrapper to access self
    def wrapped(self, *args, **kwargs):
        if self._computational_root is None:
            raise AttributeError('No computational tree has been defined, ' + \
                                  'and this function requires one. Use ' + \
                                  '`MorphTree.setCompTree()` or its ' + \
                                  'overwritten version in one of the derived' + \
                                  'classes')
        current_treetype = self.treetype
        self.treetype = 'computational'
        res = fun(self, *args, **kwargs)
        self.treetype = current_treetype
        return res
    wrapped.__doc__ = fun.__doc__
    return wrapped


class MorphLoc(object):
    """
    Stores a location on the morphology. The location is initialized starting
    from a node and x-value on the real morphology. The location is also be
    stored in the coordinates of the computational morphology. To toggle between
    coordinates, the class stores a reference to the morphology tree on which
    the location is defined, and returns either the original coordinate or the
    coordinate on the computational tree, depending on which tree is active.

    Initialized based on either a tuple or a dict where one entry specifies the
    node index and the other entry the x-coordinate specifying the location
    between parent node (x=0) or the node indicated by the index (x=1), or on
    a `neat.MorphLoc`.

    Parameters
    ----------
        loc: tuple or dict or `neat.MorphLoc`
            if tuple: (node index, x-value)
            if dict: {'node': node index, 'x': x-value}
        reftree: `neat.MorphTree`
        set_as_comploc: bool
            if True, assumes the paremeters provided in `loc` are coordinates
            on the computational tree. Doing this while no computational tree
            has been initialized in `reftree` will result in an error.
            Defaults to False

    Raises
    ------
        ValueError
            If x-coordinate of location is not in ``[0,1]``
    """

    def __init__(self, loc, reftree, set_as_comploc=False):
        self.reftree = reftree

        if isinstance(loc, tuple):
            x = float(loc[1])
            if x > 1. or x < 0.:
                raise ValueError('x-value should be in [0,1]')
            if set_as_comploc:
                self.comp_loc = {'node': int(loc[0]), 'x': x}
                self._setOriginalLoc()
            else:
                self.loc = {'node': int(loc[0]), 'x': x}
        elif isinstance(loc, dict):
            x = float(loc['x'])
            if x > 1. or x < 0.:
                raise ValueError('x-value should be in [0,1]')
            if set_as_comploc:
                self.comp_loc = loc
                self._setOriginalLoc()
            else:
                self.loc = loc
        elif isinstance(loc, MorphLoc):
            # self.__dict__.update({key: copy.deepcopy(val)
            #                       for key, val in loc.__dict__.iteritems()
            #                       if key != 'reftree'})
            self.loc = loc.loc
            self.reftree = reftree
        else:
            raise TypeError('Not a valid location type, should be tuple or dict')

    def __getitem__(self, key):
        if isinstance(key, int) and key in (0,1):
            key = 'node' if key == 0 else 'x'
        if isinstance(key, str):
            if self.reftree.treetype == 'computational':
                try:
                    return self.comp_loc[key]
                except AttributeError:
                    self._setComputationalLoc()
                    return self.comp_loc[key]
            else:
                return self.loc[key]

    def __eq__(self, other_loc):
        if type(other_loc) == dict:
            result = (other_loc['node'] == self.loc['node'])
            if self.loc['node'] != 1:
                result *= np.allclose(other_loc['x'], self.loc['x'])
            return result
        elif type(other_loc) == tuple:
            result = (other_loc[0] == self.loc['node'])
            if self.loc['node'] != 1:
                   result *= np.allclose(other_loc[1], self.loc['x'])
            return result
        elif isinstance(other_loc, MorphLoc):
            result = (other_loc.loc['node'] == self.loc['node'])
            if self.loc['node'] != 1:
                result *= np.allclose(other_loc.loc['x'], self.loc['x'])
            return result
        else:
            return NotImplemented

    def keys(self):
        return ['node', 'x']

    def __iter__(self):
        yield self['node']
        yield self['x']

    def __neq__(self, other_loc):
        result = self.__eq__(other_loc)
        if result is NotImplemented:
            return result
        else:
            return not result

    def __copy__(self):
        """
        Customization of the copy function so that `loc` and `comp_loc`
        attributes are deep copied and `reftree` attribute still refers to the
        original tree
        """
        new_loc = type(self)(copy.deepcopy(self.loc), self.reftree)
        if hasattr(self, 'comp_loc'):
            new_loc.__dict__.update({'comp_loc': copy.deepcopy(self.comp_loc)})
        return new_loc

    def __str__(self):
        return '{\'node\': %d, \'x\': %.2f }'%(self.loc['node'], self.loc['x'])

    def __repr__(self):
        return str(self)

    def _setComputationalLoc(self):
        if self.loc['node'] != 1:
            current_treetype = self.reftree.treetype
            self.reftree.treetype = 'original'
            node = self.reftree[self.loc['node']]
            # find the computational nodes that are resp. up and down from the node
            node_start = self.reftree._findCompnodeToRoot(node.parent_node)
            node_stop  = self.reftree._findCompnodeFromRoot(node)
            # length between loc and parent computational node to compute segment
            # length
            L = self.reftree.pathLength({'node': node_start.index, 'x': 1.},
                                         self.loc)
            # get the computational nodes' length
            self.reftree.treetype = 'computational'
            L_cn = self.reftree[node_stop.index].L
            self.reftree.treetype = 'original'
            # set the computational loc
            self.comp_loc = {'node': node_stop.index, 'x': L/L_cn}
            # reset treetype to its former value
            self.reftree.treetype = current_treetype
        else:
            self.comp_loc = copy.deepcopy(self.loc)

    def _setOriginalLoc(self):
        if self.comp_loc['node'] != 1:
            current_treetype = self.reftree.treetype
            self.reftree.treetype = 'computational'
            compnode = self.reftree[self.comp_loc['node']]
            self.reftree.treetype = 'original'
            node = self.reftree[self.comp_loc['node']]
            # find the computational node that is down from the original node
            pcnode = self.reftree._findCompnodeToRoot(node.parent_node)
            # find the node index and x-coordinate of the original location
            path = self.reftree.pathBetweenNodes(pcnode, node)
            L0 = 0. ; found = False
            for pathnode in path[1:]:
                L1 = L0 + pathnode.L
                Lloc = self.comp_loc['x']*compnode.L
                if Lloc == 0.: Lloc += 1e-7
                if Lloc > L0 and Lloc <= L1:
                    self.loc = {'node': pathnode.index,
                                'x': (Lloc-L0-1e-8) / pathnode.L}
                L0 = L1
            if self.loc['x'] > 1. or self.loc['x'] < 0.:
                raise ValueError('x-value should be in [0,1]')
            # reset treetype to its former value
            self.reftree.treetype = current_treetype
        else:
            self.loc = copy.deepcopy(self.comp_loc)


class MorphNode(SNode):
    """
    Node associated with `neat.MorphTree`. Stores the geometrical information
    associated with a point on the tree morphology

    Attributes
    ----------
        xyz: numpy.array of floats
            The xyz-coordinates associated with the node (um)
        R: float
            The radius of the node (um)
        swc_type: int
            The type of node, according to the .swc file format convention:
            ``1`` is dendrites, ``2`` is axon, ``3`` is basal dendrite and ``4``
            is apical dendrite.
        L: float
            The length of the node (um)
    """
    def __init__(self, index, p3d=None):
        super().__init__(index)
        if p3d != None:
            self.setP3D(*p3d)
        else:
            # bogus values, to overwrite
            self.setP3D(np.array([0.,0.,0.]), 1., 1)
            self.L = 1.
            self.R = 1.

    def setP3D(self, xyz, R, swc_type):
        """
        Set the 3d parameters of the node

        Parameters
        ----------
            xyz: `np.array`
                3D location (um)
            R: float
                Radius of the segment (um)
            swc_type: int
                Type asscoiated with the segment according to SWC standards
        """
        # morphology parameters
        self.xyz = xyz
        self.R = R
        self.swc_type = swc_type
        # auxiliary variable
        self.used_in_comp_tree = False

    def setLength(self, L):
        """
        Set the length of the segment represented by the node

        Parameters
        ----------
            L: float
                the length of the segment (um)
        """
        self.L = L

    def setRadius(self, R):
        """
        Set the radius of the segment represented by the node

        Parameters
        ----------
            L: float
                the radius of the segment (um)
        """
        self.R = R

    def getChildNodes(self, skip_inds=(2,3)):
        """
        Get the `child_nodes` of this node. Indices ``2`` and ``3`` are skipped
        by default (3-point soma convention)

        Parameters
        ----------
        skip_inds: list or tuple of ints
            Node indices of child nodes that are not added to the returned list

        Returns
        -------
        list of `neat.MorphNode`
            The child nodes
        """
        return [cnode for cnode in self._child_nodes \
                      if cnode.index not in skip_inds]

    def setChildNodes(self, cnodes):
        return super().setChildNodes(cnodes)

    child_nodes = property(getChildNodes, setChildNodes)


    def __str__(self, **kwarg):
        return super().__str__(**kwarg)


class MorphTree(STree):
    """
    Subclass of simple tree that implements neuronal morphologies. Reads in
    trees from '.swc' files (http://neuromorpho.org/).

    Neural morphologies are assumed to follow the three-point soma conventions.
    Internally however, the soma is represented as a sphere. Hence nodes with
    indices 2 and 3 do not represent anything and are skipped in iterations and
    getters.

    Can also store a simplified version of the original tree, where only nodes
    are retained that should hold computational parameters - the root, the
    bifurcation nodes and the leafs at least, although the user can also
    specify additional nodes. One tree is set as primary by changing the
    `treetype` attribute (select 'original' for the original morphology and
    'computational' for the computational morphology). Lookup operations will
    often use the primary tree. Using nodes from the other tree for lookup
    operations is unsafe and should be avoided, it is better to set the proper
    tree to primary first.

    For computational efficiency, it is possible to store sets of locations on
    the morphology, under user-specified names. These sets are stored as
    lists of `neat.MorphLoc`, and associated arrays are stored that contain the
    corresponding node indices of the locations, their x-coordinates, their
    distances to the soma and their distances to the nearest bifurcation in the
    in the direction of the soma.

    Parameters
    ----------
    file_n: str (optional)
        the file name of the morphology file. Assumed to follow the '.swc' format.
        Default is ``None``, which initialized an empty tree
    types: list of int (optional)
        The list of node types to be included. As per the '.swc' convention,
        ``1`` is soma, ``2`` is axon, ``3`` is basal dendrite and ``4`` apical
        dendrite. Default is ``[1,3,4]``.

    Attributes
    ----------
        root: `neat.MorphNode` instance
            The root of the tree.
        locs: dict {str: list of `neat.MorphLoc`}
            Stored sets of locations, key is the user-specified the name of the
            set of locations. Initialized as empty dict.
        nids: dict {str: np.array of int}
            Node indices of locations. Initialized as empty dict.
        xs: dict {str: np.array of float}
            x-coordinates of locations. Initialized as empty dict.
        d2s: dict {str: np.array of float}
            distances to soma of locations. Initialized as empty dict.
        d2b: dict {str: np.array of float}
            distances to nearest bifurcation in the direction of the soma
            of locations. Initialized as empty dict.
    """

    def __init__(self, file_n=None, types=[1,3,4]):
        self._treetype = 'original' # alternative 'computational'
        if file_n != None:
            self.readSWCTreeFromFile(file_n, types=types)
            # self._original_root = self.root
        else:
            self._original_root = None
        self._computational_root = None
        # to store sets of locations on the morphology
        self.locs = {}
        self._nids_orig = {}; self._nids_comp = {}
        self._xs_orig = {}; self._xs_comp = {}
        self.d2s = {}
        self.d2b = {}
        self.leafinds = {}

    def __getitem__(self, index, skip_inds=(2,3)):
        """
        Returns the node with given index, if no such node is in the tree, None
        is returned.

        Parameters
        ----------
            index: int
                the index of the node to be found

        Returns:
            `neat.MorphNode` or None
        """
        return self._findNode(self.root, index, skip_inds=skip_inds)

    def _findNode(self, node, index, skip_inds=(2,3)):
        """
        Sweet breadth-first/stack iteration to replace the recursive call.
        Traverses the tree until it finds the node you are looking for.
        Returns SNode when found and None when not found

        Parameters
        ----------
            node: :class:`SNode` (optional)
                node where the search is started
            index: int
                the index of the node to be found

        Returns
        -------
            :class:`SNode`
        """
        stack = [];
        stack.append(node)
        while len(stack) != 0:
            for cnode in stack:
                if cnode.index == index:
                    return cnode
                else:
                    stack.remove(cnode)
                    stack.extend(cnode.getChildNodes(skip_inds=skip_inds))
        return None # Not found!

    def __iter__(self, node=None, skip_inds=(2,3)):
        """
        Overloaded iterator from parent class that avoids iterating over the
        nodes with index 2 and 3

        Parameters
        ----------
            node: `neat.MorphNode`
                The starting node. Defaults to the root
            skip_inds: tuple of ints
                Indices of the nodes that are skipped by the iterator. Defaults
                to ``(2,3)``, the nodes that contain extra geometrical
                information on the soma.

        Yields
        ------
            `neat.MorphNode`
                Nodes in the tree
        """
        if node is None:
            node = self.root
        if node is not None:
            if node.index not in skip_inds: yield node
            for cnode in node.getChildNodes(skip_inds=skip_inds):
                for inode in self.__iter__(cnode, skip_inds=skip_inds):
                    if node.index not in skip_inds: yield inode

    def resetIndices(self):
        """
        Resets the indices in the order they appear in a depth-first iteration
        """
        for ind, node in enumerate(self):
            node.index = ind+1

    def getRoot(self):
        """
        Returns the root of the original or the computational tree, depending
        on which `treetype` is active.
        """
        if self.treetype == 'original':
            return self._original_root
        else:
            return self._computational_root

    def setRoot(self, node):
        if self.treetype == 'original':
            node.parent_node = None
            self._original_root = node
        else:
            node.parent_node = None
            self._computational_root = node

    root = property(getRoot, setRoot)

    def getNodes(self, recompute_flag=0, skip_inds=(2,3)):
        """
        Overloads the parent function to allow skipping nodes with certain
        indices and to return the nodes associated with the corresponding
        `treetype`.

        Parameters
        ----------
            recompute_flag: bool
                whether or not to re-evaluate the node list. Defaults to False.
            skip_inds: tuple of ints
                Indices of the nodes that are skipped by the iterator. Defaults
                to ``(2,3)``, the nodes that contain extra geometrical
                information on the soma.

        Returns
        -------
            list of `neat.MorphNode`
        """
        if self.treetype == 'original':
            if not hasattr(self, '_nodes_orig') or recompute_flag:
                self._nodes_orig = []
                self._gatherNodes(self.root, self._nodes_orig,
                                   skip_inds=skip_inds)
            return self._nodes_orig
        else:
            if not hasattr(self, '_nodes_comp') or recompute_flag:
                self._nodes_comp = []
                self._gatherNodes(self.root, self._nodes_comp,
                                   skip_inds=skip_inds)
            return self._nodes_comp

    def setNodes(self, illegal):
        raise AttributeError("`nodes` is a read-only attribute")

    nodes = property(getNodes, setNodes)

    def _gatherNodes(self, node, node_list=[], skip_inds=(2,3)):
        """
        Overloaded gathering function that avoids appending nodes with index 2
        or 3 to the list.

        Parameters
        ----------
            node: `neat.MorphNode`
            node_list: list of `neat.MorphNode`
        """
        if node.index not in skip_inds: node_list.append(node)
        for cnode in node.getChildNodes(skip_inds=skip_inds):
            self._gatherNodes(cnode, node_list=node_list, skip_inds=skip_inds)

    def getLeafs(self, recompute_flag=0):
        """
        Overloads the `getLeafs` of the parent class to return the leafs
        in the current `treetype`.

        Parameters
        ----------
            recompute_flag: bool
                Whether to force recomputing the leaf list. Defaults to 0.
        """
        if self.treetype == 'original':
            if not hasattr(self, '_leafs_orig') or recompute_flag:
                self._leafs_orig = [node for node in self if self.isLeaf(node)]
            return self._leafs_orig
        else:
            if not hasattr(self, '_leafs_comp') or recompute_flag:
                self._leafs_comp = [node for node in self if self.isLeaf(node)]
            return self._leafs_comp

    def setLeafs(self, illegal):
        raise AttributeError("`leafs` is a read-only attribute")

    leafs = property(getLeafs, setLeafs)

    def getNodesInBasalSubtree(self):
        """
        Return the nodes associated with the basal subtree

        Returns
        -------
            list of `neat.MorphNode`
                List of all nodes in the basal subtree
        """
        return [node for node in self if node.swc_type in [3]]

    def getNodesInApicalSubtree(self):
        """
        Return the nodes associated with the apical subtree

        Returns
        -------
            list of `neat.MorphNode`
                List of all nodes in the apical subtree
        """
        return [node for node in self if node.swc_type in [4]]

    def getNodesInAxonalSubtree(self):
        """
        Return the nodes associated with the apical subtree

        Returns
        -------
            list of `neat.MorphNode`
                List of all nodes in the apical subtree
        """
        return [node for node in self if node.swc_type in [2]]

    def setTreetype(self, treetype):
        """
        Set the active tree

        Parameters
        ----------
        treetype: 'original' or 'computational'
            the treetype thas is set to active
        """
        if treetype == 'original':
            self._treetype = treetype
            self.root = self._original_root
        elif treetype == 'computational':
            if self._computational_root is not None:
                self._treetype = treetype
                self.root = self._computational_root
            else:
                raise ValueError('no computational tree has been defined, \
                                `treetype` can only be \'original\'')
        else:
            raise ValueError('`treetype` can be \'original\' or \'computational\'')

    def getTreetype(self):
        return self._treetype

    treetype = property(getTreetype, setTreetype)

    def setComputationalRoot(self, node):
        if node is None:
            self._treetype = 'original'
        self.__computational_root = node

    def getComputationalRoot(self):
        return self.__computational_root

    _computational_root = property(getComputationalRoot, setComputationalRoot)

    def _createCorrespondingNode(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return MorphNode(node_index, p3d=p3d)

    def readSWCTreeFromFile(self, file_n, types=[1,3,4]):
        """
        Non-specific for a "tree data structure"
        Read and load a morphology from an SWC file and parse it into
        an `neat.MorphTree` object.

        On the NeuroMorpho.org website, 5 types of somadescriptions are
        considered (http://neuromorpho.org/neuroMorpho/SomaFormat.html).
        The "3-point soma" is the standard and most files are converted
        to this format during a curation step. `neat` follows this default
        specification and the *internal structure of `neat` implements
        the 3-point soma*. Additionally multi-cylinder descriptions with more
        than three nodes are also supported, but are converted to the standard
        three point description.

        Additionally, the root node of the tree must have ``index == 1``,
        ``swc_type == 1`` and occur first in the SWC file.

        Parameters
        -----------
        file_n: str
            name of the file to open
        types: list of ints
            NeuroMorpho.org segment types to be loaded

        Examples
        --------
        The three point description is

        .. code-block:: python

             1 1 x y   z r -1
             1 1 x y-r z r 1
             1 1 x y+r z r 1

        with `x,y,z` the coordinates of the soma center and `r` the soma radius

        This is a valid three point desciption

        .. code-block:: python

            # start of file
            1 1 45.3625 18.6775 -50.25 10.1267403895 -1
            2 1 45.3625 8.55075961052 -50.25 10.1267403895 1
            3 1 45.3625 28.8042403895 -50.25 10.1267403895 1
            # dendrite nodes
            4 3 37.76 12.99 -46.08 0.29 1
            5 3 26.7068019951 8.26344199599 -36.9426896493 0.795614809475 4
            # ...

        This is a valid multi-cylinder descirption

        .. code-block:: python

            # start of file
            1 1 1066.38 399.67 157.0 4.9215 -1
            2 1 1071.3 399.67 157.0 4.9215 1
            3 1 1076.22 399.67 157.0 4.9215 2
            4 1 1066.5 402.83 157.0 11.494 2
            5 1 1062.4 405.5 157.0 15.308 4
            6 1 1056.6 410.25 158.0 20.536 5
            7 1 1056.6 410.25 158.0 20.536 6
            8 1 1070.0 427.75 161.0 2.305 7
            # dendrite nodes
            9 3 1070.0 427.75 161.0 0.886 8
            # ...

        Raises
        ------
        ValueError
            If the SWC file is not consistent with the aforementioned conventions

        """
        # check soma-representation: 3-point soma or a non-standard representation
        soma_type = self.determineSomaType(file_n)

        file = open(file_n,'r')
        all_nodes = dict()
        for line in file :
            if not line.startswith('#') :
                split = line.split()
                index = int(split[0].rstrip())
                swc_type = int(split[1].rstrip())
                x = float(split[2].rstrip())
                y = float(split[3].rstrip())
                z = float(split[4].rstrip())
                radius = float(split[5].rstrip())
                parent_index = int(split[6].rstrip())
                # create the nodes
                if swc_type in types:
                    p3d = (np.array([x,y,z]), radius, swc_type)
                    node = self._createCorrespondingNode(index, p3d)
                    all_nodes[index] = (swc_type, node, parent_index)

        # check if node with index 1 is soma node (swc_type == 1)
        if all_nodes[1][0] != 1:
            raise ValueError('Node with index 1 should be soma-type, i.e. swc_type == 1')

        # one point soma representation
        if soma_type == 0:
            for index, (swc_type, node, parent_index) in list(all_nodes.items()) :
                if index == 1:
                    self.setRoot(node)
                else:
                    parent_node = all_nodes[parent_index][1]
                    self.addNodeWithParent(node, parent_node)

            # we add nodes 2 and 3 to adhere to obtain a 3-point soma representation
            # (http://neuromorpho.org/neuroMorpho/SomaFormat.html)
            # and increment the indices of all nodes that are not the soma by 2
            for node in self.__iter__(skip_inds=[]):
                if self.isRoot(node):
                    # create p3ds for the extra soma nodes
                    xyz_2 = copy.copy(node.xyz)
                    xyz_2[1] -= node.R
                    p3d_2 = (xyz_2, node.R, node.swc_type)
                    xyz_3 = copy.copy(node.xyz)
                    xyz_3[1] += node.R
                    p3d_3 = (xyz_3, node.R, node.swc_type)
                else:
                    node.index += 2
            # add extra soma nodes to the tree
            snode_2 = self._createCorrespondingNode(2, p3d_2)
            snode_3 = self._createCorrespondingNode(3, p3d_3)
            self.addNodeWithParent(snode_2, self[1])
            self.addNodeWithParent(snode_3, self[1])

        # three point soma representation
        if soma_type == 1:
            for index, (swc_type, node, parent_index) in list(all_nodes.items()) :
                if index == 1:
                    self.setRoot(node)
                elif index in (2,3):
                    # the 3-point soma representation
                    # (https://neuromorpho.org/SomaFormat.html)
                    somanode = all_nodes[1][1]
                    self.addNodeWithParent(node, somanode)
                else:
                    parent_node = all_nodes[parent_index][1]
                    self.addNodeWithParent(node, parent_node)

            # check if soma follows three point convention
            radius_arr = np.array([all_nodes[1][1].R,
                                   all_nodes[2][1].R,
                                   all_nodes[3][1].R,
                                   np.linalg.norm(all_nodes[2][1].xyz - all_nodes[1][1].xyz),
                                   np.linalg.norm(all_nodes[3][1].xyz - all_nodes[1][1].xyz)])
            if not np.allclose(np.abs(radius_arr - radius_arr[0]),
                               np.zeros_like(radius_arr), atol=2e-2):
                raise ValueError('Soma radii not consistent with three-point convention')

        # IF multiple cylinder soma representation
        elif soma_type == 2:
            self.setRoot(all_nodes[1][1])

            # get all soma info
            soma_cylinders = []
            connected_to_root = []
            for index, (swc_type, node, parent_index) in list(all_nodes.items()) :
                if swc_type == 1 and not index == 1:
                    soma_cylinders.append((node, parent_index))
                    if index > 1 :
                        connected_to_root.append(index)

            # make soma
            s_node_2, s_node_3 = \
                    self._makeSomaFromCylinders(soma_cylinders, all_nodes)

            # add soma
            self.root.R = s_node_2.R
            self.addNodeWithParent(s_node_2, self.root)
            self.addNodeWithParent(s_node_3, self.root)

            # add the other points
            for index, (swc_type, node, parent_index) in list(all_nodes.items()) :
                if swc_type == 1:
                    pass
                else:
                    parent_node = all_nodes[parent_index][1]
                    if parent_node.index in connected_to_root:
                        self.addNodeWithParent(node, self.root)
                    else:
                        self.addNodeWithParent(node, parent_node)

        # set the lengths of the nodes
        for node in self:
            if node.parent_node != None:
                L = np.sqrt(np.sum((node.parent_node.xyz - node.xyz)**2))
                # if the length is zero we can just delete it
                if L == 0:
                    self.removeSingleNode(node)
            else:
                L = 0.
            node.setLength(L)

        return self

    def _makeSomaFromCylinders(self, soma_cylinders, all_nodes):
        """
        Construct 3-point soma
        Step 1: calculate surface of all cylinders
        Step 2: make 3-point representation with the same surface
        """
        total_surf = 0
        xyz_sum = self.root.xyz
        for (node, parent_index) in soma_cylinders:

            parent = all_nodes[parent_index][1]

            nxyz = node.xyz
            pxyz = parent.xyz

            H = np.sqrt(np.sum( (nxyz - pxyz)**2 ))

            surf = 2 * np.pi * parent.R * H
            total_surf += surf

            xyz_sum += node.xyz

        # define apropriate radius
        radius = np.sqrt(total_surf / (4.*np.pi))
        rp = xyz_sum / (len(soma_cylinders)+1.)
        rp2 = np.array([rp[0], rp[1] - radius, rp[2]])
        rp3 = np.array([rp[0], rp[1] + radius, rp[2]])

        self.root.xyz = rp
        # create the soma nodes
        s_node_2 = self._createCorrespondingNode(2, (rp2, radius, 1))
        s_node_3 = self._createCorrespondingNode(3, (rp3, radius, 1))

        return s_node_2, s_node_3

    def determineSomaType(self, file_n):
        """
        Determine the soma type used in the SWC file.
        This method searches the whole file for soma entries.

        Only tbe standard three-point soma type and a multi-cylinder description
        are supported.

        Furthermore, the root node of the tree must have ``index == 1``,
        ``swc_type == 1`` and occur first in the SWC file.

        Parameters
        ----------
        file_n: string
            Name of the file containing the SWC description

        Returns
        -------
        soma_type: int
            Integer indicating one of the su[pported SWC soma formats.
            0: One point soma
            1: Default three-point soma,
            2: multiple cylinder description

        Raises
        ------
        ValueError
            If soma type is not supported (less than three nodes have soma)
        """
        file = open(file_n, 'r')
        somas = 0
        for line in file:
            if not line.startswith('#') :
                split = line.split()
                index = int(split[0].rstrip())
                s_type = int(split[1].rstrip())
                if s_type == 1 :
                    somas = somas +1
        file.close()
        if somas == 1:
            return 0
        if somas == 3:
            return 1
        elif somas < 3:
            raise ValueError('Soma description not supported, use 3-point or multi-cylinder description')
        else:
            return 2

    def _evaluateCompCriteria(self, node, eps=1e-8, rbool=False):
        """
        Return ``True`` if relative difference between node radius and parent
        node raidus is larger than margin ``eps``, or if the node is the root
        or bifurcation node.

        Parameters
        ----------
        node: `neat.MorphNode`
            node that is compared to parent node
        eps: float (optional, default ``1e-8``)
            the margin

        return
        ------
        bool
        """
        if not rbool:
            rbool = node.parent_node == None
        if not rbool:
            rbool = len(node.getChildNodes()) != 1
        if not rbool:
            cnode = node.child_nodes[0]
            rbool = np.abs(node.R - cnode.R) > eps * np.max([node.R, cnode.R])

        return rbool

    def setCompTree(self, compnodes=None, set_as_primary_tree=False, eps=1e-8):
        """
        Sets the nodes that contain computational parameters. This are a priori
        either bifurcations, leafs, the root or nodes where the neurons'
        relevant parameters change.

        Parameters
        ----------
            compnodes: list of ::class::`MorphNode`
                list of nodes that should be retained in the computational tree.
                Note that specifying bifurcations, leafs or the root is
                superfluous, since they are part of the computational tree by
                default.
            set_as_primary_tree: bool (default ``False``)
                if True, sets the computational tree as the primary tree
            eps: float (default ``1e-8``)
                relative margin for parameter change
        """
        self.removeCompTree()
        if compnodes is None:
            compnodes = []
        compnodes += [node for node in self if self._evaluateCompCriteria(node, eps=eps)]
        compnode_indices = [node.index for node in compnodes]
        nodes = copy.deepcopy(self.nodes)

        for node in nodes:
            if node.index not in compnode_indices:
                self.removeSingleNode(node)
            elif node.parent_node != None:
                orig_node = self[node.index]
                orig_bnode = node.parent_node
                L, R = self.pathLength({'node': orig_bnode.index, 'x': 1.},
                                        {'node': orig_node.index, 'x': 1.},
                                        compute_radius=1)
                node.setLength(L)
                node.setRadius(R)
                node.used_in_comp_tree = True
                orig_node.used_in_comp_tree = True
            else:
                orig_node = self[node.index]
                node.used_in_comp_tree = True
                orig_node.used_in_comp_tree = True

        self._computational_root = \
                    next(node for node in nodes if node.index == 1)
        self._leafs_comp = [node for node in nodes if self.isLeaf(node)]
        self._nodes_comp = []
        self._gatherNodes(self._computational_root, self._nodes_comp)

        if set_as_primary_tree:
            self.treetype = 'computational'
        # create conversion of all coordinate arrays
        for name in self.locs:
            self._storeCompLocs(name)

    def _findCompnodeToRoot(self, node):
        """
        !!! Computational tree has to be initialized, otherwise may results in
        error !!!

        If the input node is a node of the original tree, finds the first node
        on the path to the root that has an equivalent in the computational tree.
        If the input node has such an equivalent, it is returned itself.

        If the input node is in the computational tree, returns the node itself.

        Parameters
        ----------
            node: `neat.MorphNode` instance
                the input node

        Returns
        -------
            `neat.MorphNode` instance
        """
        if not node.used_in_comp_tree:
            node = self._findCompnodeToRoot(node.parent_node)
        return node

    def _findCompnodeFromRoot(self, node):
        """
        !!! Computational tree has to be initialized, otherwise may results in
        error !!!

        If the input node is a node of the original tree, finds the first node
        away from the root that has an equivalent in the computational tree. If
        the input node has such an equivalent, it is returned itself.

        If the input node is in the computational tree, returns the node itself.

        Parameters
        ----------
            node: `neat.MorphNode` instance
                the input node

        Returns
        -------
            `neat.MorphNode` instance
        """
        if not node.used_in_comp_tree:
            node = self._findCompnodeFromRoot(node.child_nodes[0])
        return node

    def removeCompTree(self):
        """
        Removes the computational tree
        """
        self._computational_root = None
        try:
            delattr(self, "_nodes_comp")
        except AttributeError as err:
            pass
        try:
            delattr(self, "_leafs_comp")
        except AttributeError as err:
            pass
        self.treetype = 'original'
        for node in self:
            node.used_in_comp_tree = False

    def _convertLocArgToLocs(self, locarg):
        """
        Converts locations argument to list of `neat.MorphLoc`.

        Parameters
        ----------
        locarg: list of dictionaries, tuples or `neat.MorphLoc`, or string
            * If list, entries should be valid arguments to initialize a `neat.MorphLoc`
            * If string, should be the name of a list of locations stored in `self`

        Returns
        -------
        list of `neat.MorphLoc`
            List of locations, each referencing the current tree
        """
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
        elif isinstance(locarg, str):
            self._tryName(locarg)
            locs = self.getLocs(locarg)
        else:
            raise IOError('`locarg` should be list of locs or string')
        return locs

    def _convertNodeArgToNodes(self, node_arg):
        """
        Converts a node argument to a list of nodes. Behaviour depends on the
        type of argument.

        Parameters
        ----------
        node_arg: ``None``, `neat.MorphNode`, {'apical', 'basal', 'axonal'} or iterable collection of instances of `neat.MorphNode`
            * `None`: returns all nodes
            * `neat.MorphNode`: returns list of nodes in the subtree of the given node
            * {'apical', 'basal', 'axonal'}: returns list of nodes in the apical, basal or axonal subtree
            * iterable collection of `neat.MorphNode`: returns the same list of nodes
                If an iterable collection of original nodes is given, and the treetype
                is computational, a reduced list is returned where only the corresponding
                computational nodes are included. If an iterable collection of
                computational nodes is given, and the treetype is original, a list of
                corresponding original nodes is given, but the in between nodes are not
                added.


        Returns
        -------
        list of `neat.MorphNode`
        """
        # convert the input argument to a list of nodes
        if node_arg == None:
            nodes = self.nodes
        elif isinstance(node_arg, MorphNode):
            if self.treetype == 'computational':
                # assure that a list of computational nodes is returned
                node_arg = self._findCompnodeFromRoot(node_arg)
                node_arg = self[node_arg.index]
            else:
                # assure that a list of original nodes is returned
                node_arg = self[node_arg.index]
            nodes = self.gatherNodes(node_arg)
        elif node_arg == 'apical':
            nodes = self.getNodesInApicalSubtree()
        elif node_arg == 'basal':
            nodes = self.getNodesInBasalSubtree()
        elif node_arg == 'axonal':
            nodes = self.getNodesInAxonalSubtree()
        else:
            try:
                nodes = []
                for node in node_arg:
                    assert isinstance(node, MorphNode)
                    if self.treetype == 'computational':
                        # assure that a list of computational nodes is returned
                        node_ = self._findCompnodeFromRoot(node)
                        compnode = self[node_.index]
                        if compnode not in nodes:
                            nodes.append(compnode)
                    else:
                        # assure that a list of original nodes is returned
                        nodes.append(self[node.index])
            except (AssertionError, TypeError):
                raise ValueError('input should be (i) `None`, (ii) an instance of '
                        '`neat.MorphNode`, (iii) one of the following 3 strings '
                        '\'apical\', \'basal\' or \'axonal\' or (iv) an iterable '
                        'collection of instances of :class:MorphNode')

        return nodes

    def pathLength(self, loc1, loc2, compute_radius=0):
        """
        Find the length of the direct path between loc1 and loc2

        Parameters
        ----------
            loc1: dict, tuple or `neat.MorphLoc`
                one location
            loc2: dict, tuple or `neat.MorphLoc`
                other location
            compute_radius: bool
                if True, also computes the average weighted radius of the path

        Returns
        -------
        L, R (optional)
            L: float
                length of path, in micron
            R: float
                weighted average radius of path, in micron
        """
        # define location objects
        if type(loc1) == dict or type(loc1) == tuple:
            loc1 = MorphLoc(loc1, self)
        if type(loc2) == dict or type(loc2) == tuple:
            loc2 = MorphLoc(loc2, self)
        # start path length calculation
        if loc1['node'] == loc2['node']:
            node = self[loc1['node']]
            if node.index == 1:
                L = 0. # soma is spherical and has no lenght
            else:
                L = node.L * np.abs(loc1['x'] - loc2['x'])
            if compute_radius:
                R = node.R
        else:
            node1 = self[loc1['node']]
            node2 = self[loc2['node']]
            path1 = self.pathToRoot(node1)[::-1]
            path2 = self.pathToRoot(node2)[::-1]
            path = path1 if len(path1) < len(path2) else path2
            ind = next((ii for ii in range(len(path)) if path1[ii] != path2[ii]),
                       len(path))
            if path1[ind-1] == node1:
                L  = node1.L * (1. - loc1['x'])
                L += sum(node.L for node in path2[ind:-1])
                L += node2.L * loc2['x']
                if compute_radius:
                    R  = node1.R * node1.L * (1. - loc1['x'])
                    R += sum(node.R * node.L for node in path2[ind:-1])
                    R += node2.R * node2.L * loc2['x']
                    R /= L
            elif path2[ind-1] == node2:
                L  = node1.L * loc1['x']
                L += sum(node.L for node in path1[ind:-1])
                L += node2.L * (1. - loc2['x'])
                if compute_radius:
                    R  = node1.R * node1.L * loc1['x']
                    R += sum(node.R * node.L for node in path2[ind:-1])
                    R += node2.R * node2.L * (1. - loc2['x'])
                    R /= L
            else:
                L  = node1.L * loc1['x']
                L += sum(node.L for node in path1[ind:-1])
                L += sum(node.L for node in path2[ind:-1])
                L += node2.L * loc2['x']
                if compute_radius:
                    R  = node1.R * node1.L * loc1['x']
                    R += sum(node.R * node.L for node in path1[ind:-1])
                    R += sum(node.R * node.L for node in path2[ind:-1])
                    R += node2.R * node2.L * loc2['x']
                    R /= L
        if compute_radius:
            return L, R
        else:
            return L

    @originalTreetypeDecorator
    def storeLocs(self, locs, name, warn=True):
        """
        Store locations under a specified name

        Parameters
        ----------
            locs: list of dicts, tuples or `neat.MorphLoc`
                the locations to be stored
            name: string
                name under which these locations are stored
            warn: bool (default ``True``)
                raise a `UserWarning` if two or more locations in `locs` refer
                to the soma. Choose ``False`` if this is desired to remove
                the warning.
        """
        # copy list and store in MorphLoc if necessary
        locs_ = []
        n1 = 0
        for loc in locs:
            locs_.append(MorphLoc(loc, self))
            if locs_[-1]['node'] == 1: n1 += 1
        if n1 > 1 and warn:
            warnings.warn('There are multiple locations on the soma in this set ' + \
                          'locations, this can cause issues in certain functions', UserWarning)

        self.removeLocs(name)

        self.locs[name] = locs_
        self._nids_orig[name] = np.array([loc['node'] for loc in locs_])
        self._xs_orig[name] = np.array([loc['x'] for loc in locs_])
        if self._computational_root != None:
            self._storeCompLocs(name)

    @computationalTreetypeDecorator
    def _storeCompLocs(self, name):
        self._nids_comp[name] = np.array([loc['node'] for loc in self.locs[name]])
        self._xs_comp[name] = np.array([loc['x'] for loc in self.locs[name]])

    @originalTreetypeDecorator
    def addLoc(self, loc, name):
        """
        Add location to set of locations of given name

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            the location to be added
        name: str
            the name of the set of locations to which the location is added
        """
        loc = MorphLoc(loc, self)
        self.locs[name].append(loc)
        self._nids_orig[name] = np.concatenate((self._nids_orig[name], [loc['node']]))
        self._xs_orig[name] = np.concatenate((self._xs_orig[name], [loc['x']]))
        if self._computational_root != None:
            self._addCompLoc(loc, name)

    @computationalTreetypeDecorator
    def _addCompLoc(self, loc, name):
        self._nids_comp[name] = np.concatenate((self._nids_comp[name], [loc['node']]))
        self._xs_comp[name] = np.concatenate((self._xs_comp[name], [loc['x']]))

    def clearLocs(self):
        """
        Remove all set of locs stored in the tree
        """
        self.locs = {}
        self._nids_orig = {}; self._nids_comp = {}
        self._xs_orig = {}; self._xs_comp = {}
        self.d2s = {}
        self.d2b = {}
        self.leafinds = {}

    def removeLocs(self, name):
        """
        Remove a set of locations of a given name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored
        """
        try:
            del self.locs[name]
            del self._nids_orig[name]
            del self._nids_comp[name]
            del self._xs_orig[name]
            del self._xs_comp[name]
        except KeyError: pass
            # warnings.warn('Locations of name %s were not defined'%name)
        try:
            del self.d2s[name]
        except KeyError: pass
        try:
            del self.d2b[name]
        except KeyError: pass
        try:
            del self.leafinds[name]
        except KeyError: pass

    def _tryName(self, name):
        """
        Tests if the name is in use. Raises a KeyError when it is not in use and
        prints a list of possible names

        Parameters
        ----------
            name: string
                name of the desired list of locations

        Raises
        ------
            KeyError
                If 'name' does not refer to a set of locations in use
        """
        try:
            self.locs[name]
        except KeyError as err:
            err.args = ('\'' + err.args[0] \
                             + '\' name not in use. Possible names are ' \
                             + str(list(self.locs.keys())),)
            raise

    def getLocs(self, name):
        """
        Returns a set of locations of a specified name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored

        Returns
        -------
            list of `neat.MorphLoc`
        """
        self._tryName(name)
        return self.locs[name]

    def getNodeIndices(self, name):
        """
        Returns an array of node indices of locations of a specified name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored

        Returns
        -------
            numpy.array of ints
        """
        self._tryName(name)
        return self.nids[name]

    def getNids(self):
        if self.treetype == 'original':
            return self._nids_orig
        else:
            return self._nids_comp

    def setNids(self, nids):
        if self.treetype == 'original':
            self._nids_orig = nids
        else:
            self._nids_comp = nids

    nids = property(getNids, setNids)

    def getXCoords(self, name):
        """
        Returns an array of x-values of locations of a specified name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored
        """
        self._tryName(name)
        return self.xs[name]

    def getXs(self):
        if self.treetype == 'original':
            return self._xs_orig
        else:
            return self._xs_comp

    def setXs(self, xs):
        if self.treetype == 'original':
            self._xs_orig = xs
        else:
            self._xs_comp = xs

    xs = property(getXs, setXs)

    def getLocindsOnNode(self, name, node):
        """
        Returns a list of the indices of locations in the list of a given name
        that are on a the input node, ordered for increasing x

        Parameters
        ----------
            name: string
                which list of locations to consider
            node: `neat.MorphNode`
                the node to consider. Should be part of the original
                tree
        Returns
        -------
            list of ints
                indices of locations on the path
        """
        self._tryName(name)
        nids = self.nids[name]
        xs = self.xs[name]
        # get the locinds on the node
        inds = np.where(nids == node.index)[0]
        sortinds = np.argsort(xs[inds])

        return inds[sortinds].tolist()

    def getLocindsOnNodes(self, name, node_arg):
        """
        Returns a list of the indices of locations in the list of a given name
        that are on one of the nodes specified in the node list. Within each
        node, locations are ordered for increasing x

        Parameters
        ----------
            name: string
                which list of locations to consider
            node_arg:
                see documentation of `MorphTree._convertNodeArgToNodes`
        Returns
        -------
            list of ints
                indices of locations on the path
        """
        # find locinds on all nodes
        locinds = []
        for node in self._convertNodeArgToNodes(node_arg):
            locinds.extend(self.getLocindsOnNode(name, node))

        return locinds

    def getLocindsOnPath(self, name, node0, node1, xstart=0., xstop=1.):
        """
        Returns a list of the indices of locations in the list of a given name
        that are on the given path. The path is taken to start at the input
        x-start coordinate of the first node in the list and to stop at the
        given x-stop coordinate of the last node in the list

        Parameters
        ----------
            name: string
                which list of locations to consider
            node0: :class:`SNode`
                start node of path
            node1: :class:`SNode`
                stop node of path
            xstart: float (in ``[0,1]``)
                starting coordinate on `node0`
            xstop: float (in ``[0,1]``)
                stopping coordinate on `node1`

        Returns
        -------
            list of ints
                Indices of locations on the path. If path is empty, an empty
                array is returned.
        """
        self._tryName(name)
        locs = self.locs[name]
        xs = self.xs[name]
        # find the path
        path = self.pathBetweenNodes(node0, node1)
        # find the location indices
        locinds = []
        if len(path) > 1:
            # first node in path
            node = path[0]
            ninds = np.array(self.getLocindsOnNode(name, node)).astype(int)
            if node.parent_node == None:
                locinds.extend(ninds)
            else:
                if node.parent_node == path[1]:
                    # goes runs towards root
                    inds = np.where(xs[ninds] <= xstart)[0]
                    sortinds = np.argsort(xs[ninds][inds])[::-1]
                else:
                    # path goes away from root
                    inds = np.where(xs[ninds] >= xstart)[0]
                    sortinds = np.argsort(xs[ninds][inds])
                locinds.extend(ninds[inds][sortinds])
            # middle nodes in path
            for ii, node in enumerate(path[1:-1]):
                ninds = np.array(self.getLocindsOnNode(name, node)).astype(int)
                if node.parent_node == None:
                    locinds.extend(ninds)
                elif path[ii+2] == node.parent_node:
                    # path goes towards root
                    sortinds = np.argsort(xs[ninds])
                    locinds.extend(ninds[sortinds[::-1]])
                elif path[ii] == node.parent_node:
                    # path goes away from root
                    sortinds = np.argsort(xs[ninds])
                    locinds.extend(ninds[sortinds])
                else:
                    # turning point (path only goes on this node at x=1)
                    inds = np.where((1. - xs[ninds]) < 1e-4)[0]
                    if len(inds) > 0:
                        locinds.extend(ninds[inds])
            # last node in path
            node = path[-1]
            ninds = np.array(self.getLocindsOnNode(name, node)).astype(int)
            if node.parent_node == None:
                locinds.extend(ninds)
            else:
                if node.parent_node  == path[-2]:
                    # path goes away from root
                    inds = np.where(xs[ninds] <= xstop)[0]
                    sortinds = np.argsort(xs[ninds][inds])
                else:
                    # path goes towards root
                    inds = np.where(xs[ninds] >= xstop)[0]
                    sortinds = np.argsort(xs[ninds][inds])[::-1]
                locinds.extend(ninds[inds][sortinds])
        elif len(path) == 1:
            node = path[0]
            ninds = np.array(self.getLocindsOnNode(name, node)).astype(int)
            if node.parent_node == None:
                locinds.extend(ninds)
            else:
                if xstart < xstop:
                    inds = np.where(np.logical_and(xs[ninds]>=xstart, xs[ninds]<=xstop))[0]
                    sortinds = np.argsort(xs[ninds][inds])
                else:
                    inds = np.where(np.logical_and(xs[ninds]>=xstop, xs[ninds]<=xstart))[0]
                    sortinds = np.argsort(xs[ninds][inds])[::-1]
                locinds.extend(ninds[inds][sortinds])

        return locinds

    def getNearestLocinds(self, locs, name, direction=0, check_siblings=True, pprint=False):
        """
        For each location in the input location list, find the index of the
        closest location in a set of locations stored under a given name. The
        search can go in the either go in the up or down direction or in both
        directions.

        Parameters
        ----------
            locs: list of dicts, tuples or `neat.MorphLoc`
                the locations for which the nearest location index has to be
                found
            name: string
                name under which the reference list is stored
            direction: int
                flag to indicate whether to search in both directions (0), only
                in the up direction (1) or in the down direction (2).

        Returns
        -------
            loc_indices: list of ints
                indices of the locations closest to the given locs
        """
        self._tryName(name)
        # create the locs in a desirable format
        locs_ = []
        for loc in locs:
            locs_.append(MorphLoc(loc, self))
        locs = locs_
        # look for the location indices
        loc_indices = []
        for loc in locs:
            loc_ind1 = None; loc_ind2 = None
            # find the location indices if necessary
            if direction == 0 or direction == 1:
                loc_ind1 = self._findLocsToRoot(loc, name, check_siblings=check_siblings)
            if direction == 0 or direction == 2:
                loc_ind2 = self._findLocsFromRoot(loc, name)
            # save the index of the closest location, if it exists and
            # if it is asked for
            if loc_ind1 == None and (direction == 0 or direction == 2):
                loc_indices.append(loc_ind2)
            elif loc_ind2 == None and (direction == 0 or direction == 1):
                loc_indices.append(loc_ind1)
            else:
                L1 = self.pathLength(loc, self.locs[name][loc_ind1])
                L2 = self.pathLength(loc, self.locs[name][loc_ind2])
                if L1 >= L2:
                    loc_indices.append(loc_ind2)
                else:
                    loc_indices.append(loc_ind1)
        return loc_indices

    def _findLocsFromRoot(self, loc, name):
        look_further = False
        # look if there are locs on the same node
        n_inds = np.where(loc['node'] == self.nids[name])[0]
        if len(n_inds) > 0:
            if loc['node'] == 1:
                loc_ind = n_inds[0]
            else:
                x_inds = np.where(loc['x'] <= self.xs[name][n_inds])[0]
                if len(x_inds) != 0:
                    ind = np.argmin(self.xs[name][n_inds][x_inds])
                    loc_ind = n_inds[x_inds[ind]]
                else:
                    look_further = True
        else:
            look_further = True
        # if no locs on the same node, then proceed to child nodes
        # else, return the smallest location larger than loc
        if look_further:
            node = self[loc['node']]
            cnodes = node.getChildNodes()
            loc_inds = []
            for cnode in cnodes:
                cloc_ind = self._findLocsFromRoot({'node': cnode.index, 'x': 0.}, name)
                if cloc_ind != None:
                    loc_inds.append(cloc_ind)
            # get the one that is closest, if they exist
            pl_aux = 1e4
            ind_loc = 0
            for i, l_i in enumerate(loc_inds):
                pl = self.pathLength({'node': loc['node'], 'x': 1.}, self.locs[name][l_i])
                if pl < pl_aux:
                    pl_aux = pl
                    ind_loc = i
            if pl_aux > 0. and len(loc_inds) > 0:
                loc_ind = loc_inds[ind_loc]
            elif pl_aux == 0. and node.index == 1:
                loc_ind = loc_inds[ind_loc]
            else:
                loc_ind = None
        return loc_ind

    def _findLocsToRoot(self, loc, name, check_siblings=True):
        look_further = False
        # look if there are locs on the same node
        n_inds = np.where(loc['node'] == self.nids[name] )[0]
        if len(n_inds) > 0:
            if loc['node'] == 1:
                loc_ind = n_inds[0]
            else:
                x_inds = np.where(loc['x'] >= self.xs[name][n_inds])[0]
                if len(x_inds) != 0:
                    ind = np.argmax(self.xs[name][n_inds][x_inds])
                    loc_ind = n_inds[x_inds[ind]]
                else:
                    look_further = True
        else:
            look_further = True
        if look_further:
            # if no locs on the same node, then proceed to resp. parent and child nodes
            node = self[loc['node']]
            pnode = node.getParentNode()
            loc_inds = []
            # check parent node
            if pnode != None:
                ploc_ind = self._findLocsToRoot({'node': pnode.index, 'x': 1.}, name,
                                              check_siblings=check_siblings)
                if ploc_ind != None:
                    loc_inds.append(ploc_ind)
            # check other child nodes of parent node
            if pnode != None and check_siblings:
                ocnodes = copy.copy(pnode.getChildNodes())
                ocnodes.remove(node)
            else:
                ocnodes = []
            for cnode in ocnodes:
                cloc_ind = self._findLocsFromRoot({'node': cnode.index, 'x': 0.}, name)
                if cloc_ind != None:
                    loc_inds.append(cloc_ind)
            # get the one that is closest, if they exist
            pl_aux = 1e4
            ind_loc = 0
            for i, l_i in enumerate(loc_inds):
                pl = self.pathLength({'node': loc['node'], 'x': 1.}, self.locs[name][l_i])
                if pl < pl_aux:
                    pl_aux = pl
                    ind_loc = i
            if pl_aux > 0. and len(loc_inds) > 0:
                loc_ind = loc_inds[ind_loc]
            else:
                loc_ind = None
        return loc_ind

    def getNearestNeighbourLocinds(self, loc, locarg):
        """
        Search nearest neighbours to `loc` in `locarg`.

        Parameters
        ----------
        loc: tuple, dict or `neat.MorphLoc`
            The locations for which nearest neighbours have to be found
        locarg: str or list of locs
            See documentation of `MorphTree._parseLocArg`, the set of locations
            within which to look for nearest neighbours

        Returns
        -------
        list of ints
            Indices of nearest neighbours of `loc` in `locarg`
        """
        # preprocess locarg
        loc = MorphLoc(loc, self)
        if isinstance(locarg, str):
            name = locarg
            locs = self._parseLocArg(locarg)
        else:
            name = 'nn aux'
            locs = locarg
            self.storeLocs(locs, name=name)

        nns = []
        # search for nearest neighbours
        node = self[loc['node']]
        locinds_aux = np.where(node.index == self.nids[name])[0]
        if len(locinds_aux) > 0:
            dx = self.xs[name][locinds_aux] - loc['x']
            # locs on node in down direction
            inds_down = np.where(dx >= 0)[0]
            if len(inds_down) > 0:
                ind_aux = np.argmin(dx[inds_down])
                nns.append(locinds_aux[inds_down][ind_aux])
            else:
                for c_node in node.child_nodes:
                    self._searchNNFromRoot(c_node, nns, name)
            # locs on node in up direction
            inds_up = np.where(dx <= 0)[0]
            if len(inds_up) > 0:
                ind_aux = np.argmax(dx[inds_up])
                nns.append(locinds_aux[inds_up][ind_aux])
            else:
                self._searchNNToRoot(node, nns, name)
        else:
            for c_node in node.child_nodes:
                self._searchNNFromRoot(c_node, nns, name)
            self._searchNNToRoot(node, nns, name)

        if name == 'nn aux':
            self.removeLocs(name)

        return list(set(nns))

    def _searchNNToRoot(self, node, nns, name):
        p_node = node.parent_node
        if p_node is not None:
            # up direction
            locinds_aux = np.where(p_node.index == self.nids[name])[0]
            xval = 0.
            if len(locinds_aux) > 0:
                ind_aux = np.argmax(self.xs[name][locinds_aux])
                locind = locinds_aux[ind_aux]
                nns.append(locind)
                xval = self.xs[name][locind]
            else:
                self._searchNNToRoot(p_node, nns, name)
            # down direction
            if xval < 1.-1e-5:
                for c_node in set(p_node.child_nodes) - {node}:
                    self._searchNNFromRoot(c_node, nns, name)

    def _searchNNFromRoot(self, node, nns, name):
        locinds_aux = np.where(node.index == self.nids[name])[0]
        if len(locinds_aux) > 0:
            ind_aux = np.argmin(self.xs[name][locinds_aux])
            locind = locinds_aux[ind_aux]
            nns.append(locind)
        else:
            for c_node in node.child_nodes:
                    self._searchNNFromRoot(c_node, nns, name)

    def getLeafLocinds(self, name, recompute=False):
        """
        Find the indices in the desire location list that are 'leafs', i.e.
        locations for which no other location exist that is farther from the
        root

        Parameters
        ----------
            name: string
                name of the desired set of locations
            recompute: bool (optional, default ``False``)
                whether or not to force recomputing the distances

        Returns
        -------
            list of inds
                the indices of the 'leaf' locations
        """
        try:
            if recompute:
                raise KeyError
            self.leafinds[name]
        except KeyError:
            self._tryName(name)
            self.leafinds[name] = []
            locs = self.locs[name]
            for ind, loc in enumerate(locs):
                if not self._hasLocFromRoot(loc, name):
                    self.leafinds[name].append(ind)
        return self.leafinds[name]

    def _hasLocFromRoot(self, loc, name):
        look_further = False
        # look if there are locs on the same node
        if loc['node'] != 1:
            n_inds = np.where(loc['node'] == self.nids[name] )[0]
            if len(n_inds) > 0:
                x_inds = np.where(loc['x'] < self.xs[name][n_inds])[0]
                if len(x_inds) > 0:
                    returnbool = True
                else:
                    look_further = True
            else:
                look_further = True
        else:
            look_further = True
        # if no locs on the same node, then proceed to child nodes
        if look_further:
            node = self[loc['node']]
            cnodes = node.child_nodes
            returnbool = False
            for cnode in cnodes:
                if self._hasLocFromRoot({'node': cnode.index, 'x': 0.}, name):
                    returnbool = True
        return returnbool

    def distancesToSoma(self, locarg, recompute=False):
        """
        Compute the distance of each location in a given set to the soma

        Parameters
        ----------
        locarg: list of locations or string
            if list of locations, specifies the locations, if str,
            specifies the name under which the set of location is stored
            that should be used to create the new tree

        Returns
        -------
            np.array of float
                the distances to the soma of the corresponding locations
            recompute: bool (optional)
                whether or not to force recomputing the distances
        """
        # process input argument
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
            recompute = True
            save = False
        elif isinstance(locarg, str):
            name = locarg
            self._tryName(name)
            locs = self.getLocs(name)
            recompute = not (name in self.d2s) or recompute
            save = True
        else:
            raise IOError('`locarg` should be list of locs or string')

        if recompute:
            d2s = np.array([self.pathLength({'node': 1, 'x': 0.}, loc) \
                                        for loc in locs])
        else:
            d2s = self.d2s[name]

        if save:
            self.d2s[name] = d2s

        return d2s

    def distancesToBifurcation(self, name, recompute=False):
        """
        Compute the distance of each location to the nearest bifurcation in
        the 'up' direction (towards root)

        Parameters
        ----------
        name: str
            name of the set of locations
        recompute: bool (optional, default ``False``)
            whether or not to force recomputing the distances

        Returns
        -------
        np.array of floats
            the distances to the nearest bifurcation of the corresponding
            locations
        """
        try:
            if recompute:
                raise KeyError
            return self.d2b[name]
        except KeyError:
            self._tryName(name)
            self.d2b[name] = []
            locs = self.locs[name]
            for i, loc in enumerate(locs):
                if loc['node'] != 1:
                    if loc['node'] != locs[i-1]['node']:
                        node = self[loc['node']]
                        bnode, _ = self.bifurcationNodeToRoot(node)
                    self.d2b[name].append(self.pathLength( \
                                          {'node': bnode.index, 'x': 1.}, loc))
                else:
                    self.d2b[name].append(0.)
            return self.d2b[name]

    def distributeLocsOnNodes(self, d2s, node_arg=None, name='dont save'):
        """
        Distributes locs on a given set of nodes at specified distances from the
        soma. If the specified distances are on the specified nodes, the list
        of locations will be empty. The locations are stored if the name is set
        to be something other than 'dont save'. On each node, locations are
        ordered from low to high x-values.

        Parameters
        ----------
            d2s: numpy.array of floats
                the distances from the soma at which to put the locations (micron)
            node_arg:
                see documentation of `MorphTree._convertNodeArgToNodes`
            name: string
                the name under which the locations are stored. Defaults to 'dont save'
                which means the locations are not stored

        Returns
        -------
            list of `neat.MorphLoc`
                the list of locations
        """
        # distribute the locations
        locs = []
        for node in self._convertNodeArgToNodes(node_arg):
            if node.parent_node != None:
                L0 = self.pathLength({'node': 1, 'x': 0.5},
                                      {'node': node.index, 'x': 0.})
                L1 = self.pathLength({'node': 1, 'x': 0.5},
                                      {'node': node.index, 'x': 1.})
                inds = np.where(np.logical_and(L0 < d2s, d2s <= L1))[0]
                Ls = np.sort(d2s[inds])
                locs.extend([MorphLoc((node.index, (L-L0)/(L1-L0)), self) \
                                        for L in Ls if L > 1e-12])
            elif np.any(np.abs(d2s) <= 1e-12):
                # node is soma, append a location on the soma
                locs.append(MorphLoc((node.index, 0.5), self))
        if name != 'dont save': self.storeLocs(locs, name=name)
        return locs

    @computationalTreetypeDecorator
    def distributeLocsUniform(self, dx, node_arg=None, add_bifurcations=False,
                              name='dont save'):
        """
        Distributes locations as uniform as possible, i.e. for a given distance
        between locations `dx`, locations are distributed equidistantly on each
        given node in the computational tree and their amount is computed
        so that the distance in between them is as close to `dx` as possible.
        Depth-first ordering.

        Parameters
        ----------
            dx: float (> 0)
                target distance in micron between the locations
            node_arg:
                see documentation of `MorphTree._convertNodeArgToNodes`
            add_bifurcations: bool
                whether to ensure that all bifurcation nodes are added
            name: string
                the name under which the locations are stored. Defaults to 'dont save'
                which means the locations are not stored

        Returns
        -------
            list of `neat.MorphLoc`
                the list of locations
        """
        assert dx > 0
        # distribute the locations
        locs = []
        ii = 0
        for node in self._convertNodeArgToNodes(node_arg):
            ii += 1
            if node.parent_node == None:
                locs.append(MorphLoc((node.index, 0.5), self,
                                     set_as_comploc=True))
            else:
                Nloc = np.round(node.L / dx)
                if Nloc == 0 and len(node.child_nodes) > 1 and add_bifurcations:
                    Nloc = 1
                xvals = np.arange(1, Nloc+1) / float(Nloc)
                locs.extend([
                    MorphLoc(
                        (node.index, xv), self, set_as_comploc=True
                    ) for xv in xvals
                ])
        if name != 'dont save': self.storeLocs(locs, name=name)
        return locs


    def distributeLocsRandom(self, num, dx=0.001, node_arg=None,
                             add_soma=True, name='dont save', seed=None):
        """
        Returns a list of input locations randomly distributed on the tree

        Parameters
        ----------
        num: int
            number of inputs
        dx: float (optional)
            minimal or given distance between input locations (micron)
        node_arg (optional):
            see documentation of `MorphTree._convertNodeArgToNodes`
        add_soma: bool (optional)
            whether or not to include the possibility of adding locations on the
            soma
        name: string (optional)
            the name under which the locations are stored. Defaults to 'dont save'
            which means the locations are not stored
        seed: int (optiona)
            Seed for numpy random number generator

        Returns
        -------
        list of `neat.MorphLoc`
            the locations
        """
        np.random.seed(seed)
        # use the requested subset of nodes
        nodes = [node for node in self._convertNodeArgToNodes(node_arg)
                 if node.index != 1]
        # initialize the loclist with or without soma
        if add_soma:
            locs = [MorphLoc({'node': 1, 'x': 0.}, self)]
            self.root.content['tag'] = 1
        else:
            locs = []
        # add the nodes
        for ii in range(num):
            nodes_left = [node.index for node in nodes
                            if 'tag' not in node.content]
            if len(nodes_left) < 1:
                break
            index = np.random.choice(nodes_left)
            x = np.random.random()
            locs.append(MorphLoc((index, x), self))
            node = self[index]
            self._tagNodesFromRoot(node, node, dx=dx)
            self._tagNodesToRoot(node, node, dx=dx)
        self._removeTags()
        # store the locations
        if name != 'dont save': self.storeLocs(locs, name=name)
        return locs

    def _tagNodesFromRoot(self, start_node, node, dx=0.001):
        if 'tag' not in node.content:
            if node.index == start_node.index:
                length = 0.
            else:
                length = self.pathLength({'node': start_node.index, 'x': 1.},
                                          {'node': node.index, 'x': 0.})
            if length < dx:
                node.content['tag'] = 1
                for cnode in node.child_nodes:
                    self._tagNodesFromRoot(start_node, cnode, dx=dx)

    def _tagNodesToRoot(self, start_node, node, cnode=None, dx=0.001):
        if node.index == start_node.index:
            length = 0.
        else:
            length = self.pathLength({'node': start_node.index, 'x': 1.},
                                      {'node': node.index, 'x': 1.})
        if length < dx:
            node.content['tag'] = 1
            cnodes = node.child_nodes
            if len(cnodes) > 1:
                if cnode != None:
                    cnodes = list(set(cnodes) - set([cnode]))
                for cn in cnodes:
                    self._tagNodesFromRoot(start_node, cn, dx=dx)
            pnode = node.getParentNode()
            if pnode != None:
                self._tagNodesToRoot(start_node, pnode, node, dx=dx)

    def _removeTags(self):
        for node in self:
            if 'tag' in node.content:
                del node.content['tag']

    def _parseLocArg(self, loc_arg):
        if isinstance(loc_arg, list):
            locs = [MorphLoc(loc, self) for loc in loc_arg]
        elif isinstance(loc_arg, str):
            self._tryName(loc_arg)
            locs = self.getLocs(loc_arg)
        else:
            raise IOError('invalid type for `loc_arg`, should be list or string')
        return locs

    def extendWithBifurcationLocs(self, loc_arg, name='dont save'):
        """
        Extends input loc_arg with the intermediate bifurcations. They are
        appended to the end of the list

        Parameters
        ----------
        loc_arg: list of `neat.MorphLoc` or string
            the locations
        name: string (optional)
            The name under which the list of bifurcation locs will be stored.
            Defaults to 'dont save' which means they are not stored.

        Returns
        -------
        list of `neat.MorphLoc`
            the bifurcation locs
        """
        locs = self._parseLocArg(loc_arg)
        # get the bifurcation locs
        nodes = [self[loc['node']] for loc in locs]
        bnodes = self.getBifurcationNodes(nodes)
        blocs = [MorphLoc((bnode.index, 1.), self) for bnode in bnodes]
        # retain unique locs
        all_locs = self.uniqueLocs(locs + blocs)
        # store the locations
        if name != 'dont save': self.storeLocs(all_locs, name=name)
        return all_locs

    def uniqueLocs(self, loc_arg, name='dont save'):
        """
        Gets the unique locations in the provided locs

        Parameters
        ----------
        loc_arg: list of `neat.MorphLoc` or string
            the locations
        name: string (optional)
            The name under which the list of bifurcation locs will be stored.
            Defaults to 'dont save' which means they are not stored.

        Returns
        -------
        list of `neat.MorphLoc`
            the bifurcation locs
        """
        locs = self._parseLocArg(loc_arg)
        locs_ = reduce(lambda l, x: l.append(x) or l if x not in l else l, locs, [])

        if name != 'dont save': self.storeLocs(locs_, name=name)
        return locs_

    def makeXAxis(self, dx=10., node_arg=None, loc_arg=None):
        """
        Create a set of locs suitable for serving as the x-axis for 1D plotting.
        The neurons is put on a 1D axis with a depth-first ordering.

        Parameters
        ----------
        dx: float
            target separation between the plot points (micron)
        node_arg:
            see documentation of `MorphTree._convertNodeArgToNodes`
            The nodes on which the locations for the x-axis are distributed.
            When this is given as a list of nodes, assumes a depth first
            ordering.
        loc_arg: list of locs or string
            if list of locs, these locs will be used as x-axis, if string, name
            of set of locs on the morphology that will be used as x-axis

        """
        if loc_arg is None:
            # if comptree has not been set, create a basic one for plotting
            if self._computational_root == None:
                self.setCompTree()
            # distribute the x-axis locations
            self.distributeLocsUniform(dx, node_arg=node_arg, name='xaxis')
            # get the root node
            nodes = self._convertNodeArgToNodes(node_arg)
            # check that first node is root
            for node in nodes:
                if nodes[0] in node.child_nodes:
                    raise ValueError('Input `node_arg` is not a depth-first ordered'
                                     ' list of nodes.')
            # set the node colors for both trees
            if self.treetype == 'original':
                rootnode_orig = nodes[0]
                tempnode = self._findCompnodeFromRoot(nodes[0])
                self.setNodeColors(rootnode_orig)
                self.treetype = 'computational'
                rootnode_comp = self[tempnode.index]
                self.setNodeColors(rootnode_comp)
                self.treetype = 'original'
            else:
                rootnode_comp = nodes[0]
                self.setNodeColors(rootnode_comp)
                self.treetype = 'original'
                rootnode_orig = self[rootnode.comp.index]
                self.setNodeColors(rootnode_orig)
                self.treetype = 'computational'
        else:
            if isinstance(loc_arg, list):
                self.storeLocs(loc_arg, name='xaxis')
            elif isinstance(loc_arg, str):
                self.storeLocs(self.getLocs(loc_arg), name='xaxis')
            else:
                raise IOError('`loc_org` should be string or list of locs')
        # compute the x-axis 1D array
        pinds = self.getLeafLocinds('xaxis')
        d2s = self.distancesToSoma('xaxis')
        xaxis = d2s[0:pinds[0]+1].tolist()
        d_add = d2s[pinds[0]]
        for ii in range(0,len(pinds)-1):
            xaxis.extend((d_add + d2s[pinds[ii]+1:pinds[ii+1]+1] \
                                - d2s[pinds[ii]+1]).tolist())
            d_add += d2s[pinds[ii+1]] - d2s[pinds[ii]+1]
        self.xaxis = np.array(xaxis)

    def setNodeColors(self, startnode=None):
        """
        Set the color code for the nodes for 1D plotting

        Parameters
        ----------
            node: int or `neat.MorphNode`
                index of the node or node whose subtree will be colored. Defaults
                to the root
        """
        if startnode == None: startnode = self.root
        for node in self: node.content['color'] = 0.
        self.node_color = [0.] # trick to pass the pointer and not the number itself
        self._setColorsFromRoot(startnode)

    def _setColorsFromRoot(self, node):
        node.content['color'] = self.node_color[0]
        if self.isLeaf(node):
            self.node_color[0] += 1.
        for cnode in node.child_nodes:
            self._setColorsFromRoot(cnode)

    def getXValues(self, locs):
        """
        Get the corresponding location on the x-axis of the input locations

        Parameters
        ----------
            locs: list of tuples, dicts or `neat.MorphLoc`
                list of the locations
        """
        locinds = np.array(self.getNearestLocinds(locs, 'xaxis')).astype(int)
        return self.xaxis[locinds]

    def plot1D(self, ax, parr, *args, **kwargs):
        """
        Plot an array where each element corresponds to the matching location on
        the x-axis with a depth-first ordering on a 1D plot

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be made
            parr: numpy.array of floats
                the array that will be plotted
            args, kwargs:
                arguments for `matplotlib.pyplot.plot`

        Returns
        -------
            lines: list of `matplotlib.lines.Line2D` instances
                the line segments corresponding to the value of the plotted array
                in each branch

        Raises
        ------
            AssertionError
                When the number of elements in the data array in not equal to
                the number of elements on the x-axis
        """
        assert len(parr) == len(self.locs['xaxis'])
        pinds = self.getLeafLocinds('xaxis')
        d2s = self.distancesToSoma('xaxis')
        # make the plot
        lines = []

        line = ax.plot(self.xaxis[0:pinds[0]+1], parr[0:pinds[0]+1],
                       *args, **kwargs)
        lines.append(line[0])
        if 'label' in list(kwargs.keys()):
            kwargs = copy.deepcopy(kwargs)
            del kwargs['label']
        for ii in range(0,len(pinds)-1):
            line = ax.plot(self.xaxis[pinds[ii]+1:pinds[ii+1]+1],
                            parr[pinds[ii]+1:pinds[ii+1]+1],
                            *args, **kwargs)
            lines.append(line[0])
        return lines

    def setLineData(self, lines, parr):
        """
        Update the line objects with new data

        Parameters
        ----------
            lines: list of `matplotlib.lines.Line2D` instance
                the line segments of which the data has to be updated
            parr: numpy.array of floats
                the array that will be put in the line segments

        Raises
        ------
            AssertionError
                When the number of elements in the data array in not equal to
                the number of elements on the x-axis
        """
        assert len(parr) == len(self.locs['xaxis'])
        pinds = self.getLeafLocinds('xaxis')
        d2s = self.distancesToSoma('xaxis')
        lines[0].set_data(self.xaxis[0:pinds[0]+1], parr[0:pinds[0]+1])
        for ii in range(0,len(pinds)-1):
            ll = ii+1
            lines[ll].set_data(self.xaxis[pinds[ii]+1:pinds[ii+1]+1],
                                parr[pinds[ii]+1:pinds[ii+1]+1])

    def plotTrueD2S(self, ax, parr, cmap=None, **kwargs):
        """
        Plot an array where each element corresponds to the matching location in
        the x-axis location list. Now all locations are plotted at their true
        distance from the soma.

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be made
            parr: numpy.array of floats
                the array that will be plotted
            cmap: `matplotlib.colors.Colormap` instance
                If provided, the lines will be colored according to the branch
                to which they belong, in colors specified by the colormap
            kwargs:
                keyword arguments for `matplotlib.pyplot.plot`

        Returns
        -------
            lines
            lines: list of `matplotlib.lines.Line2D`
                the line segments corresponding to the value of the plotted array
                in each branch

        Raises
        ------
            AssertionError
                When the number of elements in the data array in not equal to
                the number of elements on the x-axis
        """
        assert len(parr) == len(self.locs['xaxis'])
        locs = self.locs['xaxis']
        pinds = self.getLeafLocinds('xaxis')
        d2s = self.distancesToSoma('xaxis')
        # list of colors for plotting
        cs = {node.index: node.content['color'] for node in self}
        cplot = [cs[loc['node']] for loc in locs]
        max_cs = max(cplot)
        min_cs = min(cplot)
        if np.abs(max_cs - min_cs) < 1e-12:
            norm_cs = max_cs + 1e-2
        else:
            norm_cs = (max_cs - min_cs) * (1. + 1./100.)
        # create the truespace plot
        lines = []
        if cmap != None:
            kwargs['c'] = cmap((cplot[0]-min_cs)/norm_cs)
            if 'color' in kwargs: del kwargs['color']
        line = ax.plot(d2s[0:pinds[0]+1], parr[0:pinds[0]+1], **kwargs)
        lines.append(line[0])
        if 'label' in kwargs: del kwargs['label']
        for ii in range(0,len(pinds)-1):
            if cmap != None:
                kwargs['c'] = cmap((cs[locs[pinds[ii]+1]['node']]-min_cs)/norm_cs)
            line = ax.plot(d2s[pinds[ii]+1:pinds[ii+1]+1],
                           parr[pinds[ii]+1:pinds[ii+1]+1],
                           **kwargs)
            lines.append(line[0])
        return lines

    def _addScalebar(self, ax, borderpad=-1.8, sep=2):
        from neat.tools.plottools import scalebars
        scalebars.addScalebar(ax, hidex=False, hidey=False, matchy=False,
                                    labelx='$\mu$m',
                                    loc=8, borderpad=borderpad, sep=sep)
        ax.set_xticklabels([])

    def colorXAxis(self, ax, cmap, addScalebar=1, borderpad=-1.8):
        """
        Color the x-axis of a plot according to the morphology.

        !!! Has to be called after all lines are plotted !!!

        Furthermor, node colors have to be set first. This can be done with
        `MorphTree.setNodeColors()` or manually by adding a 'color' entry
        to the ``MorphNode.content`` dictionary

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object of which the x-axis will be colored
            cmap: `matplotlib.colors.Colormap` instance
                Colormap that determines the color of each branch
            sizex: float
                Size of scalebar (in micron). If set to None, no scalebar is
                plotted.
            borderpad: float
                Borderpad of scalebar
        """
        locs = self.locs['xaxis']
        # list of colors for plotting
        cs = {node.index: node.content['color'] for node in self}
        cplot = [cs[loc['node']] for loc in locs]
        max_cs = max(cplot)
        min_cs = min(cplot)
        if np.abs(max_cs - min_cs) < 1e-12:
            norm_cs = max_cs + 1e-2
        else:
            norm_cs = (max_cs - min_cs) * (1. + 1./100.)
        # necessary distance arrays
        pinds = self.getLeafLocinds('xaxis')
        assert len(pinds) > 0
        d2s = self.distancesToSoma('xaxis')
        # plot colored xaxis
        ylim = np.array(ax.get_ylim())
        ax.plot(self.xaxis[0:pinds[0]+1], [ylim[0]+1e-9 for _ in d2s[0:pinds[0]+1]],
                                    c=cmap((cplot[0]-min_cs)/norm_cs), lw=10)
        for ii in range(0,len(pinds)-1):
            if locs[pinds[ii]+1]['node'] in list(cs.keys()):
                ax.plot(self.xaxis[pinds[ii]+1:pinds[ii+1]+1],
                        [ylim[0]+1e-9 for _ in d2s[pinds[ii]+1:pinds[ii+1]+1]],
                        c=cmap((cs[locs[pinds[ii]+1]['node']]-min_cs)/norm_cs), lw=10)
            else:
                ax.plot(self.xaxis[pinds[ii]+1:pinds[ii+1]+1],
                        [ylim[0]+1e-9 for _ in d2s[pinds[ii]+1:pinds[ii+1]+1]],
                        c='k', lw=10)
        ax.set_ylim((ylim[0], ylim[1]))
        # add scalebar
        if addScalebar:
            self._addScalebar(ax, borderpad=borderpad)
        ax.axes.get_xaxis().set_visible(False)

    def plot2DMorphology(self, ax, node_arg=None, cs=None, cminmax=None, cmap=None,
                            use_radius=1, draw_soma_circle=1,
                            plotargs={}, textargs={},
                            marklocs=[], locargs={},
                            marklabels={}, labelargs={},
                            cb_draw=0, cb_orientation='vertical', cb_label='',
                            sb_draw=1, sb_scale=100, sb_width=5.,
                            set_lims=True, lims_margin=.1):
        """
        Plot the morphology projected on the x,y-plane

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be drawn
            node_arg:
                see documentation of `MorphTree._convertNodeArgToNodes`
            cs: dict {int: float}, None or 'x_color'
                If dict, node indices are keys and the float value will
                correspond to the plotted color. If None, the color of the tree
                will be the one specified in ``plotargs``. Note that the dict
                does not have to contain all node indices. The ones that are not
                featured in the dict are plot in the color specified in ``plotargs``.
                If 'node_color', colors will be those stored on the nodes. Note
                that choosing this option when there are nodes without 'color'
                as an entry in ``node.content`` will result in an error. Node
                colors can be set with `MorphTree.setNodeColor()``
            cminmax: (float, float) or None (default)
                The min and max values of the color scale (if cs is provided).
                If None, the min and max values of cs are used.
            cmap: `matplotlib.colors.Colormap` instance
                colormap fram which colors in ``cs`` are taken
            use_radius: bool
                If ``True``, uses the swc radius for the width of the line
                segments
            draw_soma_circle: bool
                If ``True``, draws the soma as a circle, otherwise doesn't draw
                soma
            plotargs: dict
                `kwargs` for `matplotlib.pyplot.plot`. 'c'- or 'color'-
                argument will be overwritten when cs is defined. 'lw'- or
                'linewidth' argument will be multiplied with the swc radius of
                the node if `use_radius` is ``True``.
            textargs: dict
                text properties for various labels in the plot
            marklocs: list of tuples, dicts or instances of `neat.MorphLoc`
                Location that will be plotted on the morphology
            locargs: dict or list of dict
                `kwargs` for `matplotlib.pyplot.plot` for the location.
                Use only point markers and no lines! When it is a single dict
                all location will have the same marker. When it is a list it
                should have the same length as `marklocs`.
            marklabels: dict {int: string}
                Keys are indices of locations in `marklocs`, values are strings
                that are used to annotate the corresponding locations
            labelargs: dict
                text properties for the location annotation
            cb_draw: bool
                Whether or not to draw a `matplotlib.pyplot.colorbar()`
                instance.
            cb_orientation: string, 'vertical' or 'horizontal'
                The colorbars' orientation
            cb_label: string
                The label of the colorbar
            sb_draw: bool
                Whether or not to draw a scale bar
            sb_scale: float
                Lenght of the scale bar (micron)
            sb_width: float
                Width of the scale bar
            set_lims: bool (optional, default ``True``)
                set ``ax`` limits based on the morphology
            lims_margin: float
                the margin, as fraction of total width and height of tree, at
                which the limits are placed
        """
        # default cmap
        if cmap is None:
            cmap = cm.get_cmap('jet')
        # ensure color is indicated by the 'c'-parameter in `plotargs`
        if 'color' in plotargs:
            plotargs['c'] = plotargs['color']
            del plotargs['color']
        elif 'c' not in plotargs:
            plotargs['c'] = 'k'
        # define a norm for the colors, if defined
        if cs == 'x_color':
            cs = {node.index: node.content['color'] for node in self}
        if cs is not None:
            if cminmax is None:
                if len(cs) > 0:
                    max_cs = cs[max(cs, key=cs.__getitem__)] # works for dict and list
                    min_cs = cs[min(cs, key=cs.__getitem__)] # works for dict and list
                else:
                    min_cs, max_cs = 0., 1.
            else:
                min_cs = cminmax[0]
                max_cs = cminmax[1]
            norm = pl.Normalize(vmin=min_cs, vmax=max_cs)
        # ensure linewidth is indicated as 'lw' in plotargs
        if 'linewidth' in plotargs:
            plotargs['lw'] = plotargs['linewidth']
            del plotargs['linewidth']
        elif 'lw' not in plotargs:
            plotargs['lw'] = 1.
        plotargs_orig = copy.deepcopy(plotargs)
        # locargs can be dictionary, so that the same properties hold for every
        # markloc, or can be list with the same size as marklocs, so that every
        # marker has different properties. `zorder` of the markers is also set
        # very high so that they are always in the foreground
        self.storeLocs(marklocs, 'plotlocs')
        xs = self.xs['plotlocs']
        if type(locargs) == dict:
            if 'zorder' not in locargs:
                locargs['zorder'] = 1e4
                locargs = [locargs for _ in marklocs]
        else:
            assert len(locargs) == len(marklocs)
            for locarg in locargs:
                if 'zorder' not in locarg:
                    locarg['zorder'] = 1e4
        # `marklabels` is a dictionary with as keys the index of the loc in
        # `marklocs` to which the label belongs. `labelargs` is the same for
        # every label
        for ind in marklabels: assert ind < len(marklocs)
        # plot the tree
        xlim = [0.,0.]; ylim = [0.,0.]
        for node in self._convertNodeArgToNodes(node_arg):
            if node.xyz[0] < xlim[0]: xlim[0] = node.xyz[0]
            if node.xyz[0] > xlim[1]: xlim[1] = node.xyz[0]
            if node.xyz[1] < ylim[0]: ylim[0] = node.xyz[1]
            if node.xyz[1] > ylim[1]: ylim[1] = node.xyz[1]
            # find the locations that are on the current node
            inds = self.getLocindsOnNode('plotlocs', node)
            if node.parent_node is None:
                # node is soma, draw as circle if necessary
                if draw_soma_circle:
                    if cs is not None and node.index in cs:
                        plotargs['c'] = cmap(norm(cs[node.index]))
                    else:
                        plotargs['c'] = plotargs_orig['c']
                    circ = patches.Circle(node.xyz[0:2], node.R,
                                          color=plotargs['c'])
                    ax.add_patch(circ)
                for ind in inds:
                    self._plotLoc(ax, ind, node.xyz[0], node.xyz[1],
                                   locargs, marklabels, labelargs)
            else:
                # plot line segment associated with node
                nxyz = node.xyz; pxyz = node.parent_node.xyz
                if cs is not None and node.index in cs:
                    plotargs['c'] = cmap(norm(cs[node.index]))
                else:
                    plotargs['c'] = plotargs_orig['c']
                if use_radius:
                    plotargs['lw'] = plotargs_orig['lw'] * node.R
                ax.plot([pxyz[0], nxyz[0]], [pxyz[1], nxyz[1]], **plotargs)
                # plot the locations
                for ind in inds:
                    locxyz = pxyz + (nxyz - pxyz) * xs[ind]
                    self._plotLoc(ax, ind, locxyz[0], locxyz[1],
                                   locargs, marklabels, labelargs)
        # margins
        dx = xlim[1]-xlim[0]
        dy = ylim[1]-ylim[0]
        xlim[0] -= dx*lims_margin; xlim[1] += dx*lims_margin
        ylim[0] -= dy*lims_margin; ylim[1] += dy*lims_margin
        # draw a scale bar
        if sb_draw:
            scale = sb_scale
            dy = ylim[1]-ylim[0]
            dx = xlim[1]-xlim[0]
            ax.plot([xlim[0],xlim[0]+scale],
                    [ylim[0],ylim[0]],
                    'k', linewidth=sb_width, zorder=1e5)
            txt = ax.annotate(r'' + str(scale) + ' $\mu$m',
                              xy=(xlim[0]+scale/2., ylim[0]),
                              xycoords='data', xytext=(xlim[0]+scale/2.,ylim[0]-dy/200.), ha='center', va='top',
                              **textargs)
                              # textcoords='offset points', **textargs)
            txt.set_path_effects([patheffects.withStroke(foreground="w",
                                                         linewidth=2)])
        if set_lims:
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect('equal', 'datalim')
        if cs != None and cb_draw:
            # create colorbar ax
            divider = make_axes_locatable(ax)
            if cb_orientation == 'horizontal':
                cax = divider.append_axes("bottom", "5%", pad="3%")
            else:
                cax = divider.append_axes("right", "5%", pad="3%")
            # create a mappable
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = [] # fake array for scalar mappable
            # create the colorbar
            cb = pl.colorbar(sm, cax=cax, orientation=cb_orientation)
            ticks_cb = np.round(np.linspace(min_cs, max_cs, 7), decimals=1)
            cb.set_ticks(ticks_cb)
            if cb_orientation == 'horizontal':
                cb.ax.xaxis.set_ticks_position('bottom')
            else:
                cb.ax.yaxis.set_ticks_position('right')
            cb.set_label(cb_label, **textargs)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.draw_frame = False

        ax.set_xticks([])
        ax.set_yticks([])

    def _plotLoc(self, ax, ind, xval, yval, locargs, marklabels, labelargs):
        """
        plot a location on the morphology together with its annotation
        """
        ax.plot(xval, yval, **locargs[ind])
        if ind in marklabels:
            txt = ax.annotate(marklabels[ind], xy=(xval, yval),
                              xycoords='data', xytext=(5,5),
                              textcoords='offset points', **labelargs)
            txt.set_path_effects([patheffects.withStroke(foreground="w",
                                                         linewidth=2)])

    def plotMorphologyInteractive(self, node_arg=None,
                            use_radius=1, draw_soma_circle=1,
                            plotargs={'c': 'k', 'lw': 1.},
                            project3d=False):
        """
        Show the morphology either in 3d or projected on the x,y-plane. When
        a line segment is clicked, the associated node is printed.

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be drawn
            node_arg:
                see documentation of `MorphTree._convertNodeArgToNodes`
            use_radius: bool
                If ``True``, uses the swc radius for the width of the line
                segments
            draw_soma_circle: bool
                If ``True``, draws the soma as a circle, otherwise doesn't draw
                soma
        """
        fig = pl.figure('Morphology interactive')
        ax = pl.gca(projection='3d') if project3d else pl.gca()
        # ax = pl.gca()
        if 'c' not in plotargs:
            plotargs.update({'c': 'k'})
        if 'linewidth' in plotargs:
            plotargs['lw'] = plotargs['linewidth']
            del plotargs['linewidth']
        if 'lw' not in plotargs:
            plotargs.update({'lw': 'k'})
        plotargs_orig = copy.deepcopy(plotargs)
        # plot the tree
        node_line_associators = {}
        for ii, node in enumerate(self._convertNodeArgToNodes(node_arg)):
            if node.parent_node is not None:
                # plot line segment associated with node
                nxyz = node.xyz; pxyz = node.parent_node.xyz
                if use_radius:
                    plotargs['lw'] = plotargs_orig['lw'] * node.R
                if project3d:
                    line = ax.plot([pxyz[0], nxyz[0]],
                                   [pxyz[1], nxyz[1]],
                                   [pxyz[2], nxyz[2]],
                                   label=str(ii), picker=2., **plotargs)
                else:
                    line = ax.plot([pxyz[0], nxyz[0]],
                                   [pxyz[1], nxyz[1]],
                                   label=str(ii), picker=2., **plotargs)
                node_line_associators.update({str(ii): node})
        ax.axes.get_xaxis().set_visible(0)
        ax.axes.get_yaxis().set_visible(0)
        ax.axison = 0

        # define the clickevent action
        def onPick(event):
            line = event.artist
            node = node_line_associators[line.get_label()]
            # print the associated node
            print('\n>>> line segment at ' + str(node) + \
                   ', distance to soma (um) = ' + \
                   str(self.pathLength({'node': node.index, 'x': 1},
                                       {'node': 1, 'x':0.})))
        # show morphology
        cid = fig.canvas.mpl_connect('pick_event', onPick)
        pl.show()

    @originalTreetypeDecorator
    def findCommonRoot(self, name):
        self._tryName(name)
        # get the node indices of nodes
        node_inds = self.getNodeIndices(name)
        # find the paths to the root
        paths = [set(self.pathToRoot(self[node_ind])) for node_ind in node_inds]
        # possible roots
        roots = list(set.intersection(*paths))
        # return the node of highest order
        rootind = np.argmax([self.orderOfNode(node) for node in roots])
        return roots[rootind]

    @originalTreetypeDecorator
    def createNewTree(self, name, fake_soma=False, store_loc_inds=False):
        """
        Creates a new tree where the locs of a given 'name' are now the nodes.
        Distance relations between locations are maintained (note that this
        relation is stored in `L` attribute of `neat.MorphNode`, using the `p3d`
        attribute containing the 3d coordinates does not maintain distances)

        Parameters
        ----------
            name: string
                the name under which the locations are stored that should be
                used to create the new tree
            fake_soma: bool (default `False`)
                if `True`, finds the common root of the set of locations and
                uses that as the soma of the new tree. If `False`, the real soma
                is used.
            store_loc_inds: bool (default `False`)
                store the index of each location in the `content` attribute of the
                new node (under the key 'loc ind')

        Returns
        -------
            `neat.MorphTree`
                The new tree.
        """
        self._tryName(name)
        nids = self.getNodeIndices(name)
        # create new tree
        new_tree = self.__class__()
        if fake_soma:
            # find the common root of the set of locations
            snode = self.findCommonRoot(name)
        else:
            # use the soma as root
            snode = self[1]
        p3d = (snode.xyz, snode.R, snode.swc_type)
        new_snode = new_tree._createCorrespondingNode(1, p3d)
        new_snode.L = snode.L
        new_tree.setRoot(new_snode)
        new_nodes = [new_snode]
        if store_loc_inds:
            new_snode.content['loc ind'] = None if 1 not in nids else \
                                           np.where(nids == 1)[0][0]
        # make two other soma nodes
        if fake_soma:
            for index in [2,3]:
                new_cnode = new_tree._createCorrespondingNode(index, p3d)
                new_tree.addNodeWithParent(new_cnode, new_snode)
                new_nodes.append(new_cnode)
        else:
            for cnode in snode.getChildNodes(skip_inds=[]):
                if cnode.index in [2,3]:
                    p3d = (cnode.xyz, cnode.R, cnode.swc_type)
                    new_cnode = new_tree._createCorrespondingNode(cnode.index, p3d)
                    new_tree.addNodeWithParent(new_cnode, new_snode)
                    new_nodes.append(new_cnode)
        # make rest of tree
        for cnode in snode.child_nodes:
            self._addNodesToTree(cnode, new_snode, new_tree, new_nodes, name,
                                 store_loc_inds=store_loc_inds)
        # set the lengths of the nodes
        for new_node in new_tree:
            if new_node.parent_node != None:
                L = np.sqrt(np.sum((new_node.parent_node.xyz - new_node.xyz)**2))
            else:
                L = 0.
            new_node.setLength(L)

        return new_tree

    def _addNodesToTree(self, node, new_pnode, new_tree, new_nodes, name,
                              store_loc_inds=False):
        # get the specified locs
        xs = self.xs[name]
        # check which locinds are on the branch
        ninds = self.getLocindsOnNode(name, node)
        order_inds = np.argsort(xs[ninds])
        for ind in np.array(ninds)[order_inds]:
            index = len(new_nodes) + 1
            # new coordinates
            new_xyz = node.parent_node.xyz * (1.-xs[ind]) + node.xyz * xs[ind]
            if node.parent_node.index == 1:
                new_radius = node.R
            else:
                new_radius = node.parent_node.R * (1.-xs[ind]) + node.R * xs[ind]
            # make new node
            p3d = (new_xyz, new_radius, node.swc_type)
            new_node = new_tree._createCorrespondingNode(index, p3d)
            if store_loc_inds:
                new_node.content['loc ind'] = ind
            # add new node
            new_tree.addNodeWithParent(new_node, new_pnode)
            new_nodes.append(new_node)
            # set new node as next parent node
            new_pnode = new_node
        # continue with the children
        for cnode in node.child_nodes:
            self._addNodesToTree(cnode, new_pnode, new_tree, new_nodes, name,
                                 store_loc_inds=store_loc_inds)

    @originalTreetypeDecorator
    def createCompartmentTree(self, locarg):
        """
        Creates a new compartment tree where the provided set of locations
        correspond to the nodes.

        Parameters
        ----------
        locarg: list of locations or str
            if list of locations, specifies the locations, if str,
            specifies the name under which the set of location is stored
            that should be used to create the new tree

        Returns
        -------
            `neat.MorphTree`
                The new tree.
        """
        # process input argument
        if isinstance(locarg, list):
            locs = [MorphLoc(loc, self) for loc in locarg]
            name = 'comp_locs'
            self.storeLocs(locs, name=name)
        elif isinstance(locarg, str):
            name = locarg
            self._tryName(name)
        else:
            raise IOError('`locarg` should be list of locs or string')
        nids = self.nids[name]
        xs = self.xs[name]
        # create new tree
        new_tree = CompartmentTree()
        # find the common root of the set of locations
        snode = self.findCommonRoot(name)
        # check if that root is in set of locations
        possible_loc_inds = self.getLocindsOnNode(name, snode)
        if len(possible_loc_inds) > 0:
            # create the new root node
            new_pnode = CompartmentNode(0, loc_ind=possible_loc_inds[0])
            new_tree.setRoot(new_pnode)
            new_nodes = [new_pnode]
            # create other nodes
            for loc_ind in possible_loc_inds[1:]:
                index = len(new_nodes)
                # make new node
                new_node = CompartmentNode(index, loc_ind=loc_ind)
                # add new node
                new_tree.addNodeWithParent(new_node, new_pnode)
                new_nodes.append(new_node)
                # set new node as next parent node
                new_pnode = new_node
        else:
            warnings.warn('Locations of name `' + name + '` do not define a root - ' + \
                          'adding root to set of locations')
            locs = self.getLocs(name)
            locs = [(snode.index, 1.)] + locs
            self.storeLocs(locs, name=name)
            # create the new root node
            new_pnode = CompartmentNode(0, loc_ind=0)
            new_tree.setRoot(new_pnode)
            new_nodes = [new_pnode]
        # make rest of tree
        for cnode in snode.child_nodes:
            self._addCompNodesToTree(cnode, new_pnode, new_tree, new_nodes, name)
        return new_tree

    def _addCompNodesToTree(self, node, new_pnode, new_tree, new_nodes, name):
        # get the specified locs
        xs = self.xs[name]
        # check which locinds are on the branch
        ninds = self.getLocindsOnNode(name, node)
        for loc_ind in ninds:
            index = len(new_nodes)
            # make new node
            new_node = CompartmentNode(index, loc_ind=loc_ind)
            # add new node
            new_tree.addNodeWithParent(new_node, new_pnode)
            new_nodes.append(new_node)
            # set new node as next parent node
            new_pnode = new_node
        # continue with the children
        for cnode in node.child_nodes:
            self._addCompNodesToTree(cnode, new_pnode, new_tree, new_nodes, name)

    def __copy__(self, new_tree=None):
        """
        Fill the ``new_tree`` with it's corresponding nodes in the same
        structure as ``self``, and copies all node variables that both tree
        classes have in common

        Parameters
        ----------
        new_tree: :class:`STree` or derived class (default is ``None``)
            the tree class in which the ``self`` is copied. If ``None``,
            returns a copy of ``self``.

        Returns
        -------
        The new tree instance
        """
        if new_tree is None:
            new_tree = self.__class__()

        current_treetype = self.treetype
        self.treetype = 'original'
        super().__copy__(new_tree=new_tree)
        try:
            # set the computational tree
            self.treetype = 'computational'
            new_node = new_tree._createCorrespondingNode(self.root.index)
            self.root.__copy__(new_node=new_node)
            new_tree._computational_root = new_node
            new_tree.treetype = 'computational'
            self._recurseCopy(self.root, new_tree)
        except ValueError:
            pass
        self.treetype = current_treetype
        new_tree.treetype = current_treetype

        return new_tree
