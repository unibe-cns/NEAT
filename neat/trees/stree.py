"""
File contains:

    - :class:`SNode`
    - :class:`STree`

Authors: B. Torben-Nielsen (legacy code), W. Wybo
"""

import warnings
import copy

class SNode(object):
    '''
    Simple Node for use with a simple Tree (STree)
    By design, the ``content`` attribute should be a dictionary.
    '''

    def __init__(self, index):
        self.index = index
        self._parent_node = None
        self._child_nodes = []
        self.pval = 1 # for tree plotting
        self._content = {}

    def getParentNode(self):
        return self._parent_node

    def setParentNode(self, parent_node):
        self._parent_node = parent_node

    parent_node = property(getParentNode, setParentNode)

    def getChildNodes(self):
        return self._child_nodes

    def setChildNodes(self, cnodes):
        self._child_nodes = cnodes

    child_nodes = property(getChildNodes, setChildNodes)

    def addChild(self,child_node):
        self._child_nodes.append(child_node)

    def getContent(self):
        return self._content

    def setContent(self, content):
        if isinstance(content, dict):
            self._content = content
        else :
            raise Exception("SNode.setContent must receive a dict")

    content = property(getContent, setContent)

    def makeEmpty(self):
        self._parent_node = None
        self._content = None
        self._child_nodes = []

    def removeChild(self, child_node):
        self._child_nodes.remove(child_node)

    def __getitem__(self, key):
        return self.content[key]

    def __str__(self, with_parent=False, with_children=False):
        node_string = 'SNode ' + str(self.index)
        if with_parent:
            node_string += ', Parent: ' + str(self.parent_node)
        if with_children:
            node_string += ', Children:' + \
                            str([str(cnode) for cnode in self.child_nodes])
        return node_string

    def __copy__(self) : # customization of copy.copy
        ret = SNode(self.index)
        for child in self._child_nodes :
            ret.addChild(child)
        try:
            ret.content = self.content
        except AttributeError:
            # no content variable set
            pass
        ret.setParentNode(self._parent_node)
        return ret


class STree(object):
    '''
    A simple tree for use with a simple Node (:class:`SNode`).

    Generic implementation of a tree structure as a linked list extended with
    some convenience functions
    '''

    def __init__(self, root=None):
        """
        Initialize an empty tree by default
        """
        if root is not None:
            self.root = root
        else:
            self._root = None

    def __getitem__(self, index):
        '''
        Returns the node with given index, if no such node is in the tree, None
        is returned.

        Parameters
        ----------
            index: int
                the index of the node to be found

        Returns:
            :class:`SNode` or None
        '''
        return self._findNode(self.root, index)

    def _findNode(self, node, index):
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
                    stack.extend(cnode.getChildNodes())
        return None # Not found!

    def __len__(self, node=None):
        '''
        Return the number of nodes in the tree. If an input node is specified,
        the number of nodes in the subtree of the input node is returned

        Parameters
        ----------
            node: :class:`SNode` (optional)
                The starting node. Defaults to root

        Returns
        -------
            int
        '''
        self._node_count = 0
        for node in self.__iter__(node=node):
            self._node_count += 1
        return self._node_count

    def __iter__(self, node=None):
        '''
        Iterate over the nodes in the subtree of the given node. Beware, if
        the given node is not in the tree, it will simply iterate over the
        subtree of the given node.

        Parameters
        ----------
            node: :class:`SNode` (optional)
                The starting node. Defaults to the root
        '''
        if node is None:
            node = self.root
        if node is not None:
            yield node
            if node is not None:
                for cnode in node.getChildNodes():
                    for inode in self.__iter__(cnode):
                        yield inode

    def __str__(self, node=None):
        '''
        Generate a string of the subtree of the given node. Beware, if
        the given node is not in the tree, it will simply iterate over the
        subtree of the given node.

        Parameters
        ----------
            node: :class:`SNode` (optional)
                The starting node. Defaults to the root
        '''
        if node is None:
            node = self.root
        tree_string = '>>> Tree'
        for iternode in self.__iter__(node):
            tree_string += '\n    ' + iternode.__str__(with_parent=True)
        return tree_string

    def getNodes(self, recompute_flag=1):
        '''
        Build a list of all the nodes in the tree

        Parameters
        ----------
            recompute_flag: bool
                whether or not to re-evaluate the node list

        Returns
        -------
            list of :class:`Snode`
        '''
        if not hasattr(self, '_nodes') or recompute_flag:
            self._nodes = []
            self._gatherNodes(self.root, self._nodes)
        return self._nodes

    def setNodes(self, illegal):
        raise AttributeError("`nodes` is a read-only attribute")

    nodes = property(getNodes, setNodes)

    def gatherNodes(self, node):
        '''
        Build a list of all the nodes in the subtree of the provided node

        Parameters
        ----------
            node: :class:`Snode`
                starting point node

        Returns
        -------
            list of :class:`Snode`
        '''
        nodes = []
        self._gatherNodes(node, nodes)
        return nodes

    def _gatherNodes(self, node, node_list=[]):
        '''
        Append node to list and recurse to its child nodes

        Parameters
        ----------
            node: :class:`SNode`
            node_list: list of :class:`SNode`
        '''
        node_list.append(node)
        for cnode in node.child_nodes:
            self._gatherNodes(cnode, node_list=node_list)

    def getLeafs(self, recompute_flag=1):
        '''
        Get all leaf nodes in the tree.

        Parameters
        ----------
            recompute_flag: bool
                Whether to force recomputing the leaf list. Defaults to 1.
        '''
        if not hasattr(self, '_leafs') or recompute_flag:
            self._leafs = [node for node in self if self.isLeaf(node)]
        return self._leafs

    def setLeafs(self, illegal):
        raise AttributeError("`leafs` is a read-only attribute")

    leafs = property(getLeafs, setLeafs)

    def setRoot(self, node):
        '''
        Set the root node of the tree

        Parameters
        ----------
            node: :class:`SNode`
                node to be set as root of the tree
        '''
        node.parent_node = None
        self._root = node

    def getRoot(self):
        return self._root

    root = property(getRoot, setRoot)

    def isRoot(self, node):
        '''
        Check if input node is root of the tree.

        Parameters
        ----------
            node: :class:`SNode`
        '''
        if node.getParentNode() is not None:
            return False
        else:
            return True

    def isLeaf(self, node):
        '''
        Check if input node is a leaf of the tree

        Parameters
        ----------
            node: :class:`SNode`
        '''
        if len(node.getChildNodes()) == 0:
            return True
        else:
            return False

    def createCorrespondingNode(self, node_index):
        '''
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        '''
        return SNode(node_index)

    def addNodeWithParentFromIndex(self, node_index, pnode, *args, **kwargs):
        '''
        Create a node with the given index and add it to the tree under a
        specific parent node.

        Parameters
        ----------
            node_index: int
                index of the new node
            pnode: :class:`SNode`
                parent node of the newly added node

        Raises
        ------
            ValueError
                if ``node_index`` is already in the tree
        '''
        if self[node_index] == None:
            node = self.createCorrespondingNode(node_index, *args, **kwargs)
            self.addNodeWithParent(node, pnode)
        else:
            raise ValueError('Index %d is already exists in the tree.')

    def addNodeWithParent(self, node, pnode):
        '''
        Add a node to the tree under a specific parent node

        Parameters
        ----------
            node: :class:`SNode`
                node to be added
            pnode: :class:`SNode`
                parent node of the newly added node
        '''
        if pnode is not None:
            node.setParentNode(pnode)
            pnode.addChild(node)
        else:
            warnings.warn('`pnode` was `None`, did nothing.')

    def softRemoveNode(self, node):
        '''
        Remove a node and its subtree from the tree by deleting the reference
        to it in its parent. Internally, the node and its linked subtree are not
        changed

        Parameters
        ----------
            node: :class:`SNode`
                node to be removed
        '''
        node.getParentNode().removeChild(node)

    def removeNode(self, node):
        '''
        Remove a node as well as its subtree from the tree

        Parameters
        ----------
            node: :class:`SNode`
                node to be removed
        '''
        node.getParentNode().removeChild(node)
        self._deepRemove(node)

    def _deepRemove(self, node):
        cnodes = node.getChildNodes()
        node.makeEmpty()
        for cnode in cnodes:
            self._deepRemove(cnode)

    def removeSingleNode(self, node):
        '''
        Remove a single node from the tree. The nodes' children become the
        children of the nodes' parent.

        Parameters
        ----------
            node: :class:`SNode`
                node to be removed
        '''
        if node == self.root:
            raise ValueError('Removing root is forbidden')
        cnodes = node.getChildNodes()
        pnode = node.getParentNode()
        pnode.removeChild(node)
        for cnode in cnodes:
            cnode.setParentNode(pnode)
            pnode.addChild(cnode)

    def insertNode(self, node, pnode, pcnodes=[]):
        '''
        Insert a node in the tree as a child of a specified parent. The
        original children of the parent that will become children of the node
        are specified in the ``pcnodes`` list

        Parameters
        ----------
            node: :class:`SNode`
                the node that is to be inserted
            pnode: :class:`SNode`
                the node that will become parent of the node that is to be
                inserted
            pcnodes: list of :class:`SNode`
                the current children of the pnode that will become children of
                the node
        '''
        if pnode != None:
            cnodes = pnode.getChildNodes()
            for pcnode in pcnodes:
                if pcnode in cnodes:
                    ind = cnodes.index(pcnode)
                    del cnodes[ind]
                    node.addChild(pcnode)
                    pcnode.setParentNode(node)
                else:
                    warnings.warn(str(pcnode) + ' is not a child of ' \
                                              + str(pnode) + ', ignoring it')
            node.setParentNode(pnode)
            pnode.addChild(node)
        if pnode == None:
            cnode = self.root
            cnode.setParentNode(node)
            node.setParentNode(None)
            node.addChild(cnode)
            self.root = node

    def resetIndices(self, n=0):
        '''
        Resets the indices in the order they appear in a depth-first iteration
        '''
        for ind, node in enumerate(self):
            node.index = ind+n

    def getSubTree(self, node):
        '''
        Get the subtree of the specified node. The root of the subtree is a new
        node with the same children as the original node, but None instead of a
        parent. The other nodes are retained.

        Parameters
        ----------
            node: :class:`SNode`
                root of the sub tree
        '''
        subtree = STree()
        cp = copy.copy(node)
        cp.setParentNode(None)
        subtree.setRoot(cp)
        return subtree

    def degreeOfNode(self, node):
        '''
        Compute the degree (number of leafs in its subtree) of a node.

        Parameters
        ----------
            node: :class:`SNode`
        '''
        return len([node for node in self.__iter__(node) if self.isLeaf(node)])

    def orderOfNode(self, node):
        '''
        Compute the order (number of bifurcations from the root) of a node.

        Parameters
        ----------
            node: :class:`SNode`
        '''
        ptr = self.pathToRoot(node)
        order = 0
        for node in ptr:
            if len(node.child_nodes) > 1:
                order += 1
        # order is on [0,max_order] thus subtract 1 from this calculation
        return order - 1

    def pathToRoot(self, node):
        '''
        Return the path from a given node to the root

        Parameters:
            node: :class:`SNode`

        Returns
        -------
            list of :class:`SNode`
                List of nodes from ``node`` to root. First node is the input node
                and last node is the root
        '''
        nodes = []
        self._goUpFrom(node, nodes)
        return nodes

    def _goUpFrom(self, node, nodes):
        nodes.append(node)
        pnode = node.getParentNode()
        if pnode != None :
            self._goUpFrom(pnode, nodes)

    def pathBetweenNodes(self, from_node, to_node):
        '''
        Inclusive path from ``from_node`` to ``to_node``.

        Parameters
        ----------
            from_node: :class:`SNode`
            to_node: :class:`SNode`

        Returns
        -------
            list of :class:`SNode`
                List of nodes representing the direct path between ``from_node``
                and ``to_node``, which are respectively the first and last nodes
                in the list.
        '''
        path1 = self.pathToRoot(from_node)[::-1]
        path2 = self.pathToRoot(to_node)[::-1]
        path = path1 if len(path1) < len(path2) else path2
        ind = next((ii for ii in xrange(len(path)) if path1[ii] != path2[ii]),
                   len(path))
        return path1[ind:][::-1] + path2[ind-1:]

    def pathBetweenNodesDepthFirst(self, from_node, to_node):
        '''
        Inclusive path from ``from_node`` to ``to_node``, ginven in a depth-
        first ordering.

        Parameters
        ----------
            from_node: :class:`SNode`
            to_node: :class:`SNode`

        Returns
        -------
            list of :class:`SNode`
                List of nodes representing the direct path between ``from_node``
                and ``to_node``, which are respectively the first and last nodes
                in the list.
        '''
        path1 = self.pathToRoot(from_node)[::-1]
        path2 = self.pathToRoot(to_node)[::-1]
        path = path1 if len(path1) < len(path2) else path2
        ind = next((ii for ii in xrange(len(path)) if path1[ii] != path2[ii]),
                   len(path))
        return path1[ind-1:] + path2[ind:]

    def getNodesInSubtree(self, ref_node, subtree_root=None):
        '''
        Returns the nodes in the subtree that contains the given reference nodes
        and has the given subtree root as root. If the subtree root is not
        provided, the subtree of the first child node of the root on the path to
        the reference node is given (plus the root)

        Parameters
        ----------
            ref_node: :class:`SNode`
                the reference node that is in the subtree
            subtree_root: :class:`SNode`
                what is to be the root of the subtree. If this node is not on
                the path from reference node to root, a ValueError is raised

        Returns
        -------
            list of :class:`SNode`
                List of all nodes in the subtree. It's root is in the first
                position
        '''
        if subtree_root == None:
            subtree_root = self.root
        ref_path = self.pathBetweenNodes(ref_node, subtree_root)
        if subtree_root in ref_path:
            if len(ref_path) > 1:
                subtree_nodes = [subtree_root] \
                                + self.gatherNodes(ref_path[-2])
            else:
                subtree_nodes = [ref_node] # both input nodes are the same
        else:
            raise ValueError('|subtree_root| not in path from |ref_node| \
                                root')
        return subtree_nodes

    def sisterLeafs(self, node):
        '''
        Find the leafs that are in the subtree of the nearest bifurcation node
        up from the input node.

        Parameters
        ----------
            node: :class:`SNode`
                Starting node for search

        Returns
        -------
        (node, sisterLeafs, corresponding_children)
            node: :class:`SNode`
                the bifurcation node
            sisterLeafs: list of :class:`SNode`
                The first element is the input node. The others are the leafs
                of the subtree emanating from the bifurcation node that are not
                in the subtree from the input node.
            corresponding_children: list of :class:`SNode`
                The children of the bifurcation node. If the number of leafs
                ``sisterLeafs`` is the same as the number of
                ``corresponding_children``, the subtree of each element of
                ``corresponding_children`` has exactly one leaf, the corresponding
                element in ``sisterLeafs``
        '''
        sleafs = [node]; cchildren = []
        snode = self._goUpUntil(node, None, sl=sleafs, cc=cchildren)
        return snode, sleafs, cchildren

    def _goUpUntil(self, node, c_node, sl=[], cc=[]):
        c_nodes = node.getChildNodes()
        if (c_node != None and len(c_nodes) > 1) or self.isRoot(node):
            cc.append(c_node)
            for c_node_ in set(c_nodes) - {c_node}:
                self._goDownUntil(c_node_, sl=sl)
                cc.append(c_node_)
        else:
            p_node = node.getParentNode()
            node = self._goUpUntil(p_node, node, sl=sl, cc=cc)
        return node

    def _goDownUntil(self, node, sl=[]):
        c_nodes = node.getChildNodes()
        if len(c_nodes) == 0:
            sl.append(node)
        else:
            for c_node in c_nodes:
                self._goDownUntil(c_node, sl=sl)

    def upBifurcationNode(self, node, cnode=None):
        '''
        Find the nearest bifurcation node up from the input node.

        Parameters
        ----------
            node: :class:`SNode`
                Starting node for search
            cnode: :class:`SNode`
                For recursion, don't touch default

        Returns
        -------
        (node, cnode)
            node: :class:`SNode`
                the bifurcation node
            cnode: :class:`SNode`
                The bifurcation node's child on the path to the input node.

        '''
        if cnode == None or len(node.getChildNodes()) <= 1:
            pnode = node.getParentNode()
            if pnode != None:
                node, cnode = self.upBifurcationNode(pnode, cnode=node)
        return node, cnode