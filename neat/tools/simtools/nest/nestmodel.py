from ....trees.compartmenttree import CompartmentTree, CompartmentNode

import warnings

try:
    import nest
except ModuleNotFoundError:
    warnings.warn('NEST not available, importing non-functional nest module only for doc generation', UserWarning)
    # universal iterable mock object
    class N(object):
        def __init__(self):
            pass

        def __getattr__(self,attr):
            try:
                return super(N, self).__getattr__(attr)
            except AttributeError:
                return self.__global_handler

        def __global_handler(self, *args, **kwargs):
            return N()

        def __iter__(self):  # make iterable
            return self

        def __next__(self):
            raise StopIteration

        def __mul__(self, other):  # make multipliable
            return 1.0

        def __rmul__(self, other):
            return self * other

        def __call__(self, *args, **kwargs): # make callable
            return N()
    nest = N()
    np_array = np.array
    def array(*args, **kwargs):
        if isinstance(args[0], N):
            print(args)
            print(kwargs)
            return np.eye(2)
        else:
            return np_array(*args, **kwargs)
    np.array = array


def loadNestModel(name):
    nest.Install(name + "_module")


class NestCompartmentNode(CompartmentNode):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def _makeCompartmentDict(self):
        g_dict = {
            f'gbar_{key}{self.index}': self.currents[key][0] for key in self.currents if key != 'L'
        }
        e_dict = {
            f'e_{key}{self.index}': self.currents[key][1] for key in self.currents if key != 'L'
        }
        p_dict = {
            'g_L': self.currents['L'][0],
            'e_L': self.currents['L'][1],
            'C_m': self.ca*1e3, # convert uF to nF
            'g_c': self.g_c
        }
        p_dict.update(g_dict)
        p_dict.update(e_dict)

        if self.parent_node is None:
            parent_idx = -1
        else:
            parent_idx = self.parent_node.index

        return {"parent_idx": parent_idx, "params": p_dict}


class NestCompartmentTree(CompartmentTree):
    def __init__(self, root=None):
        super().__init__(root=root)

    def _createCorrespondingNode(self, index, **kwargs):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
        node_index: int
            index of the new node
        **kwargs
            Parameters of the parent class node initialization function
        """
        return NestCompartmentNode(index, **kwargs)

    def _getCompartmentsStatus(self):
        # ensure that the all node indices are equal to the position where
        # they appear in the iteration, so that they also correspond to the NEST
        # model indices
        self.resetIndices()

        return [node._makeCompartmentDict() for node in self]

    def initModel(self, model_name, n, **kwargs):
        """
        Initialize n nest instantiations of the current model.

        Parameters
        ----------
        model_name: str
            name of the model, corresponds to the name given when the nest model
            was installed with `neatmodels install`
        n: int (> 0)
            The number of copies of the model to be instantiated
        **kwargs
            Keyword arguments to the `nest.Create()` function
        """
        models = nest.Create(model_name + "_model", n, **kwargs)
        models.compartments = self._getCompartmentsStatus()

        return models
