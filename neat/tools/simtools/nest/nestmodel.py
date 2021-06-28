from ....trees.compartmenttree import CompartmentTree, CompartmentNode

try:
    import nest
except ModuleNotFoundError:
    warnings.warn('NEST not available', UserWarning)

def createNestModel(ctree, nestml_name):
    cm = nest.Create("cm_main_" + nestml_name)

    for node in ctree:
        g_dict = {'gbar_%s'%key: node.currents[key][0] for key in node.currents if key is not 'L'}
        e_dict = {'e_%s'%key:    node.currents[key][1] for key in node.currents if key is not 'L'}
        p_dict = {'g_L': node.currents['L'][0],
                  'e_L': node.currents['L'][1],
                  'C_m': node.ca,
                  'g_c': node.g_c}
        p_dict.update(g_dict)
        p_dict.update(e_dict)

        parent_idx = -1 if ctree.isRoot(node) else node.parent_node.index
        nest.AddCompartment(cm, node.index, parent_idx, p_dict)

    return cm