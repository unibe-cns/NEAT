from ..channels import channels_branco

from neat import PhysTree


def getL23PyramidPas():
    """
    Return a passive model of the L2/3 pyramidal cell
    """
    phys_tree = PhysTree('../morphologies/L23PyrBranco.swc', types=[1,2,3,4])

    # set specific membrane capacitance and axial resistance
    phys_tree.setPhysiology(1. # Cm [uF/cm^2]
                            150./1e6 # Ra[MOhm*cm]
                            )

    # passive membrane conductance
    phys_tree.setLeakCurrent(1e6/150., -75.)
    phys_tree.setLeakCurrent(0.02*1e6, -75., node_arg='axonal')

    return phys_tree


def getL23PyramidNaK():
    """
    Return a model of the L2/3 pyramidal cell with somatic and basal Na- and
    K-channels
    """
    tadj = 3.21
    rm = 10000. # Ohm * cm^2
    el = -75.           # mV

    phys_tree = PhysTree('../morphologies/L23PyrBranco.swc')

    phys_tree.setPhysiology(1.       # Cm [uF/cm^2]
                            150./1e6 # Ra[MOhm*cm]
                            )

    # channels
    Na  = channels_branco.Na()
    K_v = channels_branco.K_v()

    # somatic channels
    phys_tree.addCurrent(Na,  tadj*1500.*1e2, 60., node_arg=[phys_tree[1]]) # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_v, tadj*200.*1e2, -90., node_arg=[phys_tree[1]]) # pS/um^2 -> uS/cm^2
    phys_tree.setLeakCurrent(1./rm*1e6, el, node_arg=[phys_tree[1]])

    # basal channels
    phys_tree.addCurrent(Na,  tadj*40.*1e2,  60., node_arg='basal') # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_v, tadj*30.*1e2, -90., node_arg='basal') # pS/um^2 -> uS/cm^2
    phys_tree.setLeakCurrent(1./rm*1e6, el, node_arg='basal')

    # passive apical dendrite
    phys_tree.addCurrent('L', 1./rm*1e6, e_rev=el, node_arg='apical')

    return phys_tree


def getL23Pyramid():
    """
    Return a model of the L2/3 pyramidal cell with somatic and basal Na-, K- and
    Ca-channels
    """
        cm = 1.             # uF / cm^2
        rm = 10000.         # Ohm * cm^2
        ri = 150.           # Ohm * cm
        el = -75.           # mV
        g_axon = 0.02       # S /cm^2
        tadj = 3.21

    phys_tree = PhysTree('../morphologies/L23PyrBranco.swc')

    phys_tree.setPhysiology(1.       # Cm [uF/cm^2]
                            150./1e6 # Ra[MOhm*cm]
                            )

    # somatic channels
    phys_tree.addCurrent(Na,   tadj*1500.*1e2, 60., node_arg=[phys_tree[1]]) # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_v,  tadj*200.*1e2, -90., node_arg=[phys_tree[1]]) # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_m,  tadj*2.2*1e2,  -90., node_arg=[phys_tree[1]]) # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_ca, tadj*2.5*1e2,  -90., node_arg=[phys_tree[1]]) # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(Ca_H, tadj*0.5*1e2,  140., node_arg=[phys_tree[1]]) # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(Ca_T, 0.0003*1e6,     60., node_arg=[phys_tree[1]]) # mho/cm^2 -> uS/cm^2
    phys_tree.setLeakCurrent(1./rm*1e6, el, node_arg=[phys_tree[1]])

    # basal channels
    phys_tree.addCurrent(Na,   tadj*40.*1e2,   60., node_arg='basal') # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_v,  tadj*30.*1e2,  -90., node_arg='basal')  # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_m,  tadj*0.05*1e2, -90., node_arg='basal')  # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(K_ca, tadj*2.5*1e2,  -90., node_arg='basal') # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(Ca_H, tadj*0.5*1e2,  140., node_arg='basal')  # pS/um^2 -> uS/cm^2
    phys_tree.addCurrent(Ca_T, 0.0006*1e6,     60., node_arg='basal') # mho/cm^2 -> mho/cm^2
    phys_tree.setLeakCurrent(1./rm*1e6, el, node_arg='basal')

    # passive apical dendrite
    phys_tree.setLeakCurrent(1./rm*1e6, el, node_arg='apical')

    return phys_tree
