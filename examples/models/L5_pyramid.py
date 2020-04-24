from ..channels import channels_hay

from neat import PhysTree


def getL5Pyramid():
    """
    Return a minimal model of the L5 pyramid for BAC-firing
    """
    # load the morphology
    phys_tree = PhysTree('../morphologies/cell1_simplified.swc')

    # set specific membrane capacitance and axial resistance
    phys_tree.setPhysiology(lambda x: 1. if x < .1 else 2. # Cm [uF/cm^2]
                            100./1e6 # Ra[MOhm*cm]
                            )

    # channels present in tree
    Kv3_1  = channels_hay.Kv3_1()
    Na_Ta  = channels_hay.Na_Ta()
    Ca_LVA = channels_hay.Ca_LVA()
    Ca_HVA = channels_hay.Ca_HVA()
    h_HAY  = channels_hay.h_HAY()

    # soma ion channels [uS/cm^2]
    phys_tree.addCurrent(Kv3_1,  0.766    *1e6, -85., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(Na_Ta,  0.0211   *1e6,  50., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(Ca_LVA, 0.00432  *1e6,  50., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(Ca_HVA, 0.000567 *1e6,  50., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(h_HAY,  0.0002   *1e6, -45., node_arg=[phys_tree[1]])
    phys_tree.setLeakCurrent(0.0000344 *1e6, -90., node_arg=[phys_tree[1]])

    # basal ion channels [uS/cm^2]
    phys_tree.addCurrent(h_HAY, 0.0002 *1e6, -45., node_arg='basal')
    phys_tree.setLeakCurrent(0.0000535 *1e6, -90., node_arg='basal')

    # apical ion channels [uS/cm^2]
    phys_tree.addCurrent(Kv3_1, 0.000298 *1e6, -85., node_arg='apical')
    phys_tree.addCurrent(Na_Ta, 0.0211   *1e6,  50., node_arg='apical')
    phys_tree.addCurrent(Ca_LVA, lambda x: 0.0198*1e6   if (x>685. and x<885.) else 0.0198*1e-2*1e6,   50., node_arg='apical')
    phys_tree.addCurrent(Ca_HVA, lambda x: 0.000437*1e6 if (x>685. and x<885.) else 0.000437*1e-1*1e6, 50., node_arg='apical')
    phys_tree.addCurrent(h_HAY,  lambda x: 0.0002*1e6 * (-0.8696 + 2.0870 * np.exp(x/323.)),          -45., node_arg='apical')
    phys_tree.setLeakCurrent(0.0000447*1e6, -90., node_arg='apical')

    return phys_tree