import numpy as np

import nest
import nest.lib.hl_api_exceptions as nestexceptions

import pytest

from neat import PhysTree
from neat import CompartmentNode, CompartmentTree
from neat import NestCompartmentNode, NestCompartmentTree, loadNestModel


class TestNest:
    def loadTwoCompartmentModel(self):
        # simple two compartment model
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(root=pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.addNodeWithParent(cnode, pnode)

        for ii, cn in enumerate(self.ctree):
            cn.loc_ind = ii

    def testModelConstruction(self):
        loadNestModel("default")
        with pytest.raises(nestexceptions.NESTErrors.DynamicModuleManagementError):
            loadNestModel("default")

        self.loadTwoCompartmentModel()

        nct = self.ctree.__copy__(new_tree=NestCompartmentTree())
        cm_model = nct.initModel("default", 1)

        compartments_info = cm_model.compartments
        assert compartments_info[0]["comp_idx"] == 0
        assert compartments_info[0]["parent_idx"] == -1
        assert compartments_info[1]["comp_idx"] == 1
        assert compartments_info[1]["parent_idx"] == 0


if __name__ == "__main__":
    tn = TestNest()
    tn.testModelConstruction()