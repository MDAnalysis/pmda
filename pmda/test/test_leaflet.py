from __future__ import absolute_import, division, print_function

import pytest
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)

import MDAnalysis
from MDAnalysisTests.datafiles import Martini_membrane_gro
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT
from pmda import leaflet


class TestLeafLet(object):

    @pytest.fixture()
    def u_one_frame(self):
        return MDAnalysis.Universe(Martini_membrane_gro)

    @pytest.fixture()
    def universe(self):
        return MDAnalysis.Universe(Martini_membrane_gro)

    @pytest.fixture()
    def correct_values(self):
        pass

    @pytest.fixture()
    def correct_values_single_frame(self):
        return [range(1, 2150, 12), range(2521, 4670, 12)]

    def test_leaflet(self, universe):
        pass

    def test_leaflet_step(self, universe, correct_values):
        pass

    def test_leaflet_single_frame(self,
                                  u_one_frame,
                                  correct_values_single_frame):
        lipid_heads = u_one_frame.select_atoms("name PO4")
        u_one_frame.trajectory.rewind()
        leaflets = leaflet.LeafletFinder(u_one_frame, lipid_heads).run(start=0,
                                                                       stop=1)
        assert_almost_equal(leaflets.results[0].indices,
                            correct_values_single_frame, err_msg="error: " +
                            "leaflets should match test values")
