from __future__ import absolute_import, division, print_function

import pytest
from numpy.testing import (assert_almost_equal, assert_array_equal,
                                   assert_array_almost_equal)

import MDAnalysis
from MDAnalysisTests.datafiles import (PSF, DCD)

from pmda import leaflet


class TestLeafLet(object):

    @pytest.fixture()
    def universe(self):
        return MDAnalysis.Universe(PSF, DCD)

    @pytest.fixture()
    def correct_values(self):
        pass

    @pytest.fixture()
    def correct_values_frame_5(self):
        return [range(214)]

    def test_leaflet(self, universe):
        pass

    def test_leaflet_step(self, universe, correct_values):

        pass

    def test_leaflet_single_frame(self, universe, correct_values_frame_5):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        leaflets = leaflet.LeafletFinder(universe,ca).run(start=5,stop=6)
        print(leaflets.results)
        assert_almost_equal(leaflets.results, correct_values_frame_5, err_msg="error: leaflets should match " +
                                            "test values")