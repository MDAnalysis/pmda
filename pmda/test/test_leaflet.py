from __future__ import absolute_import, division, print_function

import pytest
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)

import MDAnalysis
from MDAnalysisTests.datafiles import Martini_membrane_gro
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT
from dask import multiprocessing
from pmda import leaflet
import numpy as np


class TestLeafLet(object):

    @pytest.fixture()
    def u_one_frame(self):
        return MDAnalysis.Universe(Martini_membrane_gro)

    @pytest.fixture()
    def universe(self):
        return MDAnalysis.Universe(GRO_MEMPROT, XTC_MEMPROT)

    @pytest.fixture()
    def correct_values(self):
        return [np.array([36507, 36761, 37523, 37650, 38031, 38285]),
                np.array([36634]),
                np.array([36507, 36761, 38285, 39174]),
                np.array([36634]),
                np.array([36507, 36761, 37650, 38285, 39174, 39936]),
                np.array([36634]),
                np.array([36507, 36761, 37650, 38285, 39174, 39428, 39936]),
                np.array([36634]),
                np.array([36507, 36761]),
                np.array([36634])]

    @pytest.fixture()
    def correct_values_single_frame(self):
        return [np.arange(1, 2150, 12), np.arange(2521, 4670, 12)]

    def test_leaflet(self, universe, correct_values):
        lipid_heads = universe.select_atoms("name P and resname POPG")
        universe.trajectory.rewind()
        leaflets = leaflet.LeafletFinder(universe, lipid_heads)
        leaflets.run(scheduler=multiprocessing, n_jobs=1)
        results = [atoms.indices for atomgroup in leaflets.results
                   for atoms in atomgroup]
        [assert_almost_equal(x, y, err_msg="error: leaflets should match " +
                             "test values") for x, y in
         zip(results, correct_values)]

    def test_leaflet_single_frame(self,
                                  u_one_frame,
                                  correct_values_single_frame):
        lipid_heads = u_one_frame.select_atoms("name PO4")
        u_one_frame.trajectory.rewind()
        leaflets = leaflet.LeafletFinder(u_one_frame,
                                         lipid_heads).run(start=0, stop=1)

        assert_almost_equal([atoms.indices for atomgroup in leaflets.results
                            for atoms in atomgroup],
                            correct_values_single_frame, err_msg="error: " +
                            "leaflets should match test values")
