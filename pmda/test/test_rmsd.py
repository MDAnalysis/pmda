# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

from __future__ import absolute_import, division, print_function
import pytest
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)
import MDAnalysis
from MDAnalysisTests.datafiles import (PSF, DCD)
from pmda.rms import RMSD

class TestRMSD(object):
    @pytest.fixture()
    def universe(self):
        return MDAnalysis.Universe(PSF, DCD)

    @pytest.fixture()
    def correct_values(self):
        return [[0, 1.0, 0], [49, 50.0, 4.68953]]

    @pytest.fixture()
    def correct_values_frame_5(self):
        return [[5, 6.0, 0.91544906]]

    def test_rmsd(self, universe):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        RMSD1 = RMSD(ca, ca).run(n_jobs=2)
        assert_array_equal(RMSD1.rmsd.shape, (universe.trajectory.n_frames, 3))

    def test_rmsd_step(self, universe, correct_values):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        RMSD1 = RMSD(ca, ca).run(step=49)
        assert_almost_equal(RMSD1.rmsd, correct_values, 4,
                            err_msg="error: rmsd profile should match " +
                            "test values")

    def test_rmsd_single_frame(self, universe, correct_values_frame_5):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        RMSD1 = RMSD(ca, ca).run(start=5, stop=6)
        assert_almost_equal(RMSD1.rmsd, correct_values_frame_5, 4,
                            err_msg="error: rmsd profile should match " +
                            "test values")
