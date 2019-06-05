# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
from __future__ import absolute_import, division


import numpy as np
import MDAnalysis as mda
from MDAnalysisTests.datafiles import PSF, DCD
import pytest
from numpy.testing import assert_equal

from pmda import custom

@pytest.fixture
def universe():
    return mda.Universe(PSF, DCD)

def custom_function_vector(mobile):
    return mobile.center_of_geometry()

def custom_function_scalar(ag):
    return ag.radius_of_gyration()

@pytest.mark.parametrize('custom_function', [
    custom_function_vector,
    custom_function_scalar,
    ])
@pytest.mark.parametrize('step', [None, 1, 2, 3, 7, 33])
def test_AnalysisFromFunction(scheduler, universe, custom_function, step):
    ana1 = custom.AnalysisFromFunction(custom_function, universe, universe.atoms).run(
        step=step
    )
    ana2 = custom.AnalysisFromFunction(custom_function, universe, universe.atoms).run(
        step=step
    )
    ana3 = custom.AnalysisFromFunction(custom_function, universe, universe.atoms).run(
        step=step
    )

    results = []
    for ts in universe.trajectory[::step]:
        results.append(custom_function(universe.atoms))
    results = np.asarray(results)

    for ana in (ana1, ana2, ana3):
        assert_equal(results, ana.results)


def custom_function_2(mobile, ref, ref2):
    return mobile.centroid() - ref.centroid() + 2 * ref2.centroid()

def test_AnalysisFromFunction_otherAgs(universe, step=2):
    u1 = universe
    u2 = universe.copy()
    u3 = universe.copy()
    ana = custom.AnalysisFromFunction(
        custom_function_2, u1, u1.atoms, u2.atoms, u3.atoms
    ).run(step=step)

    results = []
    for ts in u1.trajectory[::step]:
        results.append(custom_function_2(u1.atoms, u2.atoms, u3.atoms))
    results = np.asarray(results)
    assert_equal(results, ana.results)


def test_analysis_class(universe, step=2):
    ana_class = custom.analysis_class(custom_function)
    assert issubclass(ana_class, custom.AnalysisFromFunction)

    ana = ana_class(universe, universe.atoms).run(step=step)

    results = []
    for ts in universe.trajectory[::step]:
        results.append(custom_function(universe.atoms))
    results = np.asarray(results)

    assert_equal(results, ana.results)
    with pytest.raises(ValueError):
        ana_class(2)
