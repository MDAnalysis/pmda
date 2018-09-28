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
from MDAnalysisTests.util import no_deprecated_call
import pytest
from numpy.testing import assert_equal

from pmda import custom


def custom_function(mobile):
    return mobile.center_of_geometry()


def test_AnalysisFromFunction(scheduler):
    u = mda.Universe(PSF, DCD)
    step = 2
    ana1 = custom.AnalysisFromFunction(custom_function, u, u.atoms).run(
        step=step, scheduler=scheduler
    )
    ana2 = custom.AnalysisFromFunction(custom_function, u, u.atoms).run(
        step=step, scheduler=scheduler
    )
    ana3 = custom.AnalysisFromFunction(custom_function, u, u.atoms).run(
        step=step, scheduler=scheduler
    )

    results = []
    for ts in u.trajectory[::step]:
        results.append(custom_function(u.atoms))
    results = np.asarray(results)

    for ana in (ana1, ana2, ana3):
        assert_equal(results, ana.results)


def custom_function_2(mobile, ref, ref2):
    return mobile.centroid() - ref.centroid() + 2 * ref2.centroid()


def test_AnalysisFromFunction_otherAgs():
    u = mda.Universe(PSF, DCD)
    u2 = mda.Universe(PSF, DCD)
    u3 = mda.Universe(PSF, DCD)
    step = 2
    ana = custom.AnalysisFromFunction(
        custom_function_2, u, u.atoms, u2.atoms, u3.atoms
    ).run(step=step)

    results = []
    for ts in u.trajectory[::step]:
        results.append(custom_function_2(u.atoms, u2.atoms, u3.atoms))
    results = np.asarray(results)
    assert_equal(results, ana.results)


def test_analysis_class():
    ana_class = custom.analysis_class(custom_function)
    assert issubclass(ana_class, custom.AnalysisFromFunction)

    u = mda.Universe(PSF, DCD)
    step = 2
    ana = ana_class(u, u.atoms).run(step=step)

    results = []
    for ts in u.trajectory[::step]:
        results.append(custom_function(u.atoms))
    results = np.asarray(results)

    assert_equal(results, ana.results)
    with pytest.raises(ValueError):
        ana_class(2)


def test_analysis_class_decorator():
    # Issue #1511
    # analysis_class should not raise
    # a DeprecationWarning
    u = mda.Universe(PSF, DCD)

    def distance(a, b):
        return np.linalg.norm((a.centroid() - b.centroid()))

    Distances = custom.analysis_class(distance)

    with no_deprecated_call():
        Distances(u, u.atoms[:10], u.atoms[10:20]).run()
