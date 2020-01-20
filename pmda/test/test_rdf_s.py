# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
from __future__ import absolute_import, division

import pytest

from numpy.testing import assert_almost_equal

import MDAnalysis as mda
import numpy as np
from pmda.rdf import InterRDF_s
from MDAnalysis.analysis import rdf

from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT


@pytest.fixture(scope='module')
def u():
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)


@pytest.fixture(scope='module')
def sels(u):
    s1 = u.select_atoms('name ZND and resid 289')
    s2 = u.select_atoms(
         '(name OD1 or name OD2) and resid 51 and sphzone 5.0 (resid 289)')
    s3 = u.select_atoms('name ZND and (resid 291 or resid 292)')
    s4 = u.select_atoms('(name OD1 or name OD2) and sphzone 5.0 (resid 291)')
    ags = [[s1, s2], [s3, s4]]
    return ags


@pytest.fixture(scope='module')
def rdf_s(u, sels, scheduler):
    return InterRDF_s(u, sels).run()


def test_nbins(u, sels):
    rdf = InterRDF_s(u, sels, nbins=412).run()

    assert len(rdf.bins) == 412


def test_range(u, sels):
    rmin, rmax = 1.0, 13.0
    rdf = InterRDF_s(u, sels, range=(rmin, rmax)).run()

    assert rdf.edges[0] == rmin
    assert rdf.edges[-1] == rmax


def test_count_size(rdf_s):
    # ZND vs OD1 & OD2
    # should see 2 elements in rdf.count
    # 1 element in rdf.count[0]
    # 2 elements in rdf.count[0][0]
    # 2 elements in rdf.count[1]
    # 2 elements in rdf.count[1][0]
    # 2 elements in rdf.count[1][1]
    assert len(rdf_s.count) == 2
    assert len(rdf_s.count[0]) == 1
    assert len(rdf_s.count[0][0]) == 2
    assert len(rdf_s.count[1]) == 2
    assert len(rdf_s.count[1][0]) == 2
    assert len(rdf_s.count[1][1]) == 2


def test_count(rdf_s):
    # should see one distance with 5 counts in count[0][0][1]
    # should see one distance with 3 counts in count[1][1][0]
    assert len(rdf_s.count[0][0][1][rdf_s.count[0][0][1] == 5]) == 1
    assert len(rdf_s.count[1][1][0][rdf_s.count[1][1][0] == 3]) == 1


def test_double_run(rdf_s):
    # running rdf twice should give the same result
    assert len(rdf_s.count[0][0][1][rdf_s.count[0][0][1] == 5]) == 1
    assert len(rdf_s.count[1][1][0][rdf_s.count[1][1][0] == 3]) == 1


def test_cdf(rdf_s):
    assert_almost_equal(rdf_s.cdf[0][0][0][-1],
                        rdf_s.count[0][0][0].sum()/rdf_s.n_frames)


def test_reduce(rdf_s):
    # should see numpy.array addtion
    res = []
    single_frame = np.array([np.array([1, 2]), np.array([3])])
    res = rdf_s._reduce(res, single_frame)
    res = rdf_s._reduce(res, single_frame)
    assert_almost_equal(res[0], np.array([2, 4]))
    assert_almost_equal(res[1], np.array([6]))


@pytest.mark.parametrize("n_blocks", [1, 2, 3, 4])
def test_same_result(u, sels, n_blocks):
    # should see same results from analysis.rdf.InterRDF_s
    # and pmda.rdf.InterRDF_s
    nrdf = rdf.InterRDF_s(u, sels).run()
    prdf = InterRDF_s(u, sels).run(n_blocks=n_blocks)
    assert_almost_equal(nrdf.count[0][0][0], prdf.count[0][0][0])
    assert_almost_equal(nrdf.rdf[0][0][0], prdf.rdf[0][0][0])


@pytest.mark.parametrize("density, value", [
    (True, 13275.775440503656),
    (False, 0.021915460340071267)])
def test_density(u, sels, density, value):
    rdf = InterRDF_s(u, sels, density=density).run()
    assert_almost_equal(max(rdf.rdf[0][0][0]), value)
