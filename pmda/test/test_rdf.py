# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
from __future__ import absolute_import

import pytest

import MDAnalysis as mda
from pmda.rdf import InterRDF
from MDAnalysis.analysis import rdf

from numpy.testing import assert_almost_equal

from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT


@pytest.fixture(scope='module')
def u():
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)


@pytest.fixture(scope='module')
def sels(u):
    s1 = u.select_atoms('name OD1 and resname ASP')
    s2 = u.select_atoms('name OD2 and resname ASP')
    return s1, s2


def test_nbins(u):
    s1 = u.atoms[:3]
    s2 = u.atoms[3:]
    rdf = InterRDF(s1, s2, nbins=412).run()

    assert len(rdf.bins) == 412


def test_range(u):
    s1 = u.atoms[:3]
    s2 = u.atoms[3:]
    rmin, rmax = 1.0, 13.0
    rdf = InterRDF(s1, s2, range=(rmin, rmax)).run()

    assert rdf.edges[0] == rmin
    assert rdf.edges[-1] == rmax


def test_count_sum(sels):
    # OD1 vs OD2
    # should see 577 comparisons in count
    s1, s2 = sels
    rdf = InterRDF(s1, s2).run()
    assert rdf.count.sum() == 577


def test_count(sels):
    # should see two distances with 7 counts each
    s1, s2 = sels
    rdf = InterRDF(s1, s2).run()
    assert len(rdf.count[rdf.count == 3]) == 7


def test_double_run(sels):
    # running rdf twice should give the same result
    s1, s2 = sels
    rdf = InterRDF(s1, s2).run()
    rdf.run()
    assert len(rdf.count[rdf.count == 3]) == 7


def test_same_result(sels):
    # should see same results from analysis.rdf and pmda.rdf
    s1, s2 = sels
    nrdf = rdf.InterRDF(s1, s2).run()
    prdf = InterRDF(s1, s2).run()
    assert_almost_equal(nrdf.rdf, prdf.rdf)
    assert_almost_equal(nrdf.count, prdf.count)


@pytest.mark.parametrize('exclusion_block, value', [
            (None, 577),
            ((1, 1), 397)])
def test_exclusion(sels, exclusion_block, value):
    # should see 397 comparisons in count when given exclusion_block
    # should see 577 comparisons in count when exclusion_block is none
    s1, s2 = sels
    rdf = InterRDF(s1, s2, exclusion_block=exclusion_block).run()
    assert rdf.count.sum() == value
