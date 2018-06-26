# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Radial Distribution Functions --- :mod:`pmda.rdf`
================================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.rdf`.

Classes:
-------

.. autoclass:: InterRDF

See Also
--------
MDAnalysis.analysis.rdf

"""

from __future__ import absolute_import, division

import numpy as np

from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.util import blocks_of

from .parallel import ParallelAnalysisBase

class InterRDF(ParallelAnalysisBase):
    """Intermolecular pair distribution function

    InterRDF(g1, g2, nbins=75, range=(0.0, 15.0))

    Arguments
    ---------
    g1 : AtomGroup
      First AtomGroup
    g2 : AtomGroup
      Second AtomGroup
    nbins : int (optional)
          Number of bins in the histogram [75]
    range : tuple or list (optional)
          The size of the RDF [0.0, 15.0]
    exclusion_block : tuple (optional)
          A tuple representing the tile to exclude from the distance
          array. [None]
    start : int (optional)
          The frame to start at (default is first)
    stop : int (optional)
          The frame to end at (default is last)
    step : int (optional)
          The step size through the trajectory in frames (default is
          every frame)

    Example
    -------
    First create the :class:`InterRDF` object, by supplying two
    AtomGroups then use the :meth:`run` method ::

      rdf = InterRDF(ag1, ag2)
      rdf.run()

    Results are available through the :attr:`bins` and :attr:`rdf`
    attributes::

      plt.plot(rdf.bins, rdf.rdf)

    The `exclusion_block` keyword allows the masking of pairs from
    within the same molecule.  For example, if there are 7 of each
    atom in each molecule, the exclusion mask `(7, 7)` can be used.

    .. versionadded:: 0.13.0

    """
    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None):
        u = g1.universe
        super(InterRDF, self).__init__(u, (g1, g2))

        # collect all atomgroups with the same trajectory object as universe
        trajectory = u.trajectory
        self.nA = len(g1)
        self.nB = len(g2)
        self.nf = trajectory.n_frames
        self.rdf_settings = {'bins': nbins,
                             'range': range}
        self._exclusion_block = exclusion_block

        edges = np.histogram([-1], **self.rdf_settings)[1]
        self.edges = edges
        self.bins = 0.5 * (edges[:-1] + edges[1:])

        # Allocate a results array which we will reuse
        self.result = np.zeros((len(g1), len(g2)), dtype=np.float64)

    def _prepare(self):
        # Empty histogram to store the RDF
        self.count = self.bins * 0.0

        # Need to know average volume
        self.volume = 0.0

    def _single_frame(self, ts, atomgroups):
        g1, g2 = atomgroups
        u = g1.universe
        d = distance_array(g1.positions, g2.positions,
                           box=u.dimensions)
        # If provided exclusions, create a mask of _result which
        # lets us take these out
        if self._exclusion_block is not None:
            self._exclusion_mask = blocks_of(d,
                                             *self._exclusion_block)
            self._maxrange = self.rdf_settings['range'][1] + 1.0
        else:
            self._exclusion_mask = None
        # Maybe exclude same molecule distances
        if self._exclusion_mask is not None:
            self._exclusion_mask[:] = self._maxrange
        count = []
        count = np.histogram(d, **self.rdf_settings)[0]
        volume = u.trajectory.ts.volume

        return {'count': count, 'volume': volume}

    def _conclude(self, ):
        for block in self._results:
            for data in block:
                self.count += data['count']
                self.volume += data['volume']

        # Number of each selection
        N = self.nA * self.nB

        # If we had exclusions, take these into account
        if self._exclusion_block:
            xA, xB = self._exclusion_block
            nblocks = self.nA / xA
            N -= xA * xB * nblocks

        # Volume in each radial shell
        vol = np.power(self.edges[1:], 3) - np.power(self.edges[:-1], 3)
        vol *= 4/3.0 * np.pi

        # Average number density
        box_vol = self.volume / self.nf
        density = N / box_vol

        rdf = self.count / (density * vol * self.nf)
        self.rdf = rdf
