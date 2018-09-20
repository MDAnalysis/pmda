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
=================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.rdf`.

See Also
--------
MDAnalysis.analysis.rdf


Classes
-------
.. autoclass:: InterRDF
   :members:
   :inherited-members:

"""

from __future__ import absolute_import, division

import numpy as np

from MDAnalysis.lib import distances

from .parallel import ParallelAnalysisBase


class InterRDF(ParallelAnalysisBase):
    """Intermolecular pair distribution function

    Attributes
    ----------
    bins : array
         The distance :math:`r` at which the distribution
         function :math:`g(r)` is determined; these are calculated as
         the centers of the bins that were used for histogramming.
    rdf : array
         The value of the pair distribution function :math:`g(r)` at
         :math:`r`.

    Parameters
    ----------
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

    See Also
    --------
    MDAnalysis.analysis.rdf.InterRDF


    .. versionadded:: 0.2.0

    """
    # pylint: disable=redefined-builtin
    # continue to use 'range' as long as MDAnalysis uses it so that
    # the user interface remains consistent
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

    # pylint: enable=redefined-builtin

    def _prepare(self):
        # Empty histogram to store the RDF
        self.count = self.bins * 0.0

        # Need to know average volume
        self.volume = 0.0
        # Set the max range to filter the search radius
        self._maxrange = self.rdf_settings['range'][1]

    def _single_frame(self, ts, atomgroups):
        g1, g2 = atomgroups
        u = g1.universe
        pairs, dist = distances.capped_distance(g1.positions,
                                                g2.positions,
                                                self._maxrange,
                                                box=u.dimensions)
        # If provided exclusions, create a mask of _result which
        # lets us take these out.
        if self._exclusion_block is not None:
            idxA = pairs[:, 0]//self._exclusion_block[0]
            idxB = pairs[:, 1]//self._exclusion_block[1]
            mask = np.where(idxA != idxB)[0]
            dist = dist[mask]
        count = np.histogram(dist, **self.rdf_settings)[0]
        volume = u.trajectory.ts.volume

        return np.array([count, np.array(volume, dtype=np.float64)])

    def _conclude(self, ):
        self.count = np.sum(self._results[:, 0])
        self.volume = np.sum(self._results[:, 1])
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

    @staticmethod
    def _reduce(res, result_single_frame):
        """ 'add' action for an accumulator"""
        if isinstance(res, list) and len(res) == 0:
            # Convert res from an empty list to a numpy array
            # which has the same shape as the single frame result
            res = result_single_frame
        else:
            # Add two numpy arrays
            res += result_single_frame
        return res
