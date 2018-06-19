from __future__ import absolute_import, division

import MDAnalysis as mda

import numpy as np
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.util import blocks_of

from .parallel import ParallelAnalysisBase

class InterRDF(ParallelAnalysisBase):
    def __init__(self, g1, g2,
                 nbins=75, range=(0.0, 15.0), exclusion_block=None,
                 **kwargs):
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

        count, edges = np.histogram([-1], **self.rdf_settings)
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
        #self.count += count

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
