# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2019 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

"""
Calculating Root-Mean-Square Fluctuations (RMSF) --- :mod:`pmda.rmsf`
=====================================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.rms`.

.. autoclass:: RMSF
    :members:
    :inherited-members:


See Also
--------
MDAnalysis.analysis.rms.RMSF

"""

from __future__ import absolute_import, division

import numpy as np

from pmda.parallel import ParallelAnalysisBase

from pmda.util import fold_second_order_moments


class RMSF(ParallelAnalysisBase):
    def __init__(self, atomgroup, **kwargs):
        super().__init__(atomgroup.universe, **kwargs)
        self.atomgroup = atomgroup

    def _prepare(self):
        self.sumsquares = np.zeros((self.atomgroup.n_atoms, 3))
        self.mean = self.sumsquares.copy()
        self._results = [self.sumsquares, self.mean] * self.n_frames

    def _single_frame(self):
        k = self._block_i
        if k == 0:
            self.sumsquares = np.zeros((self.atomgroup.n_atoms, 3))
            self.mean = self.atomgroup.positions
        else:
            self.sumsquares += (k / (k+1.0)) * (self.atomgroup.positions - self.mean) ** 2
            self.mean = (k * self.mean + self.atomgroup.positions) / (k + 1)
        self._results[self._frame_index] = np.asarray([self.sumsquares, self.mean])

    def _conclude(self):
        n_blocks = len(self._results)
        # serial case
        if n_blocks == 1:
            # get length of trajectory slice
            self.mean = self._results[0][-1][1]
            self.sumsquares = self._results[0][-1][0]
            self.rmsf = np.sqrt(self.sumsquares.sum(axis=1) / self.n_frames)
        # parallel case
        else:
            vals = []
            for i in range(n_blocks):
                vals.append((len(self._blocks[i]), self._results[i][-1][1],
                                 self._results[i][-1][0]))
            # combine block results using fold method
            results = fold_second_order_moments(vals)
            self.mean = results[1]
            self.sumsquares = results[2]
            self.rmsf = np.sqrt(self.sumsquares.sum(axis=1) / self.n_frames)
            self._negative_rmsf(self.rmsf)

    @staticmethod
    def _negative_rmsf(rmsf):
        if not (rmsf >= 0).all():
            raise ValueError("Some RMSF values negative; overflow " +
                             "or underflow occurred")
