# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
RMSD analysis tools
"""
from __future__ import absolute_import

from MDAnalysis.analysis import rms
import numpy as np

from .parallel import ParallelAnalysisBase


class RMSD(ParallelAnalysisBase):
    def __init__(self, mobile, ref):
        universe = mobile.universe
        super(RMSD, self).__init__(universe, (mobile, ))
        self._ref_pos = ref.positions.copy()

    def _prepare(self):
        pass

    def _conclude(self):
        self.rmsd = np.hstack(self._results)

    def _single_frame(self, ts, atomgroups):
        return rms.rmsd(atomgroups[0].positions, self._ref_pos)
