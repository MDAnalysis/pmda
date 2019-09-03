# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2019 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

"""
pmda.rms
========

Ready to use root-mean-square analyses built via MDAnalysis with dask.

The full documentation can be found at https://www.mdanalysis.org/pmda/.

Note that by default no modules are imported in the top-level name
space. Import what you need::

    import pmda.rms.rmsd.RMSD
    import pmda.rms.rmsf.RMSF
"""

from __future__ import absolute_import
from .rmsd import RMSD
from .rmsf import RMSF

__all__ = ["RMSD", "RMSF"]
