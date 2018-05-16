# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

"""
PMDA --- Parallel MDAnalysis
============================

Ready to use analysis and buildings blocks to write parallel analysis
algorithms using MDAnalysis with dask.

The full documentation can be found at https://www.mdanalysis.org/pmda/.

Note that by default no modules are imported in the top-level name
space. Import what you need::

   import pmda.parallel
   import pmda.custom
   import pmda.rms
   import pmda.contacts

"""
from __future__ import absolute_import

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
