# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Parallel Analysis helper --- :mod:`pmda.custom`
===============================================

This module contains the class `AnalysisFromFunction` and the decorator
`analysis_class`. Both can be used to generate custom analysis classes that can
be run in parallel from functions that take one or more atom groups from the
same universe and return a value.

"""
from MDAnalysis.core.universe import Universe
from MDAnalysis.coordinates.base import ProtoReader
import numpy as np

from .parallel import ParallelAnalysisBase


class AnalysisFromFunction(ParallelAnalysisBase):
    """
    Create an analysis from a function working on AtomGroups

    The function that is used

    Attributes
    ----------
    results : ndarray
        results of calculation are stored after call to ``run``

    Example
    -------
    >>> # Create a new function to analyze a single frame
    >>> def rotation_matrix(mobile, ref):
    >>>     return mda.analysis.align.rotation_matrix(mobile, ref)[0]
    >>> # now run an analysis using the standalone function
    >>> rot = AnalysisFromFunction(rotation_matrix,
                                   universe, mobile, ref).run()
    >>> print(rot.results)

    Raises
    ------
    ValueError : if ``function`` has the same kwargs as ``BaseAnalysis``

    See Also
    --------
    analysis_class
    """

    def __init__(self, function, universe, *args, **kwargs):
        """Parameters
        ----------
        function : callable
            function to evaluate at each frame. The first arguments are assumed
            to be 'mobile' Atomgroups if they belong to the same universe. All
            other Atomgroups are assumed to be reference. Here 'mobile' means
            they will be iterated over.
        universe : :class:`~MDAnalysis.core.groups.Universe`
            a :class:`MDAnalysis.core.groups.Universe` (the
            `atomgroups` in other args must belong to this Universe)
        *args : list
           arguments for ``function``
        **kwargs : dict
           keyword arguments for ``function``.
        """
        self.function = function
        super().__init__(universe)
        self.args = args
        self.kwargs = kwargs

    def _prepare(self):
        self.results = []

    def _single_frame(self):
        return self.function(*self.args, **self.kwargs)

    def _conclude(self):
        self.results = np.concatenate(self._results)


def analysis_class(function):
    """Transform a function operating on a single frame to an analysis class

    Parameters
    ----------
    function : callable
        The function that can analyze a single or more atomgroups. It is always
        assumed that the mobile atomgroups (which will be iterated over) come
        first. All atomgroups that come directly after the first that are part
        of the same universe will iterated over

    Returns
    -------
    A new AnalysisClass with function as analysis

    Example
    -------
    For an usage in a library we recommend the following style:

    >>> def rotation_matrix(mobile, ref):
    >>>     return mda.analysis.align.rotation_matrix(mobile, ref)[0]
    >>> RotationMatrix = analysis_class(rotation_matrix)

    It can also be used as a decorator:

    >>> @analysis_class
    >>> def RotationMatrix(mobile, ref):
    >>>     return mda.analysis.align.rotation_matrix(mobile, ref)[0]

    >>> rot = RotationMatrix(u, mobile, ref, step=2).run()
    >>> print(rot.results)

    See Also
    --------
    AnalysisFromFunction

    """

    class WrapperClass(AnalysisFromFunction):
        """Custom Analysis Function"""

        def __init__(self, universe=None, *args, **kwargs):
            if not isinstance(universe, Universe):
                print(type(universe))
                raise ValueError(
                    "First argument needs to be an MDAnalysis Universe.")
            super().__init__(function, universe, *args,
                                               **kwargs)

    return WrapperClass
