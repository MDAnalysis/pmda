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
from __future__ import absolute_import

from MDAnalysis.core.groups import AtomGroup
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
                                   trajectory, mobile, ref).run()
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
        Universe : :class:`~MDAnalysis.core.groups.Universe`
            a :class:`MDAnalysis.core.groups.Universe` (the
            `atomgroups` must belong to this Universe)
        *args : list
           arguments for ``function``
        **kwargs : dict
           keyword arguments for ``function``. keyword arguments with name
           'universe' or 'atomgroups' will be ignored! Mobile atomgroups to
           analyze can not be passed as keyword arguments currently.

        """

        self.function = function

        # collect all atomgroups with the same trajectory object as universe
        trajectory = universe.trajectory
        arg_ags = []
        self.other_args = []
        for arg in args:
            if isinstance(arg,
                          AtomGroup) and arg.universe.trajectory == trajectory:
                arg_ags.append(arg)
            else:
                self.other_args.append(arg)

        super(AnalysisFromFunction, self).__init__(universe, arg_ags)
        self.kwargs = kwargs

    def _prepare(self):
        self.results = []

    def _single_frame(self, ts, atomgroups):
        args = atomgroups + self.other_args
        return self.function(*args, **self.kwargs)

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

    >>> rot = RotationMatrix(u.trajectory, mobile, ref, step=2).run()
    >>> print(rot.results)

    See Also
    --------
    AnalysisFromFunction

    """

    class WrapperClass(AnalysisFromFunction):
        """Custom Analysis Function"""

        def __init__(self, trajectory=None, *args, **kwargs):
            if not (isinstance(trajectory, ProtoReader) or isinstance(
                    trajectory, Universe)):
                print(type(trajectory))
                raise ValueError(
                    "First argument needs to be an MDAnalysis reader object.")
            super(WrapperClass, self).__init__(function, trajectory, *args,
                                               **kwargs)

    return WrapperClass
