# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Contact analysis tools
"""
from __future__ import absolute_import, division

import MDAnalysis as mda
from MDAnalysis.analysis.contacts import (contact_matrix, hard_cut_q,
                                          radius_cut_q, soft_cut_q)
from MDAnalysis.lib.distances import distance_array
import numpy as np

from .parallel import ParallelAnalysisBase


class Contacts(ParallelAnalysisBase):
    """Calculate contacts based observables.

    The standard methods used in this class calculate the fraction of native
    contacts *Q* from a trajectory.


    By defining your own method it is possible to calculate other observables
    that only depend on the distances and a possible reference distance. The
    **Contact API** prescribes that this method must be a function with call
    signature ``func(r, r0, **kwargs)`` and must be provided in the keyword
    argument `method`.

    Attributes
    ----------
    timeseries : list
        list containing *Q* for all refgroup pairs and analyzed frames

    """

    def __init__(self,
                 mobiles,
                 refs,
                 method="hard_cut",
                 radius=4.5,
                 kwargs=None):
        """
        Parameters
        ----------
        mobiles : tuple(AtomGroup, AtomGroup)
            two contacting groups that change over time
        refs : tuple(AtomGroup, AtomGroup)
            two contacting atomgroups in their reference conformation. This
            can also be a list of tuples containing different atom groups
        radius : float, optional (4.5 Angstroms)
            radius within which contacts exist in refgroup
        method : string | callable (optional)
            Can either be one of ``['hard_cut' , 'soft_cut']`` or a callable
            with call signature ``func(r, r0, **kwargs)`` (the "Contacts API").
        kwargs : dict, optional
            dictionary of additional kwargs passed to `method`. Check
            respective functions for reasonable values.
        """
        universe = mobiles[0].universe
        super(Contacts, self).__init__(universe, mobiles)

        if method == 'hard_cut':
            self.fraction_contacts = hard_cut_q
        elif method == 'soft_cut':
            self.fraction_contacts = soft_cut_q
        else:
            if not callable(method):
                raise ValueError("method has to be callable")
            self.fraction_contacts = method

        # contacts formed in reference
        self.r0 = []
        self.initial_contacts = []

        if isinstance(refs[0], mda.core.groups.AtomGroup):
            refA, refB = refs
            self.r0.append(distance_array(refA.positions, refB.positions))
            self.initial_contacts.append(contact_matrix(self.r0[-1], radius))
        else:
            for refA, refB in refs:
                self.r0.append(distance_array(refA.positions, refB.positions))
                self.initial_contacts.append(
                    contact_matrix(self.r0[-1], radius))

        self.fraction_kwargs = kwargs if kwargs is not None else {}

    def _prepare(self):
        self.timeseries = None

    def _single_frame(self, ts, atomgroups):
        grA, grB = atomgroups
        # compute distance array for a frame
        d = distance_array(grA.positions, grB.positions)

        y = np.empty(len(self.r0) + 1)
        y[0] = ts.frame
        for i, (initial_contacts,
                r0) in enumerate(zip(self.initial_contacts, self.r0)):
            # select only the contacts that were formed in the reference state
            r = d[initial_contacts]
            r0 = r0[initial_contacts]
            y[i + 1] = self.fraction_contacts(r, r0, **self.fraction_kwargs)

        if len(y) == 1:
            y = y[0]
        return y

    def _conclude(self):
        self.timeseries = np.hstack(self._results)


def q1q2(atomgroup, radius=4.5):
    """Perform a q1-q2 analysis.

    Compares native contacts between the starting structure and final structure
    of a trajectory [Franklin2007]_.

    Parameters
    ----------
    atomgroup : mda.core.groups.AtomGroup
        atomgroup to analyze
    radius : float, optional
        distance at which contact is formed

    Returns
    -------
    contacts : :class:`Contacts`
        Contact Analysis that is set up for a q1-q2 analysis

    """
    # I need independent atomgroup copy at the start and beginning of the
    # simulation for a q1-q2 analysis
    u_orig = atomgroup.universe
    indices = atomgroup.indices

    def create_refs(frame):
        """create stand alone AGs from selections at frame"""
        u = mda.Universe(u_orig.filename, u_orig.trajectory.filename)
        u.trajectory[frame]
        return [u.atoms[indices], u.atoms[indices]]

    first_frame_refs = create_refs(0)
    last_frame_refs = create_refs(-1)
    return Contacts(
        (atomgroup, atomgroup), (first_frame_refs, last_frame_refs),
        radius=radius,
        method=radius_cut_q,
        kwargs={'radius': radius})
