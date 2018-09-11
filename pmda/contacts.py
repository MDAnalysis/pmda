# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""Native contacts analysis --- :mod:`pmda.contacts`
================================================================

This module contains classes to analyze native contacts *Q* over a
trajectory. Native contacts of a conformation are contacts that exist
in a reference structure and in the conformation. Contacts in the
reference structure are always defined as being closer then a distance
`radius`. The fraction of native contacts for a conformation can be
calculated in different ways. This module supports 3 different metrics
listed below, as well as custom metrics.

1. *Hard Cut*: To count as a contact the atoms *i* and *j* have to be at least
   as close as in the reference structure.

2. *Soft Cut*: The atom pair *i* and *j* is assigned based on a soft potential
   that is 1 if the distance is 0, 1/2 if the distance is the same as in
   the reference and 0 for large distances. For the exact definition of the
   potential and parameters have a look at function :func:`soft_cut_q`.

3. *Radius Cut*: To count as a contact the atoms *i* and *j* cannot be further
   apart than some distance `radius`.

The "fraction of native contacts" *Q(t)* is a number between 0 and 1 and
calculated as the total number of native contacts for a given time frame
divided by the total number of contacts in the reference structure.


Examples for contact analysis
-----------------------------

One-dimensional contact analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example we analyze the opening ("unzipping") of salt bridges
when the AdK enzyme opens up; this is one of the example trajectories
in MDAnalysis. ::

    import MDAnalysis as mda
    from pmda import contacts
    from MDAnalysis.tests.datafiles import PSF,DCD
    import matplotlib.pyplot as plt
    # example trajectory (transition of AdK from closed to open)
    u = mda.Universe(PSF,DCD)
    # crude definition of salt bridges as contacts between NH/NZ in ARG/LYS and
    # OE*/OD* in ASP/GLU. You might want to think a little bit harder about the
    # problem before using this for real work.
    sel_basic = "(resname ARG LYS) and (name NH* NZ)"
    sel_acidic = "(resname ASP GLU) and (name OE* OD*)"
    # reference groups (first frame of the trajectory, but you could also use a
    # separate PDB, eg crystal structure)
    acidic = u.select_atoms(sel_acidic)
    basic = u.select_atoms(sel_basic)
    # set up analysis of native contacts ("salt bridges"); salt bridges have a
    # distance <6 A
    ca1 = contacts.Contacts(mobiles=(acidic, basic),
                            refs=(acidic, basic),
                            radius=6.0)
    # analyze trajectory and perform analysis of "native contacts" Q
    ca1.run(n_jobs=-1)
    # print number of averave contacts
    average_contacts = np.mean(ca1.timeseries[:, 1])
    print('average contacts = {}'.format(average_contacts))
    # plot time series q(t)
    f, ax = plt.subplots()
    ax.plot(ca1.timeseries[:, 0], ca1.timeseries[:, 1])
    ax.set(xlabel='frame', ylabel='fraction of native contacts',
           title='Native Contacts, average = {:.2f}'.format(average_contacts))
    fig.show()


The first graph shows that when AdK opens, about 20% of the salt
bridges that existed in the closed state disappear when the enzyme
opens. They open in a step-wise fashion (made more clear by the movie
`AdK_zipper_cartoon.avi`_).

.. _`AdK_zipper_cartoon.avi`:
   http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2803350/bin/NIHMS150766-supplement-03.avi

.. rubric:: Notes

Suggested cutoff distances for different simulations

* For all-atom simulations, cutoff = 4.5 Å
* For coarse-grained simulations, cutoff = 6.0 Å


Two-dimensional contact analysis (q1-q2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze a single DIMS transition of AdK between its closed and open
conformation and plot the trajectory projected on q1-q2 [Franklin2007]_ ::


    import MDAnalysis as mda
    from pmda import contacts
    from MDAnalysisTests.datafiles import PSF, DCD
    u = mda.Universe(PSF, DCD)
    q1q2 = contacts.q1q2(u.select_atoms('name CA'), radius=8)
    q1q2.run()

    f, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.5))
    ax[0].plot(q1q2.timeseries[:, 0], q1q2.timeseries[:, 1], label='q1')
    ax[0].plot(q1q2.timeseries[:, 0], q1q2.timeseries[:, 2], label='q2')
    ax[0].legend(loc='best')
    ax[1].plot(q1q2.timeseries[:, 1], q1q2.timeseries[:, 2], '.-')
    f.show()

Compare the resulting pathway to the `MinActionPath result for AdK`_
[Franklin2007]_.

.. _MinActionPath result for AdK:
   http://lorentz.dynstr.pasteur.fr/joel/adenylate.php


Writing your own contact analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`Contacts` class has been designed to be extensible for your own
analysis. As an example we will analyze when the acidic and basic groups of AdK
are in contact which each other; this means that at least one of the contacts
formed in the reference is closer than 2.5 Å.

For this we define a new function to determine if any contact is closer than
2.5 Å; this function must implement the API prescribed by :class:`Contacts`::

    def is_any_closer(r, r0, dist=2.5):
        return np.any(r < dist)

The first two parameters `r` and `r0` are provided by :class:`Contacts` when it
calls :func:`is_any_closer` while the others can be passed as keyword args
using the `kwargs` parameter in :class:`Contacts`.

Next we are creating an instance of the :class:`Contacts` class and use the
:func:`is_any_closer` function as an argument to `method` and run the
analysis::

    # crude definition of salt bridges as contacts between NH/NZ in ARG/LYS and
    # OE*/OD* in ASP/GLU. You might want to think a little bit harder about the
    # problem before using this for real work.
    sel_basic = "(resname ARG LYS) and (name NH* NZ)"
    sel_acidic = "(resname ASP GLU) and (name OE* OD*)"

    # reference groups (first frame of the trajectory, but you could also use a
    # separate PDB, eg crystal structure)
    acidic = u.select_atoms(sel_acidic)
    basic = u.select_atoms(sel_basic)

    nc = contacts.Contacts(mobiles=(acidic, basic), refs=(acidic, basic)
                           method=is_any_closer,
                           kwargs={'dist': 2.5}).run(n_jobs=-1)

    bound = nc.timeseries[:, 1]
    frames = nc.timeseries[:, 0]

    f, ax = plt.subplots()

    ax.plot(frames, bound, '.')
    ax.set(xlabel='frame', ylabel='is Bound',
           ylim=(-0.1, 1.1))

    f.show()


Functions
---------

.. autofunction:: q1q2

Classes
-------

.. autoclass:: Contacts
   :members:
   :inherited-members:

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

    References
    ----------
    .. [Franklin2007] Franklin, J., Koehl, P., Doniach, S., & Delarue,
       M. (2007).  MinActionPath: Maximum likelihood trajectory for large-scale
       structural transitions in a coarse-grained locally harmonic energy
       landscape.  Nucleic Acids Research, 35(SUPPL.2), 477–482.
       doi: `10.1093/nar/gkm342 <http://doi.org/10.1093/nar/gkm342>`_
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
