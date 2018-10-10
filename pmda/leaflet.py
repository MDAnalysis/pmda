# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2018 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
LeafletFInder Analysis tool --- :mod:`pmda.leaflet`
==========================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.leaflet`.

.. autoclass:: LeafletFinder
   :members:
   :undoc-members:
   :inherited-members:

"""
from __future__ import absolute_import

import numpy as np
import dask.bag as db
import networkx as nx
from scipy.spatial import cKDTree

import MDAnalysis as mda
from dask import distributed, multiprocessing
from joblib import cpu_count

from .parallel import ParallelAnalysisBase, Timing
from .util import timeit


class LeafletFinder(ParallelAnalysisBase):
    """Parallel Leaflet Finder analysis.

    Identify atoms in the same leaflet of a lipid bilayer.
    This class implements and parallelizes the *LeafletFinder* algorithm
    [Michaud-Agrawal2011]_.

    The parallelization is done based on [Paraskevakos2018]_.

    Attributes
    ----------

    Parameters
    ----------
        Universe : :class:`~MDAnalysis.core.groups.Universe`
            a :class:`MDAnalysis.core.groups.Universe` (the
            `atomgroup` must belong to this Universe)
        atomgroup : tuple of :class:`~MDAnalysis.core.groups.AtomGroup`
            atomgroups that are iterated in parallel

    Note
    ----
    At the moment, this class has far fewer features than the serial
    version :class:`MDAnalysis.analysis.leaflet.LeafletFinder`.

    This version offers Leaflet Finder algorithm 4 ("Tree-based Nearest 
    Neighbor and Parallel-Connected Com- ponents (Tree-Search)") in 
    [Paraskevakos2018]_.

    """

    def __init__(self, universe, atomgroup):
        """Parameters
        ----------
        Universe : :class:`~MDAnalysis.core.groups.Universe`
            a :class:`MDAnalysis.core.groups.Universe` (the
            `atomgroup` must belong to this Universe)

        atomgroup : tuple of :class:`~MDAnalysis.core.groups.AtomGroup`
            atomgroup that are iterated in parallel

        Attributes
        ----------

        """
        self._trajectory = universe.trajectory
        self._top = universe.filename
        self._traj = universe.trajectory.filename
        self._atomgroup = atomgroup
        self._results = list()

    def _find_parcc(self, data, cutoff=15.0):
        window, index = data[0]
        num = window[0].shape[0]
        i_index = index[0]
        j_index = index[1]
        graph = nx.Graph()
        if i_index == j_index:
            train = window[0]
            test = window[1]
        else:
            train = np.vstack([window[0], window[1]])
            test = np.vstack([window[0], window[1]])
        tree = cKDTree(train, leafsize=40)
        edges = tree.query_ball_point(test, cutoff)
        edge_list = [list(zip(np.repeat(idx, len(dest_list)), dest_list))
                     for idx, dest_list in enumerate(edges)]

        edge_list_flat = np.array([list(item) for sublist in edge_list for
                                   item in sublist])
        if i_index == j_index:
            res = edge_list_flat.transpose()
            res[0] = res[0] + i_index - 1
            res[1] = res[1] + j_index - 1
        else:
            removed_elements = list()
            for i in range(edge_list_flat.shape[0]):
                if (edge_list_flat[i, 0] >= 0 and
                    edge_list_flat[i, 0] <= num - 1) and \
                    (edge_list_flat[i, 1] >= 0 and
                     edge_list_flat[i, 1] <= num - 1) or \
                    (edge_list_flat[i, 0] >= num and
                     edge_list_flat[i, 0] <= 2 * num - 1) and \
                    (edge_list_flat[i, 1] >= num and
                     edge_list_flat[i, 1] <= 2 * num - 1):
                    removed_elements.append(i)
            res = np.delete(edge_list_flat, removed_elements,
                            axis=0).transpose()
            res[0] = res[0] + i_index - 1
            res[1] = res[1] - num + j_index - 1

        edges = [(res[0, k], res[1, k]) for k in range(0, res.shape[1])]
        graph.add_edges_from(edges)

        # partial connected components

        subgraphs = nx.connected_components(graph)
        comp = [g for g in subgraphs]
        return comp

    def _single_frame(self, scheduler_kwargs, n_blocks, cutoff=15.0):
        """Perform computation on a single trajectory frame.

        Must return computed values as a list. You can only **read**
        from member variables stored in ``self``. Changing them during
        a run will result in undefined behavior. `ts` and any of the
        atomgroups can be changed (but changes will be overwritten
        when the next time step is read).

        Parameters
        ----------
        scheduler_kwargs : Dask Scheduler parameters.
        cutoff : float (optional)
            head group-defining atoms within a distance of `cutoff`
            Angstroms are deemed to be in the same leaflet [15.0]

        Returns
        -------
        values : anything
            The output from the computation over a single frame must
            be returned. The `value` will be added to a list for each
            block and the list of blocks is stored as :attr:`_results`
            before :meth:`_conclude` is run. In order to simplify
            processing, the `values` should be "simple" shallow data
            structures such as arrays or lists of numbers.

        """

        # Get positions of the atoms in the atomgroup and find their number.
        atoms = self._atomgroup.positions
        matrix_size = atoms.shape[0]
        arraged_coord = list()
        part_size = int(matrix_size / n_blocks)
        # Partition the data based on a 2-dimensional partitioning
        for i in range(1, matrix_size + 1, part_size):
            for j in range(1, matrix_size + 1, part_size):
                arraged_coord.append(([atoms[i - 1:i - 1 + part_size],
                                       atoms[j - 1:j - 1 + part_size]],
                                      [i, j]))

        # Distribute the data over the available cores, apply the map function
        # and execute.
        parAtoms = db.from_sequence(arraged_coord,
                                    npartitions=len(arraged_coord))
        parAtomsMap = parAtoms.map_partitions(self._find_parcc,cutoff=cutoff)
        Components = parAtomsMap.compute(**scheduler_kwargs)

        # Gather the results and start the reduction. TODO: think if it can go
        # to the private _reduce method of the based class.
        result = list(Components)
        # Create the overall connected components of the graph
        while len(result) != 0:
            item1 = result[0]
            result.pop(0)
            ind = []
            for i, item2 in enumerate(Components):
                if item1.intersection(item2):
                    item1 = item1.union(item2)
                    ind.append(i)
            ind.reverse()
            [Components.pop(j) for j in ind]
            Components.append(item1)

        # Change output for and return.
        indices = [np.sort(list(g)) for g in Components]
        return indices

    def run(self,
            start=None,
            stop=None,
            step=None,
            scheduler=None,
            n_jobs=1,
            n_blocks=None,
            cutoff=15.0):
        """Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        scheduler : dask scheduler, optional
            Use dask scheduler, defaults to multiprocessing. This can be used
            to spread work to a distributed scheduler
        n_jobs : int, optional
            number of tasks to start, if `-1` use number of logical cpu cores.
            This argument will be ignored when the distributed scheduler is
            used
        n_blocks : int, optional
            number of partitions to divide trajectory frame into. If ``None``
            set equal to sqrt(n_jobs) or number of available workers in
            scheduler.

        """
        if scheduler is None:
            scheduler = multiprocessing

        if n_jobs == -1:
            n_jobs = cpu_count()

        if n_blocks is None:
            if scheduler == multiprocessing:
                n_blocks = n_jobs
            elif isinstance(scheduler, distributed.Client):
                n_blocks = len(scheduler.ncores())
            else:
                raise ValueError(
                    "Couldn't guess ideal number of blocks from scheduler."
                    "Please provide `n_blocks` in call to method.")

        universe = mda.Universe(self._top, self._traj)
        scheduler_kwargs = {'get': scheduler.get}
        if scheduler == multiprocessing:
            scheduler_kwargs['num_workers'] = n_jobs

        start, stop, step = self._trajectory.check_slice_indices(
            start, stop, step)
        n_frames = len(range(start, stop, step))
        with timeit() as total:
            with self.readonly_attributes():
                frames = list()
                timings = list()
                for frame in range(start, stop, step):
                    with timeit() as b_compute:
                        components = self. \
                               _single_frame(scheduler_kwargs=scheduler_kwargs,
                                             n_blocks=n_blocks,
                                             cutoff=cutoff)
                    leaflet1 = self._atomgroup[components[0]]
                    leaflet2 = self._atomgroup[components[1]]
                    timings.append(b_compute.elapsed)
                    self._results.append([leaflet1, leaflet2])
            with timeit() as conclude:
                self._conclude()
        self.timing = Timing(
            np.hstack(timings), total.elapsed, conclude.elapsed)
        return self

    def _conclude(self):
        self.results = self._results
