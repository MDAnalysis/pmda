# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2018 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
LeafletFinder Analysis tool --- :mod:`pmda.leaflet`
===================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.leaflet`.

.. autoclass:: LeafletFinder
   :members:
   :undoc-members:
   :inherited-members:

"""
from __future__ import absolute_import, division

import numpy as np
import dask.bag as db
import networkx as nx
from scipy.spatial import cKDTree

import MDAnalysis as mda
import dask
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

    This version offers LeafletFinder algorithm 4 ("Tree-based Nearest
    Neighbor and Parallel-Connected Components (Tree-Search)") in
    [Paraskevakos2018]_.

    Currently, periodic boundaries are not taken into account.

    The calculation is parallelized on a per-frame basis;
    at the moment, no parallelization over trajectory blocks is performed.

    """

    def __init__(self, universe, atomgroups):
        self._atomgroup = atomgroups
        self._results = list()

        super(LeafletFinder, self).__init__(universe, (atomgroups,))

    def _find_connected_components(self, data, cutoff=15.0):
        """Perform the Connected Components discovery for the atoms in data.

        Parameters
        ----------
        data : Tuple of lists of Numpy arrays
            This is a data and index tuple. The data are organized as
            `([AtomPositions1<NumpyArray>,AtomPositions2<NumpyArray>],
            [index1,index2])`. `index1` and `index2` are showing the
            position of the `AtomPosition` in the adjacency matrix and
            allows to correct the node number of the produced graph.
        cutoff : float (optional)
            head group-defining atoms within a distance of `cutoff`
            Angstroms are deemed to be in the same leaflet [15.0]

        Returns
        -------
        values : list.
            A list of all the connected components of the graph that is
            generated from `data`

        """
        # pylint: disable=unsubscriptable-object
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
                     edge_list_flat[i, 1] <= 2 * num - 1) or \
                    (edge_list_flat[i, 0] >= num and
                     edge_list_flat[i, 0] <= 2 * num - 1) and \
                    (edge_list_flat[i, 1] >= 0 and
                     edge_list_flat[i, 1] <= num - 1):
                    removed_elements.append(i)
            res = np.delete(edge_list_flat, removed_elements,
                            axis=0).transpose()
            res[0] = res[0] + i_index - 1
            res[1] = res[1] - num + j_index - 1
        if res.shape[1] == 0:
            res = np.zeros((2, 1), dtype=np.int)

        edges = [(res[0, k], res[1, k]) for k in range(0, res.shape[1])]
        graph.add_edges_from(edges)

        # partial connected components

        subgraphs = nx.connected_components(graph)
        comp = [g for g in subgraphs]
        return comp

    # pylint: disable=arguments-differ
    def _single_frame(self, ts, atomgroups, scheduler_kwargs, n_jobs,
                      cutoff=15.0):
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
        atoms = ts.positions[atomgroups.indices]
        matrix_size = atoms.shape[0]
        arranged_coord = list()
        part_size = int(matrix_size / n_jobs)
        # Partition the data based on a 2-dimensional partitioning
        for i in range(1, matrix_size + 1, part_size):
            for j in range(i, matrix_size + 1, part_size):
                arranged_coord.append(([atoms[i - 1:i - 1 + part_size],
                                       atoms[j - 1:j - 1 + part_size]],
                                      [i, j]))
        # Distribute the data over the available cores, apply the map function
        # and execute.
        parAtoms = db.from_sequence(arranged_coord,
                                    npartitions=len(arranged_coord))
        parAtomsMap = parAtoms.map_partitions(self._find_connected_components,
                                              cutoff=cutoff)
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
            for j in ind:
                Components.pop(j)
            Components.append(item1)

        # Change output for and return.
        indices = [np.sort(list(g)) for g in Components]
        return indices

    # pylint: disable=arguments-differ
    def run(self,
            start=None,
            stop=None,
            step=None,
            n_jobs=-1,
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
        n_jobs : int, optional
            number of tasks to start, if `-1` use number of logical cpu cores.
            This argument will be ignored when the distributed scheduler is
            used

        """
        # are we using a distributed scheduler or should we use
        # multiprocessing?
        scheduler = dask.config.get('scheduler', None)
        if scheduler is None:
            # maybe we can grab a global worker
            try:
                scheduler = dask.distributed.worker.get_client()
            except ValueError:
                pass

        if n_jobs == -1:
            n_jobs = cpu_count()

        # we could not find a global scheduler to use and we ask for a single
        # job. Therefore we run this on the single threaded scheduler for
        # debugging.
        if scheduler is None and n_jobs == 1:
            scheduler = 'single-threaded'

        # fall back to multiprocessing, we tried everything
        if scheduler is None:
            scheduler = 'multiprocessing'

        scheduler_kwargs = {'scheduler': scheduler}
        if scheduler == 'multiprocessing':
            scheduler_kwargs['num_workers'] = n_jobs

        with timeit() as b_universe:
            universe = mda.Universe(self._top, self._traj)

        start, stop, step = self._trajectory.check_slice_indices(
            start, stop, step)
        with timeit() as total:
            with timeit() as prepare:
                self._prepare()

            with self.readonly_attributes():
                timings = list()
                times_io = []
                for frame in range(start, stop, step):
                    with timeit() as b_io:
                        ts = universe.trajectory[frame]
                    times_io.append(b_io.elapsed)
                    with timeit() as b_compute:
                        components = self. \
                               _single_frame(ts=ts,
                                             atomgroups=self._atomgroup,
                                             scheduler_kwargs=scheduler_kwargs,
                                             n_jobs=n_jobs,
                                             cutoff=cutoff)
                    timings.append(b_compute.elapsed)
                    leaflet1 = self._atomgroup[components[0]]
                    leaflet2 = self._atomgroup[components[1]]
                    self._results.append([leaflet1, leaflet2])
            with timeit() as conclude:
                self._conclude()
        self.timing = Timing(times_io,
                             np.hstack(timings), total.elapsed,
                             b_universe.elapsed, prepare.elapsed,
                             conclude.elapsed)
        return self

    def _conclude(self):
        self.results = self._results
