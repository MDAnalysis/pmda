
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

from dask import distributed
import dask
import pytest


@pytest.fixture(scope="session", params=(1, 2))
def client(tmpdir_factory, request):
    with tmpdir_factory.mktemp("dask_cluster").as_cwd():
        lc = distributed.LocalCluster(n_workers=request.param, processes=True)
        client = distributed.Client(lc)

        yield client

        client.close()
        lc.close()


@pytest.fixture(scope='session', params=('distributed',
                                         'processes',
                                         'synchronous'))
def scheduler(request, client):
    if request.param == 'distributed':
        arg = client
    else:
        arg = request.param
    with dask.config.set(scheduler=arg):
        yield
