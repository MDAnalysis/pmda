# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

from setuptools import setup
import versioneer

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='pmda',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Parallel Molecular Analysis Tools',
    long_description=long_description,
    author='Max Linke',
    license="GPLv2",
    download_url="https://github.com/MDAnalysis/pmda/releases",
    keywords="science parallel",
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'MDAnalysis>=0.18',
        'dask',
        'six',
        'joblib',  # cpu_count func currently
    ],
    tests_require=[
        'pytest',
        'MDAnalysisTests>=0.18',  # keep
    ], )
