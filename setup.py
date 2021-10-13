# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

from setuptools import setup, find_packages
import sys
import versioneer

# Make sure I have the right Python version.
if sys.version_info[:2] < (3, 6):
    print('PMDA requires Python 3.6 or better. Python {0:d}.{1:d} detected'.format(*
          sys.version_info[:2]))
    print('Please upgrade your version of Python.')
    sys.exit(-1)

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='pmda',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Parallel Molecular Dynamics Analysis tools',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/MDAnalysis/pmda/',
    author='Max Linke',
    author_email='max.linke88@gmail.com',
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
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        'Documentation': 'https://www.mdanalysis.org/pmda/',
        'Source': 'https://github.com/MDAnalysis/pmda/',
        'Issue Tracker': 'https://github.com/MDAnalysis/pmda/issues/',
        'Mailing list': 'https://groups.google.com/group/mdnalysis-discussion',
    },
    packages=find_packages(),
    install_requires=[
        #  'MDAnalysis>=2.0.0',
        'mdanalysis @ git+https://github.com/MDAnalysis/mdanalysis#egg=mdanalysis&subdirectory=package',
        'dask>=0.18',
        'distributed',
        'six',
        'joblib', # cpu_count func currently
        'networkx',
        'scipy',
    ],
    tests_require=[
        'pytest',
        #  'MDAnalysisTests>=2.0.0',  # keep
        'mdanalysistests @ git+https://github.com/MDAnalysis/mdanalysis#egg=mdanalysistests&subdirectory=testsuite'
    ], )
