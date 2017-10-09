from setuptools import setup

setup(
    name='pmda',
    version='0.1',
    description='Parallel Molecular Analysis Tools',
    author='Max Linke',
    install_requires=[
        'MDAnalysis>=0.16',
        'dask',
        'joblib',  # cpu_count func currently
    ],
    tests_require=[
        'pytest',
    ], )
