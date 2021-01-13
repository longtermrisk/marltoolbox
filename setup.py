import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='marltoolbox',
    version='0.0.4',
    packages=find_packages(),
    description='CLR MARL research framework toolbox',
    long_description=read('README.md'),
    license='MIT',
    install_requires=[
        'ray[rllib]==1.0.0', 'gym', 'torch', 'tensorboard', 'numba', 'matplotlib',
        "pytest"
    ],
)
