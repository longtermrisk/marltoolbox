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
        'ray[rllib]==1.0.0', 'gym==0.17.3', 'torch==1.7.0', 'tensorboard==1.15.0',
        'numba==0.51.2', 'matplotlib==3.3.2', "pytest"
    ],
)
