import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="marltoolbox",
    version="0.0.5",
    packages=find_packages(),
    description="CLR MARL research framework toolbox",
    long_description=read("README.md"),
    license="MIT",
    install_requires=[
        "ray[rllib]>=1.2.0",
        "gym==0.17.3",
        "torch>=1.6.0,<=1.7.0",
        "tensorboard==1.15.0",
        "numba>=0.51.2",
        "matplotlib>=3.3.2",
        "wandb",
        "ordered-set",
        "seaborn==0.9.0",
        "tqdm",
    ],
    extras_require={
        "lola": [
            "click",
            "gym>=0.10.5",
            "mock",
            "numpy>=1.11",
            "dm-sonnet==1.20",
            "tensorflow>=1.8.0,<2.0.0",
            "trueskill",
        ],
        "ci": [
            "pytest",
            "pytest-cov",
            "flake8",
            "ipython",
            "notebook",
            "jupyter_contrib_nbextensions",
            "flaky",
            "pytest-xdist",
        ],
        "dev": ["black[d]"],
    },
)
