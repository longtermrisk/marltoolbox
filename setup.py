import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="marltoolbox",
    version="0.0.6",
    packages=find_packages(),
    description="CLR MARL research framework toolbox",
    long_description=read("README.md"),
    license="MIT",
    install_requires=["ray[rllib]", "torch"],
    extras_require={
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
