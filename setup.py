# coding: utf-8

from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name='motifwalk',
    version='0.1.0',
    description="Motif-aware random walk for graph embedding",
    long_description=readme,
    author="Hoang Nguyen",
    author_email="hoangnt@net.c.titech.ac.jp",
    url="https://gear.github.io/mage",
    license=license,
    packages=find_packages(exclude=('tests','docs'))
)
