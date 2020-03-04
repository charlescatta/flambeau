#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='Flambeau',
      setup_requires=['setuptools-pipfile'],
      use_pipfile=True,
      description='Utilities for PyTorch',
      author='Charles Catta',
      python_requires='>=3.7',
      author_email='charles.catta+git@gmail.com',
      packages=find_packages(include='./flambeau'))



