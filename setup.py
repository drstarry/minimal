#!/usr/bin/python
# coding: utf-8

from setuptools import setup, find_packages


install_requires = [
    'click',
    'numpy',
    'matplotlib'
]

entry_points = '''
[console_scripts]
minimal = minimal.cli.cli:cli
'''

setup(name='minimal',
      version='0.0.1',
      author='Rui Dai | Starry',
      author_email='drstarry@gmail.com',
      description='minimal',
      license='PRIVATE',
      packages=find_packages(),
      install_requires=install_requires,
      entry_points=entry_points)
