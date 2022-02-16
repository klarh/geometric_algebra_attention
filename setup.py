#!/usr/bin/env python

import os
from setuptools import setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

version_fname = os.path.join(THIS_DIR, 'geometric_algebra_attention', 'version.py')
with open(version_fname) as version_file:
    exec(version_file.read())

readme_fname = os.path.join(THIS_DIR, 'README.md')
with open(readme_fname) as readme_file:
    long_description = readme_file.read()

setup(name='geometric_algebra_attention',
      author='Matthew Spellings',
      author_email='matthew.p.spellings@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Library for geometric algebra-based attention mechanisms',
      extras_require={},
      install_requires=[],
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=[
          'geometric_algebra_attention',
          'geometric_algebra_attention.base',
          'geometric_algebra_attention.jax',
          'geometric_algebra_attention.keras',
          'geometric_algebra_attention.pytorch',
          'geometric_algebra_attention.tensorflow',
      ],
      project_urls={
          'Documentation': 'http://geometric_algebra_attention.readthedocs.io/',
          'Source': 'https://github.com/klarh/geometric_algebra_attention',
          },
      python_requires='>=3',
      version=__version__
      )
