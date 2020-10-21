#!/usr/bin/env python
# -*-coding:utf-8-*-

import os
from setuptools import setup, find_packages
import fenci

REQUIREMENTS = []
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fenci',
    version=fenci.__version__,
    description='中文分词',
    url='https://github.com/a358003542/fenci',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='wanze',
    author_email='a358003542@gmail.com',
    maintainer='wanze',
    maintainer_email='a358003542@gmail.com',
    license='MIT',
    platforms='Linux,windows',
    keywords=['word_segment', 'nlp'],
    classifiers=['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: MIT License',
                 'Environment :: Console',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Text Processing',
                 'Programming Language :: Python :: 3.6'],
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    include_package_data=True,
    install_requires=REQUIREMENTS,
)
