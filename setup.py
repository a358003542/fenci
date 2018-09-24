#!/usr/bin/env python
# -*-coding:utf-8-*-


from setuptools import setup, find_packages
import bihu_word_segment
import codecs

REQUIREMENTS = []


def long_description():
    with codecs.open('README.md', encoding='utf-8') as f:
        return f.read()


setup(
    name='bihu_word_segment',
    version=bihu_word_segment.__version__,
    description='中文分词',
    url='https://github.com/a358003542/bihu_word_segment',
    long_description=long_description(),
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
    setup_requires=REQUIREMENTS,
    install_requires=REQUIREMENTS,
)
