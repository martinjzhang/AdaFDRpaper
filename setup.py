from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path

setup(name='nfdr2',
    version='0.1.0',
    description='Stanford University',
    url='https://github.com/fxia22/realenv',
    author='Stanford University',
    zip_safe=False)