from setuptools import setup
import os

BEMPP_PATH = os.path.realpath(__file__)
__version__ = open(os.path.join(BEMPP_PATH, "./VERSION"), "r").readlines()[0].strip("\n")

setup(
        name='Bempp-cl',
        version=__version__,
        packages=['bempp'],
        license='MIT',
        include_package_data=True,
)

