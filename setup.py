from setuptools import setup
import os


VERSION_FILE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "VERSION")
)

__version__ = open(VERSION_FILE).readlines()[0].strip("\n")

setup(
        name='Bempp-cl',
        version=__version__,
        packages=['bempp'],
        license='MIT',
        include_package_data=True,
)

