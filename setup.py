from setuptools import setup

from bempp import version

setup(
        name='Bempp-cl',
        version=version.__version__,
        packages=['bempp'],
        license='MIT',
        include_package_data=True,
)

