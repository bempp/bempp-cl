from setuptools import setup, find_packages

from bempp import version

setup(
    name="Bempp-cl",
    version=version.__version__,
    author="Timo Betcke",
    author_email="timo.betcke@gmail.com",
    description="The Bempp boundary element library",
    url="https://github.com/bempp/bempp-cl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["numpy>=1.21.0", "scipy", "numba>=0.55.2", "meshio>=4.0.16"],
    extras_require={
       "optional": ["plotly", "pyopencl"],
       "lint": ["pydocstyle", "flake8", "pytest", "pytest-xdist"],
    }
)
