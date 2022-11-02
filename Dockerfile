# Dockerfile giving an environment in which Bempp-cl can be run
#
# Docker images created by this file are pushed to hub.docker.com/u/bempp
#
# Authors:
#   Matthew Scroggs <mws48@cam.ac.uk>
#   Timo Betcke <t.betcke@ucl.ac.uk>
#
# Based on the FEniCSx Docker file written by:
#   Jack S. Hale <jack.hale@uni.lu>
#   Lizao Li <lzlarryli@gmail.com>
#   Garth N. Wells <gnw20@cam.ac.uk>
#   Jan Blechta <blechta@karlin.mff.cuni.cz>
#

ARG GMSH_VERSION=4.6.0
ARG TINI_VERSION=0.19.0
ARG EXAFMM_VERSION=0.1.1
ARG FENICSX_BASIX_TAG=main
ARG FENICSX_FFCX_TAG=main
ARG FENICSX_DOLFINX_TAG=main
ARG FENICSX_UFL_TAG=main
ARG MAKEFLAGS


########################################

FROM ubuntu:20.04 as bempp-dev-env
LABEL maintainer="Matthew Scroggs <bempp@mscroggs.co.uk>"
LABEL description="Bempp-cl development environment"

ARG GMSH_VERSION
ARG MAKEFLAGS
ARG EXAFMM_VERSION

WORKDIR /tmp

# Install dependencies available via apt-get.
# - First set of packages are required to build and run Bempp-cl.
# - Second set of packages are recommended and/or required to build
#   documentation or tests.
# - Third set of packages are optional, but required to run gmsh
#   pre-built binaries.
# - Fourth set of packages are optional, required for meshio.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    cmake \
    git \
    ipython3 \
    pkg-config \
    python-is-python3 \
    python3-dev \
    python3-matplotlib \
    python3-mpi4py \
    python3-pip \
    python3-pyopencl \
    python3-scipy \
    python3-setuptools \
    jupyter \
    wget && \
    apt-get -y install \
    libfftw3-dev \
    libfltk-gl1.3 \
    libfltk-images1.3 \
    libfltk1.3 \
    libfreeimage3 \
    libgl2ps1.4 \
    libglu1-mesa \
    libilmbase24 \
    libjxr0 \
    libocct-data-exchange-7.3 \
    libocct-foundation-7.3 \
    libocct-modeling-algorithms-7.3 \
    libocct-modeling-data-7.3 \ 
    libocct-ocaf-7.3 \
    libocct-visualization-7.3 \
    libopenblas-dev \
    libopenexr24 \
    libopenjp2-7 \
    libraw19 \
    libtbb2 \
    libxcursor1 \
    libxinerama1 && \
    apt-get -y install \
    python3-lxml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages (via pip)
RUN pip3 install --no-cache-dir "numpy>=1.21,<1.23" numba>=0.55.2 meshio>=4.0.16 && \
    pip3 install --no-cache-dir flake8 pytest pydocstyle pytest-xdist

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

RUN git clone -b v${EXAFMM_VERSION} https://github.com/exafmm/exafmm-t.git
RUN cd exafmm-t && sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 setup.py install

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM ubuntu:20.04 as bempp-dev-env-with-dolfin
LABEL maintainer="Matthew Scroggs <bempp@mscroggs.co.uk>"
LABEL description="Bempp-cl development environment with FEniCS"

ARG GMSH_VERSION
ARG MAKEFLAGS
ARG EXAFMM_VERSION

WORKDIR /tmp

# Install dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    cmake \
    git \
    ipython3 \
    pkg-config \
    python-is-python3 \
    python3-dev \
    python3-mpi4py \
    python3-pip \
    python3-pyopencl \
    python3-setuptools \
    jupyter \
    wget && \
    apt-get -y install \
    libfftw3-dev \
    libfltk-gl1.3 \
    libfltk-images1.3 \
    libfltk1.3 \
    libfreeimage3 \
    libgl2ps1.4 \
    libglu1-mesa \
    libilmbase24 \
    libjxr0 \
    libocct-data-exchange-7.3 \
    libocct-foundation-7.3 \
    libocct-modeling-algorithms-7.3 \
    libocct-modeling-data-7.3 \ 
    libocct-ocaf-7.3 \
    libocct-visualization-7.3 \
    libopenblas-dev \
    libopenexr24 \
    libopenjp2-7 \
    libraw19 \
    libtbb2 \
    libxcursor1 \
    libxinerama1
RUN pip3 install --no-cache-dir meshio>=4.0.16 \
    numba numpy==1.20 scipy matplotlib && \
    pip3 install --no-cache-dir flake8 pytest pydocstyle pytest-xdist
RUN apt-get -y install \
    python3-dolfin \
    python3-lxml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages (via pip)

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

RUN git clone -b v${EXAFMM_VERSION} https://github.com/exafmm/exafmm-t.git
RUN cd exafmm-t && sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 setup.py install

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM dolfinx/dev-env:stable as bempp-dev-env-with-dolfinx
LABEL maintainer="Matthew Scroggs <bempp@mscroggs.co.uk>"
LABEL description="Bempp-cl development environment with FEniCSx"

ARG DOLFINX_MAKEFLAGS
ARG FENICSX_BASIX_TAG
ARG FENICSX_FFCX_TAG
ARG FENICSX_DOLFINX_TAG
ARG FENICSX_UFL_TAG
ARG BEMPP_VERSION
ARG EXAFMM_VERSION

WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    libpugixml-dev \
    python3-pyopencl \
    python3-pybind11 \
    libfftw3-dev \
    pkg-config \
    python-is-python3 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages (via pip)
RUN pip3 install --no-cache-dir meshio>=4.0.16 "numpy>=1.21,<1.23" matplotlib && \
    pip3 install --upgrade six

# Install Basix
RUN git clone --depth 1 --branch ${FENICSX_BASIX_TAG} https://github.com/FEniCS/basix.git basix-src && \
    cd basix-src && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    pip3 install ./python

# Install FEniCSx components
RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git@${FENICSX_UFL_TAG} && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git@${FENICSX_FFCX_TAG}

# Install FEniCSx
RUN git clone --depth 1 --branch ${FENICSX_DOLFINX_TAG} https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-complex-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex ../cpp && \
    ninja ${DOLFINX_MAKEFLAGS} install && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-complex-32 pip3 install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies --ignore-installed .

# complex by default.
ENV LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:$LD_LIBRARY_PATH \
        PATH=/usr/local/dolfinx-complex/bin:$PATH \
        PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:$PKG_CONFIG_PATH \
        PETSC_ARCH=linux-gnu-complex-32 \
        PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.8/dist-packages:$PYTHONPATH

# Download and install ExaFMM
RUN wget -nc --quiet https://github.com/exafmm/exafmm-t/archive/v${EXAFMM_VERSION}.tar.gz && \
    tar -xf v${EXAFMM_VERSION}.tar.gz && \
    cd exafmm-t-${EXAFMM_VERSION} && \
    sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 setup.py install

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM dolfinx/dev-env:stable as bempp-dev-env-with-dolfinx-numba
LABEL maintainer="Matthew Scroggs <bempp@mscroggs.co.uk>"
LABEL description="Bempp-cl development environment with FEniCSx (Numba only)"

ARG DOLFINX_MAKEFLAGS
ARG FENICSX_BASIX_TAG
ARG FENICSX_FFCX_TAG
ARG FENICSX_DOLFINX_TAG
ARG FENICSX_UFL_TAG
ARG BEMPP_VERSION
ARG EXAFMM_VERSION

WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    python3-pybind11 \
    python3-matplotlib \
    libfftw3-dev \
    pkg-config \
    python-is-python3 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages (via pip)
RUN pip3 install --no-cache-dir meshio>=4.0.16 "numpy>=1.21,<1.23" && \
    pip3 install --upgrade six

# Install Basix
RUN git clone --depth 1 --branch ${FENICSX_BASIX_TAG} https://github.com/FEniCS/basix.git basix-src && \
    cd basix-src && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    pip3 install ./python

# Install FEniCSx components
RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git@${FENICSX_UFL_TAG} && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git@${FENICSX_FFCX_TAG}

# Install FEniCSx
RUN git clone --depth 1 --branch ${FENICSX_DOLFINX_TAG} https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-complex-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex ../cpp && \
    ninja ${DOLFINX_MAKEFLAGS} install && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-complex-32 pip3 install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies --ignore-installed .

# complex by default.
ENV LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:$LD_LIBRARY_PATH \
        PATH=/usr/local/dolfinx-complex/bin:$PATH \
        PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:$PKG_CONFIG_PATH \
        PETSC_ARCH=linux-gnu-complex-32 \
        PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.8/dist-packages:$PYTHONPATH

# Download and install ExaFMM
RUN wget -nc --quiet https://github.com/exafmm/exafmm-t/archive/v${EXAFMM_VERSION}.tar.gz && \
    tar -xf v${EXAFMM_VERSION}.tar.gz && \
    cd exafmm-t-${EXAFMM_VERSION} && \
    sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 setup.py install

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM bempp-dev-env-with-dolfinx as with-dolfinx
LABEL description="Bempp-cl environment with FEniCSx"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 setup.py install

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM bempp-dev-env-with-dolfin as with-dolfin
LABEL description="Bempp-cl environment with FEniCS"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 setup.py install

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM bempp-dev-env-with-dolfinx as lab
LABEL description="Bempp Jupyter Lab"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 setup.py install
RUN cp -r bempp-cl/notebooks /root/example_notebooks
RUN rm /root/example_notebooks/conftest.py /root/example_notebooks/test_notebooks.py

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab plotly
EXPOSE 8888/tcp

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

########################################

FROM bempp-dev-env-with-dolfinx-numba as numba-lab
LABEL description="Bempp Jupyter Lab (Numba only)"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 setup.py install
RUN cp -r bempp-cl/notebooks /root/example_notebooks
RUN rm /root/example_notebooks/conftest.py /root/example_notebooks/test_notebooks.py

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab plotly
EXPOSE 8888/tcp

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

########################################

FROM bempp-dev-env-with-dolfin as fenics-lab
LABEL description="Bempp Jupyter Lab with legacy FEniCS"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && pip3 install .
RUN cp -r bempp-cl/notebooks /root/example_notebooks
RUN rm /root/example_notebooks/conftest.py /root/example_notebooks/test_notebooks.py

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab plotly
EXPOSE 8888/tcp

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
