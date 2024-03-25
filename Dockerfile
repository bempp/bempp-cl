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
#

ARG GMSH_VERSION=4.11.1
ARG TINI_VERSION=0.19.0
ARG EXAFMM_VERSION=0.1.1
ARG FENICSX_BASIX_TAG=main
ARG FENICSX_FFCX_TAG=main
ARG FENICSX_DOLFINX_TAG=main
ARG FENICSX_UFL_TAG=main
ARG MAKEFLAGS


########################################

FROM ubuntu:22.04 as bempp-dev-env
LABEL maintainer="Matthew Scroggs <bempp@mscroggs.co.uk>"
LABEL description="Bempp-cl development environment"

ARG GMSH_VERSION
ARG MAKEFLAGS
ARG EXAFMM_VERSION

WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        wget \
        git \
        pkg-config \
        build-essential \
        # ExaFMM dependencies
        libfftw3-dev \
        libopenblas-dev \
        # Gmsh dependencies
        libfltk-gl1.3 \
        libfltk-images1.3 \
        libfltk1.3 \
        libglu1-mesa \
        # OpenCL
        libpocl-dev \
        # Python
        python3-dev \
        python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 -m pip install --no-cache-dir matplotlib pyopencl numpy scipy numba meshio && \
    python3 -m pip install --no-cache-dir flake8 pytest pydocstyle pytest-xdist

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

RUN git clone -b v${EXAFMM_VERSION} https://github.com/exafmm/exafmm-t.git
RUN cd exafmm-t && sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 -m pip install .

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM ubuntu:22.04 as bempp-dev-env-numba
LABEL maintainer="Matthew Scroggs <bempp@mscroggs.co.uk>"
LABEL description="Bempp-cl development environment"

ARG GMSH_VERSION
ARG MAKEFLAGS
ARG EXAFMM_VERSION

WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        wget \
        git \
        pkg-config \
        build-essential \
        # ExaFMM dependencies
        libfftw3-dev \
        libopenblas-dev \
        # Gmsh dependencies
        libfltk-gl1.3 \
        libfltk-images1.3 \
        libfltk1.3 \
        libglu1-mesa \
        # Python
        python3-dev \
        python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 -m pip install --no-cache-dir matplotlib numpy scipy numba meshio && \
    python3 -m pip install --no-cache-dir flake8 pytest pydocstyle pytest-xdist

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

RUN git clone -b v${EXAFMM_VERSION} https://github.com/exafmm/exafmm-t.git
RUN cd exafmm-t && sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 -m pip install .

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM ghcr.io/fenics/test-env:current-openmpi as bempp-dev-env-with-dolfinx
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
        # OpenCL
        libpocl-dev \
        # ExaFMM dependencies
        libfftw3-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 -m pip install --no-cache-dir matplotlib pyopencl numpy scipy numba meshio && \
    python3 -m pip install --no-cache-dir flake8 pytest pydocstyle pytest-xdist

# Install Python packages (via pip)
RUN python3 -m pip install --no-cache-dir meshio numpy matplotlib pyopencl

# Install Basix
RUN git clone --depth 1 --branch ${FENICSX_BASIX_TAG} https://github.com/FEniCS/basix.git basix-src && \
    cd basix-src && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    python3 -m pip install ./python

# Install FEniCSx components
RUN python3 -m pip install --no-cache-dir ipython && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ufl.git@${FENICSX_UFL_TAG} && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ffcx.git@${FENICSX_FFCX_TAG}

# Install FEniCSx
RUN git clone --depth 1 --branch ${FENICSX_DOLFINX_TAG} https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-complex64-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex ../cpp && \
    ninja ${DOLFINX_MAKEFLAGS} install && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-complex64-32 python3 -m pip install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies --ignore-installed .

# complex by default.
ENV LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:$LD_LIBRARY_PATH \
        PATH=/usr/local/dolfinx-complex/bin:$PATH \
        PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:$PKG_CONFIG_PATH \
        PETSC_ARCH=linux-gnu-complex64-32 \
        PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.8/dist-packages:$PYTHONPATH

# Download and install ExaFMM
RUN wget -nc --quiet https://github.com/exafmm/exafmm-t/archive/v${EXAFMM_VERSION}.tar.gz && \
    tar -xf v${EXAFMM_VERSION}.tar.gz && \
    cd exafmm-t-${EXAFMM_VERSION} && \
    sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 -m pip install .

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM ghcr.io/fenics/test-env:current-openmpi as bempp-dev-env-with-dolfinx-numba
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
    # ExaFMM dependencies
    libfftw3-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages (via pip)
RUN python3 -m pip install --no-cache-dir meshio numpy matplotlib

# Install Basix
RUN git clone --depth 1 --branch ${FENICSX_BASIX_TAG} https://github.com/FEniCS/basix.git basix-src && \
    cd basix-src && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    python3 -m pip install ./python

# Install FEniCSx components
RUN python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ufl.git@${FENICSX_UFL_TAG} && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ffcx.git@${FENICSX_FFCX_TAG}

# Install FEniCSx
RUN git clone --depth 1 --branch ${FENICSX_DOLFINX_TAG} https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-complex64-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex ../cpp && \
    ninja ${DOLFINX_MAKEFLAGS} install && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-complex64-32 python3 -m pip install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies --ignore-installed .

# complex by default.
ENV LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:$LD_LIBRARY_PATH \
        PATH=/usr/local/dolfinx-complex/bin:$PATH \
        PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:$PKG_CONFIG_PATH \
        PETSC_ARCH=linux-gnu-complex64-32 \
        PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.8/dist-packages:$PYTHONPATH

# Download and install ExaFMM
RUN wget -nc --quiet https://github.com/exafmm/exafmm-t/archive/v${EXAFMM_VERSION}.tar.gz && \
    tar -xf v${EXAFMM_VERSION}.tar.gz && \
    cd exafmm-t-${EXAFMM_VERSION} && \
    sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 -m pip install .

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM bempp-dev-env-with-dolfinx as with-dolfinx
LABEL description="Bempp-cl environment with FEniCSx"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 -m pip install .

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

########################################

FROM bempp-dev-env-with-dolfinx as lab
LABEL description="Bempp Jupyter Lab"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 -m pip install .
RUN cp -r bempp-cl/notebooks /root/example_notebooks
RUN rm /root/example_notebooks/conftest.py /root/example_notebooks/test_notebooks.py

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    python3 -m pip install --no-cache-dir jupyter jupyterlab plotly
EXPOSE 8888/tcp

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

########################################

FROM bempp-dev-env-with-dolfinx-numba as numba-lab
LABEL description="Bempp Jupyter Lab (Numba only)"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 -m pip install .
RUN cp -r bempp-cl/notebooks /root/example_notebooks
RUN rm /root/example_notebooks/conftest.py /root/example_notebooks/test_notebooks.py

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    python3 -m pip install --no-cache-dir jupyter jupyterlab plotly
EXPOSE 8888/tcp

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
