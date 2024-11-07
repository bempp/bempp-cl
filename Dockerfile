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

FROM ubuntu:24.04 as bempp-dev-env
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
        python3-venv \
        python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV VIRTUAL_ENV /bempp-env
ENV PATH /bempp-env/bin:$PATH
RUN python3 -m venv ${VIRTUAL_ENV}

# Install Python packages (via pip)
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

FROM ubuntu:24.04 as bempp-dev-env-numba
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
        python3-venv \
        python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV VIRTUAL_ENV /bempp-env
ENV PATH /bempp-env/bin:$PATH
RUN python3 -m venv ${VIRTUAL_ENV}

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

# Install Python packages (via pip)
RUN python3 -m pip install --no-cache-dir nanobind scikit-build-core[pyproject] && \
    python3 -m pip install --no-cache-dir matplotlib pyopencl numpy scipy numba meshio pyopencl && \
    python3 -m pip install --no-cache-dir flake8 pytest pydocstyle pytest-xdist

# Install UFL, Basix and FFCx
RUN python3 -m pip install --no-cache-dir ipython && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ufl.git@${FENICSX_UFL_TAG} && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/basix.git@${FENICSX_BASIX_TAG} && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ffcx.git@${FENICSX_FFCX_TAG}

ENV PETSC_ARCH=linux-gnu-complex128-32

# Install DOLFINx
RUN git clone --depth 1 --branch ${FENICSX_DOLFINX_TAG} https://github.com/fenics/dolfinx.git
RUN mkdir dolfinx/build
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DDOLFINX_ENABLE_PETSC=true -B dolfinx/build -S dolfinx/cpp
RUN cmake --build dolfinx/build && cmake --install dolfinx/build
ENV LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/dolfinx-complex/bin:$PATH
ENV PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:$PKG_CONFIG_PATH
ENV CMAKE_PREFIX_PATH=/usr/local/dolfinx-complex/lib/cmake:$CMAKE_PREFIX_PATH
RUN cd dolfinx/python && \
    python3 -m pip install -r build-requirements.txt && \
    python3 -m pip install --check-build-dependencies --no-build-isolation --config-settings=cmake.build-type="Release" .

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
RUN python3 -m pip install --no-cache-dir nanobind scikit-build-core[pyproject] && \
    python3 -m pip install --no-cache-dir meshio numpy matplotlib

# Install UFL, Basix and FFCx
RUN python3 -m pip install --no-cache-dir ipython && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ufl.git@${FENICSX_UFL_TAG} && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/basix.git@${FENICSX_BASIX_TAG} && \
    python3 -m pip install --no-cache-dir git+https://github.com/FEniCS/ffcx.git@${FENICSX_FFCX_TAG}

ENV PETSC_ARCH=linux-gnu-complex128-32

# Install DOLFINx
RUN git clone --depth 1 --branch ${FENICSX_DOLFINX_TAG} https://github.com/fenics/dolfinx.git
RUN mkdir dolfinx/build
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DDOLFINX_ENABLE_PETSC=true -B dolfinx/build -S dolfinx/cpp
RUN cmake --build dolfinx/build && cmake --install dolfinx/build
ENV LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/dolfinx-complex/bin:$PATH
ENV PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:$PKG_CONFIG_PATH
ENV CMAKE_PREFIX_PATH=/usr/local/dolfinx-complex/lib/cmake:$CMAKE_PREFIX_PATH
RUN cd dolfinx/python && \
    python3 -m pip install -r build-requirements.txt && \
    python3 -m pip install --check-build-dependencies --no-build-isolation --config-settings=cmake.build-type="Release" .

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
RUN python3 -m pip install --no-cache-dir jupytext nbconvert ipykernel
RUN python3 bempp-cl/examples/generate_notebooks.py
RUN cp -r bempp-cl/examples/notebooks /root/example_notebooks
RUN python3 -m pip uninstall -y jupytext

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
RUN python3 -m pip install --no-cache-dir jupytext
RUN python3 bempp-cl/examples/generate_notebooks.py
RUN cp -r bempp-cl/examples/notebooks /root/example_notebooks
RUN python3 -m pip uninstall -y jupytext

# Clear /tmp
RUN rm -rf /tmp/*

WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    python3 -m pip install --no-cache-dir jupyter jupyterlab plotly
EXPOSE 8888/tcp

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
