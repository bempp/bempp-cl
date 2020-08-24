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
ARG EXAFMM_VERSION=v0.1.0
ARG MAKEFLAGS

# TODO: once ExaFMM has a release number, use this to wget the correct version
# ARG EXAFMM_VERSION = ## 2020-08-07

########################################

FROM ubuntu:20.04 as bempp-dev-env
LABEL maintainer="Bempp <bempp@googlegroups.org>"
LABEL description="Bempp-cl development environment"

ARG GMSH_VERSION
ARG MAKEFLAGS

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
    python3-numpy \
    python3-pip \
    python3-pyopencl \
    python3-scipy \
    python3-setuptools \
    jupyter \
    wget && \
    apt-get -y install \
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
    python3-dolfin && \
    apt-get -y install \
    python3-lxml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages (via pip)
RUN pip3 install --no-cache-dir numba meshio>=4.0.16 && \
    pip3 install --no-cache-dir flake8 pytest pydocstyle

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

WORKDIR /root

########################################

FROM bempp-dev-env AS bempp-dev-env-with-pyexafmm

WORKDIR /tmp
RUN git clone https://github.com/exafmm/pyexafmm.git
RUN cd pyexafmm && python3 setup.py install

WORKDIR /root

########################################

FROM bempp-dev-env AS bempp-dev-env-with-exafmm

ARG EXAFMM_VERSION

WORKDIR /tmp
RUN git clone -b ${EXAFMM_VERSION} https://github.com/exafmm/exafmm-t.git
RUN cd exafmm-t && sed -i 's/march=native/march=ivybridge/g' ./setup.py && python3 setup.py install

WORKDIR /root

########################################
FROM bempp-dev-env-with-exafmm as lab
LABEL description="Bempp Jupyter Lab"

WORKDIR /tmp
RUN git clone https://github.com/bempp/bempp-cl
RUN cd bempp-cl && python3 setup.py install
RUN cp -r bempp-cl/notebooks /root/example_notebooks

WORKDIR /root

ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --no-cache-dir jupyter jupyterlab
EXPOSE 8888/tcp

ENTRYPOINT ["/tini", "--", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
