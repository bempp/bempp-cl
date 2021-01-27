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

ARG DOLFINX_CMAKE_CXX_FLAGS
ARG DOLFINX_CMAKE_BUILD_TYPE=Release
ARG TINI_VERSION=0.19.0
ARG EXAFMM_VERSION=v0.1.0
ARG PETSC_ARCH=linux-gnu-complex-32

########################################

FROM dolfinx/dev-env as bempp-dev-env
LABEL maintainer="Bempp <bempp@googlegroups.org>"
LABEL description="Bempp-cl development environment"

ARG DOLFINX_CMAKE_BUILD_TYPE
ARG PETSC_ARCH

WORKDIR /tmp

# Install dependencies available via apt-get.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    python3-pyopencl \
    python3-mpi4py \
    pkg-config \
    python-is-python3 \
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
    libxinerama1

# Install Python packages (via pip)
RUN pip3 install --no-cache-dir numba meshio>=4.0.16 && \
    pip3 install --no-cache-dir flake8 pytest pydocstyle pytest-xdist

# Install FEniCSx componenets
RUN pip3 install --no-cache-dir ipython && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/basix.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git

# Install DOLFIN-X
RUN	 git clone https://github.com/FEniCS/dolfinx.git && \
	 cd dolfinx/ && \
	 mkdir -p build && \
	 cd build && \
	 PETSC_ARCH=${PETSC_ARCH} cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} ../cpp/ && \
	 ninja -j3 install

# Build Python layer
RUN cd dolfinx/python && \
	pip3 -v install .

# Use FEniCSx in complex by default.
ENV LD_LIBRARY_PATH=/usr/local/dolfinx/lib:$LD_LIBRARY_PATH \
        PATH=/usr/local/dolfinx/bin:$PATH \
        PKG_CONFIG_PATH=/usr/local/dolfinx/lib/pkgconfig:$PKG_CONFIG_PATH \
        PETSC_ARCH=${PETSC_ARCH} \
        PYTHONPATH=/usr/local/dolfinx/lib/python3.8/dist-packages:$PYTHONPATH

# Install older DOLFIN
RUN apt-get -y install python3-dolfin

# Clean
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
