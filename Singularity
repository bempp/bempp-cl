Bootstrap: docker
From: ubuntu:bionic


%help
    A Singularity image for bempp-cl

%setup 
    echo "SETUP"

%environment
    PYTHONPATH=/home/bempp-cl

%post
    apt-get update
    apt-get -y dist-upgrade
    apt-get -y install python3-pip
    apt-get -y install software-properties-common \
                       libopenblas-base \
                       ocl-icd-opencl-dev \
                       build-essential \
                       cmake \
                       git \
                       pkg-config \
                       libclang-6.0-dev \
                       clang-6.0 llvm-6.0 \
                       make \
                       ninja-build \
                       ocl-icd-libopencl1 \
                       ocl-icd-dev \
                       ocl-icd-opencl-dev \
                       libhwloc-dev \
                       zlib1g \
                       zlib1g-dev \
                       clinfo \
                       dialog \
                       apt-utils \
                       gmsh
    add-apt-repository -y ppa:fenics-packages/fenics
    apt-get update && apt-get install -y --no-install-recommends fenics && apt-get -y dist-upgrade

    pip3 install pybind11 matplotlib numpy scipy jupyter plotly pytest pyopencl numba meshio
    ipython3 kernel install
    jupyter nbextension enable --py widgetsnbextension

    cd /home
    git clone https://github.com/pocl/pocl.git
    cd /home/pocl
    git checkout release_1_4
    mkdir build && cd build && cmake -G Ninja -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-6.0 -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release ..
    ninja install

    cd /home
    git clone https://github.com/bempp/bempp-cl.git

%runscript

    exec "$@"



    
                       





