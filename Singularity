Bootstrap: docker
From: ubuntu:bionic


%help
    A Singularity image for bempp-cl

%setup 
    echo "SETUP"

%environment
    PATH=/usr/local/bin:/opt/miniconda/envs/bempp/bin:/usr/bin:/usr/sbin:/sbin:/bin

%post

    apt-get update
    apt-get -y install software-properties-common wget gmsh git binutils build-essential

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
    ls
    bash ./miniconda.sh -b -p /opt/miniconda
    export PATH=/opt/miniconda/bin:$PATH
    conda create --yes -n bempp python=3.7
    conda install -n bempp --yes numpy scipy matplotlib numba pytest jupyter plotly git pip
    conda install -n bempp --yes -c conda-forge pocl pyopencl fenics
    conda install -n bempp --yes "libblas=*=*mkl"
    /opt/miniconda/envs/bempp/bin/pip install meshio
    /opt/miniconda/envs/bempp/bin/pip install git+git://github.com/bempp/bempp-cl@master

    ln -s /opt/miniconda/envs/bempp/bin/python /usr/local/bin/python
    ln -s /opt/miniconda/envs/bempp/bin/ipython /usr/local/bin/ipython
    ln -s /opt/miniconda/envs/bempp/bin/jupyter /usr/local/bin/jupyter


%runscript
    exec "$@"
