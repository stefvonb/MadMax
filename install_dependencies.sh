#!/bin/bash

LHAPDF_VERSION=6.2.3

mkdir -p dependencies

CURRENT_LHAPDF_VERSION=`lhapdf-config --version`
if [ $LHAPDF_VERSION == $CURRENT_LHAPDF_VERSION ]; then
    echo LHAPDF $LHAPDF_VERSION already installed
else
    wget https://lhapdf.hepforge.org/downloads/?f=LHAPDF-${LHAPDF_VERSION}.tar.gz -O dependencies/LHAPDF-${LHAPDF_VERSION}.tar.gz
    cd dependencies
    tar xzf LHAPDF-${LHAPDF_VERSION}.tar.gz
    rm LHAPDF-${LHAPDF_VERSION}.tar.gz
    cd LHAPDF-${LHAPDF_VERSION}
    ./configure --prefix $PWD/../
    make && make install
    cd ../../
fi

if [ `which cmake` == "" ] || [ ! -d "dependencies/CMake" ]; then
    cd dependencies
    git clone https://github.com/Kitware/CMake.git    
    cd CMake
    ./bootstrap --prefix=$PWD/../
    make && make install
    cd ../..
else
    echo CMake already installed
fi

export PATH=$PWD/dependencies/bin:$PATH
export LD_LIBRARY_PATH=$PWD/dependencies/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PWD/dependencies/lib:$DYLD_LIBRARY_PATH

if [ -d "dependencies/nlopt" ]; then
    echo NLOPT already installed
else
    cd dependencies
    git clone git://github.com/stevengj/nlopt
    cd nlopt
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$PWD/../.. ..
    make && make install
    cd ../../../
fi

PYTHON_LIB_DIR=`find dependencies/lib -name "python*"`
echo export PATH=$PWD/dependencies/bin:\$PATH > env.sh
echo export LD_LIBRARY_PATH=$PWD/dependencies/lib:\$LD_LIBRARY_PATH >> env.sh
echo export DYLD_LIBRARY_PATH=$PWD/dependencies/lib:\$DYLD_LIBRARY_PATH >> env.sh
echo export PYTHONPATH=$PWD/$PYTHON_LIB_DIR/site-packages:\$PYTHONPATH >> env.sh

echo
echo Finished... to use the installed dependencies, please do the following:
echo -----------------------------------------------------------------------
cat env.sh
echo
echo \(or just source env.sh\)
echo 


