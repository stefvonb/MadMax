#!/bin/sh

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
EXEC_PREFIX=$SCRIPTPATH/.env

LHAPDF_VERSION=6.2.3

mkdir -p $EXEC_PREFIX

if command -v lhapdf-config &> /dev/null; then
    CURRENT_LHAPDF_VERSION=$(lhapdf-config --version)
fi
if [ $LHAPDF_VERSION = $CURRENT_LHAPDF_VERSION ]; then
    echo LHAPDF version $LHAPDF_VERSION already installed
else
    curl https://lhapdf.hepforge.org/downloads/?f=LHAPDF-${LHAPDF_VERSION}.tar.gz --output $EXEC_PREFIX/LHAPDF-${LHAPDF_VERSION}.tar.gz
    cd $EXEC_PREFIX
    tar xzf LHAPDF-${LHAPDF_VERSION}.tar.gz
    rm LHAPDF-${LHAPDF_VERSION}.tar.gz
    cd LHAPDF-${LHAPDF_VERSION} && ./configure --prefix $EXEC_PREFIX && make && make install
    cd $SCRIPTPATH
fi

if ! command -v cmake &> /dev/null || [ ! -d "$EXEC_PREFIX/CMake" ]; then
     cd $EXEC_PREFIX
     git clone https://github.com/Kitware/CMake.git    
     cd CMake
     ./bootstrap --prefix=$EXEC_PREFIX && make && make install
     cd $SCRIPTPATH
 else
     echo CMake already installed
 fi

export PATH=$EXEC_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$EXEC_PREFIX/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$EXEC_PREFIX/lib:$DYLD_LIBRARY_PATH

if [ -d "$EXEC_PREFIX/nlopt" ]; then
    echo NLOPT already installed
else
    cd $EXEC_PREFIX
    git clone git://github.com/stevengj/nlopt
    cd nlopt
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$EXEC_PREFIX .. && make && make install
    cd $SCRIPTPATH
fi

PYTHON_LIB_DIR=`find $EXEC_PREFIX/lib -name "python*"`
echo export PATH=$EXEC_PREFIX/bin:\$PATH > env.sh
echo export LD_LIBRARY_PATH=$EXEC_PREFIX/lib:\$LD_LIBRARY_PATH >> env.sh
echo export DYLD_LIBRARY_PATH=$EXEC_PREFIX/lib:\$DYLD_LIBRARY_PATH >> env.sh
echo export PYTHONPATH=$EXEC_PREFIX/$PYTHON_LIB_DIR/site-packages:\$PYTHONPATH >> env.sh

echo
echo Finished... to use the installed dependencies, please do the following:
echo -----------------------------------------------------------------------
cat env.sh
echo
echo \(or just source env.sh\)
echo 


