1. clone github repos sofa_env and sofa_zoo
2. pip install both
3. 
make conda env 
conda activate /home/btl/NewVolume/Jackson/conda_sofa

ls $FOLDER_TARGET/lib/libSofa.Component.AnimationLoop.*
FOLDER_SRC=~/NewVolume/Jackson/sofa/src
FOLDER_TARGET=~/NewVolume/Jackson/sofa/build
FOLDER_SP3=$FOLDER_SRC/applications/plugins/SofaPython3

PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PYTHON_EXE=$(which python3)
PYTHON_ROOT_DIR=$CONDA_PREFIX

SOFA_ROOT=~/NewVolume/Jackson/lap_gym/sofa_env/SOFA
SOFAPYTHON3_ROOT=$SOFA_ROOT/plugins/SofaPython3

4. may need to apt install some packages:
sudo apt -y install build-essential software-properties-common python3-software-properties
sudo apt -y install libboost-all-dev
sudo apt -y install libpng-dev libjpeg-dev libtiff-dev libglew-dev zlib1g-dev
sudo apt -y install libeigen3-dev
sudo apt -y install libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
sudo apt install qttools5-dev qttools5-dev-tools qtwebengine5-dev libqt5opengl5-dev
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
     libnss3-dev libssl-dev libreadline-dev libffi-dev curl libbz2-dev
5. may need to do these if theres python issues:
cd /usr/src
sudo curl -O https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
sudo tar -xf Python-3.10.14.tgz
cd Python-3.10.14
sudo ./configure --enable-optimizations
sudo make -j$(nproc)
sudo make altinstall

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

sudo find /usr/local -name "libpython3.10.so.1.0"

cd /usr/src/Python-3.10.14
sudo make clean

sudo ./configure --enable-optimizations --enable-shared
sudo make -j$(nproc)
sudo make altinstall

ls /usr/local/lib/libpython3.10.so.1.0

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ldconfig  # refresh linker cache (optional but can help)
