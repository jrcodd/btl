This document will act as a log for all the coding stuff that I have done
Log:
conda env:
conda env create --file environment.yml --prefix ~/NewVolume/Jackson/conda_sofa

git clone https://github.com/ScheiklP/sofa_env.git 

Building SOFA: 

conda activate /home/btl/NewVolume/Jackson/conda_sofa


sudo apt -y install build-essential software-properties-common python3-software-properties
sudo apt -y install libboost-all-dev
sudo apt -y install libpng-dev libjpeg-dev libtiff-dev libglew-dev zlib1g-dev
sudo apt -y install libeigen3-dev
sudo apt -y install libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

FOLDER_SRC=~/NewVolume/Jackson/sofa/src
FOLDER_TARGET=~/NewVolume/Jackson/sofa/build
FOLDER_SP3=$FOLDER_SRC/applications/plugins/SofaPython3

PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PYTHON_EXE=$(which python3)
PYTHON_ROOT_DIR=$CONDA_PREFIX

mkdir -p $FOLDER_SRC
mkdir -p $FOLDER_TARGET
git clone https://github.com/sofa-framework/sofa.git $FOLDER_SRC
cd $FOLDER_SRC
git checkout v23.12
cd $FOLDER_SP3
git init
git remote add origin https://github.com/sofa-framework/SofaPython3.git
git pull origin master
git checkout f1ac0f03efd6f6e7c30df8b18259e16da523f0b2

python3 sofa_env/scenes/controllable_object_example/controllable_env.py

cmake -Wno-dev \
-S $FOLDER_SRC -B $FOLDER_TARGET \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
-DCMAKE_BUILD_TYPE=Release \
-DSOFA_FETCH_SOFAPYTHON3=OFF \
-DPLUGIN_SOFAPYTHON3=ON \
-DPython_EXECUTABLE=$PYTHON_EXE \
-DPython_ROOT_DIR=$PYTHON_ROOT_DIR \
-DSP3_LINK_TO_USER_SITE=ON \
-DSP3_PYTHON_PACKAGES_LINK_DIRECTORY=$PYTHON_PKG_PATH \
-DPLUGIN_SOFACARVING=ON \
-DSP3_BUILD_TEST=OFF \
-DSOFA_BUILD_TESTS=OFF \
-DSOFA_BUILD_GUI_QT=ON \
-DSOFA_BUILD_SOFACOMPONENT_USERINTERFACE_CONFIGURATIONSETTING=ON \

cmake --build $FOLDER_TARGET -j 32 --target install

echo "" | tee -a $FOLDER_TARGET/install/lib/plugin_list.conf.default
echo "$FOLDER_TARGET/install/plugins/SofaPython3/lib/libSofaPython3.so 1.0" | tee -a $FOLDER_TARGET/install/lib/plugin_list.conf.default

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHON_ROOT_DIR/lib

SOFA_ROOT=$FOLDER_TARGET/install
SOFAPYTHON3_ROOT=$FOLDER_TARGET/install/plugins/SofaPython3

conda env config vars set SOFA_ROOT=~/NewVolume/Jackson/sofa/build/install/
conda env config vars set SOFAPYTHON3_ROOT=~/NewVolume/Jackson/sofa/build/install/plugins/SofaPython3
conda install -c conda-forge gcc=12.1.0

python3 scenes/controllable_object_example/controllable_env.py

rm -rf $FOLDER_TARGET
mkdir -p $FOLDER_TARGET

conda install sofa-app sofa-python3 --channel sofa-framework --channel conda-forge

runSofa

Proposition:
Use commits from the date of the paper for sofa and sofapython3


---------------------------------------
Checking SOFA_ROOT and SOFAPYTHON3_ROOT
Using environment variable SOFA_ROOT: /home/btl/NewVolume/Jackson/sofa/build/install/
---------------------------------------
WARNING: Multiple instance of a singleton class
[WARNING] [ObjectFactory] cannot create template alias Vec1 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec2 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec3 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec6 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Rigid2 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Rigid3 as it already exists
[WARNING] [ObjectFactory] cannot create template alias CompressedRowSparseMatrix as it already exists
[WARNING] [ObjectFactory] cannot create template alias CompressedRowSparseMatrixMat3x3 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Mat2x2 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Mat3x3 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Mat4x4 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Mat6x6 as it already exists
[WARNING] [ObjectFactory] cannot create template alias Rigid as it already exists
[WARNING] [ObjectFactory] cannot create template alias Rigid2f as it already exists
[WARNING] [ObjectFactory] cannot create template alias Rigid3f as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec1f as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec2f as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec3f as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec6f as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec1d as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec2d as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec3d as it already exists
[WARNING] [ObjectFactory] cannot create template alias Vec6d as it already exists
[WARNING] [ObjectFactory] cannot create template alias Rigid2d as it already exists
[WARNING] [ObjectFactory] cannot create template alias Rigid3d as it already exists
[WARNING] [ObjectFactory] cannot create template alias float as it already exists
[WARNING] [ObjectFactory] cannot create template alias double as it already exists
[WARNING] [ObjectFactory] cannot create template alias vector<int> as it already exists
[WARNING] [ObjectFactory] cannot create template alias vector<unsigned_int> as it already exists
[WARNING] [ObjectFactory] cannot create template alias vector<float> as it already exists
[WARNING] [ObjectFactory] cannot create template alias vector<double> as it already exists
[WARNING] [ObjectFactory] cannot create template alias int as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<int> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<double> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<bool> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<Vec<2u,unsigned int>> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<Vec<2u,double>> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<Vec<3u,double>> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<Vec<4u,double>> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<RigidCoord<2u,double>> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<RigidDeriv<2u,double>> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<RigidCoord<3u,double>> as it already exists
[WARNING] [ObjectFactory] cannot create template alias Data<RigidDeriv<3u,double>> as it already exists
Segmentation fault (core dumped)

sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

-- Sofa.GUI.Qt: Qt5WebEngine not found, QDocBrowser will not be built.

sudo apt install qtbase5-dev qttools5-dev qttools5-dev-tools qtwebengine5-dev libqt5opengl5-dev

ldd ~/NewVolume/Jackson/lap_gym/sofa_env/SOFA/plugins/FreeMotionAnimationLoop/lib/libSofaPython3.so

sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
     libnss3-dev libssl-dev libreadline-dev libffi-dev curl libbz2-dev

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

-----------------------------------------------------------------
working!

steps to follow to re create:
# 1. clone lapgym repo
# 2. create/activate conda env 
conda activate /home/btl/NewVolume/Jackson/conda_sofa
# 3. clear sofa build folder
rm -rf $FOLDER_TARGET
mkdir -p $FOLDER_TARGET
# 4. clone and build sofa:
FOLDER_SRC=~/NewVolume/Jackson/sofa/src
FOLDER_TARGET=~/NewVolume/Jackson/sofa/build
FOLDER_SP3=$FOLDER_SRC/applications/plugins/SofaPython3

PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PYTHON_EXE=$(which python3)
PYTHON_ROOT_DIR=$CONDA_PREFIX

mkdir -p $FOLDER_SRC
mkdir -p $FOLDER_TARGET
git clone https://github.com/sofa-framework/sofa.git $FOLDER_SRC
cd $FOLDER_SRC
git checkout 28889c6ee5ebce4e68749558cb7ca48228236a35
cd $FOLDER_SP3
git init
git remote add origin https://github.com/sofa-framework/SofaPython3.git
git pull origin master
git checkout 8d8f5103d8c1bc2f4ffa918f1682d084248ecf56

cmake -Wno-dev \
-S $FOLDER_SRC -B $FOLDER_TARGET \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
-DCMAKE_BUILD_TYPE=Release \
-DSOFA_FETCH_SOFAPYTHON3=OFF \
-DPLUGIN_SOFAPYTHON3=ON \
-DPython_EXECUTABLE=$PYTHON_EXE \
-DPython_ROOT_DIR=$PYTHON_ROOT_DIR \
-DSP3_LINK_TO_USER_SITE=ON \
-DSP3_PYTHON_PACKAGES_LINK_DIRECTORY=$PYTHON_PKG_PATH \
-DPLUGIN_SOFACARVING=ON \
-DSP3_BUILD_TEST=OFF \
-DSOFA_BUILD_TESTS=OFF \
-DSOFA_BUILD_GUI_QT=ON \
-DSOFA_BUILD_SOFACOMPONENT_USERINTERFACE_CONFIGURATIONSETTING=ON \

cmake --build $FOLDER_TARGET -j 32 --target install


# 5. set sofa location variables

echo "" | tee -a $FOLDER_TARGET/install/lib/plugin_list.conf.default
echo "$FOLDER_TARGET/install/plugins/SofaPython3/lib/libSofaPython3.so 1.0" | tee -a $FOLDER_TARGET/install/lib/plugin_list.conf.default

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHON_ROOT_DIR/lib


SOFA_ROOT=~/NewVolume/Jackson/sofa/build/install/
SOFAPYTHON3_ROOT=~/NewVolume/Jackson/sofa/build/install/plugins/SofaPython3

# 6. go into sofa_zoo and run ppo.py files for rl trainings in sofa_zoo/sofa_zoo/envs/
python3 ~/NewVolume/Jackson/lap_gym/sofa_zoo/sofa_zoo/envs/deflect_spheres/ppo.py

# 7. Next steps
# try and implement SOFACUDA to increase simulation speed by offloading some tasks to gpu
# plugins = ["SofaCUDA",....]
# Replace standard Vec3d template with CUDA template
node.addObject('MechanicalObject', template="CudaVec3f")

# For collision models
node.addObject('TriangleCollisionModel', template="CudaVec3f")

rm -rf $FOLDER_TARGET
mkdir -p $FOLDER_TARGET

cmake -Wno-dev \
-S $FOLDER_SRC -B $FOLDER_TARGET \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
-DCMAKE_BUILD_TYPE=Release \
-DSOFA_FETCH_SOFAPYTHON3=OFF \
-DPLUGIN_SOFAPYTHON3=ON \
-DPLUGIN_FREEMOTIONANIMATIONLOOP=ON \
-DPython_EXECUTABLE=$PYTHON_EXE \
-DPython_ROOT_DIR=$PYTHON_ROOT_DIR \
-DSP3_LINK_TO_USER_SITE=ON \
-DSP3_PYTHON_PACKAGES_LINK_DIRECTORY=$PYTHON_PKG_PATH \
-DPLUGIN_SOFACARVING=ON \
-DSP3_BUILD_TEST=OFF \
-DSOFA_BUILD_TESTS=OFF \
-DSOFA_BUILD_GUI_QT=ON \
-DSOFA_BUILD_SOFACOMPONENT_USERINTERFACE_CONFIGURATIONSETTING=ON \

[INFO]    [PluginManager] Loaded plugin: /home/btl/NewVolume/Jackson/sofa/build/install/lib/libSofa.Component.Constraint.Projective.so
Process ForkServerProcess-1:
Traceback (most recent call last):
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/subproc_vec_env.py", line 46, in _worker
    observation, reset_info = env.reset(seed=data[0], **maybe_options)
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/gymnasium/wrappers/time_limit.py", line 75, in reset
    return self.env.reset(**kwargs)
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_env/sofa_env/scenes/deflect_spheres/deflect_spheres_env.py", line 551, in reset
    super().reset(seed)
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_env/sofa_env/base.py", line 207, in reset
    self._init_sim()
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_env/sofa_env/scenes/deflect_spheres/deflect_spheres_env.py", line 296, in _init_sim
    super()._init_sim()
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_env/sofa_env/base.py", line 286, in _init_sim
    self.scene_creation_result = getattr(self._scene_description_module, "createScene")(self._sofa_root_node, **self.create_scene_kwargs)
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_env/sofa_env/scenes/deflect_spheres/scene_description.py", line 100, in createScene
    add_scene_header(
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_env/sofa_env/sofa_templates/scene_header.py", line 169, in add_scene_header
    root_node.addObject(animation_loop.value)
ValueError: Object type FreeMotionAnimationLoop<> was not created  
The object 'FreeMotionAnimationLoop' is not in the factory.  
This component has been MOVED from SofaConstraint to Sofa.Component.AnimationLoop since SOFA v22.06.
To continue using this component you may need to update your scene by adding
<RequiredPlugin name='Sofa.Component.AnimationLoop'/>  

[ERROR]   [PythonScript] EOFError
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_zoo/sofa_zoo/sofa_zoo/envs/deflect_spheres/ppo.py", line 104, in <module>
    model.learn(
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/ppo/ppo.py", line 315, in learn
    return super().learn(
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/on_policy_algorithm.py", line 264, in learn
    total_timesteps, callback = self._setup_learn(
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/base_class.py", line 423, in _setup_learn
    self._last_obs = self.env.reset()  # type: ignore[assignment]
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/vec_transpose.py", line 113, in reset
    observations = self.venv.reset()
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/vec_frame_stack.py", line 41, in reset
    observation = self.venv.reset()
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/vec_normalize.py", line 295, in reset
    obs = self.venv.reset()
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/vec_video_recorder.py", line 67, in reset
    obs = self.venv.reset()
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/vec_monitor.py", line 70, in reset
    obs = self.venv.reset()
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/subproc_vec_env.py", line 137, in reset
    results = [remote.recv() for remote in self.remotes]
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/subproc_vec_env.py", line 137, in <listcomp>
    results = [remote.recv() for remote in self.remotes]
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/multiprocessing/connection.py", line 383, in _recv
    raise EOFError

runSofa -l | grep FreeMotionAnimationLoop

export LD_LIBRARY_PATH=/home/btl/NewVolume/Jackson/sofa/build/install/lib:$LD_LIBRARY_PATH

export SOFA_PLUGIN_PATH=/home/btl/NewVolume/Jackson/sofa/build/install/lib

import Sofa
Sofa.Core.PluginManager.getInstance().getListOfPlugins()


cmake -Wno-dev -S $FOLDER_SRC -B $FOLDER_TARGET -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DSOFA_FETCH_SOFAPYTHON3=OFF -DPLUGIN_SOFAPYTHON3=ON -DPython_EXECUTABLE=$PYTHON_EXE -DPython_ROOT_DIR=$PYTHON_ROOT_DIR -DSP3_LINK_TO_USER_SITE=ON -DSP3_PYTHON_PACKAGES_LINK_DIRECTORY=$PYTHON_PKG_PATH -DPLUGIN_SOFACARVING=ON -DSP3_BUILD_TEST=OFF -DSOFA_BUILD_TESTS=OFF -DSOFA_BUILD_GUI_QT=ON \
-DPLUGIN_SOFACOMPONENT_ANIMATIONLOOP_FREEMOTION=ON \
-DSOFA_BUILD_SOFACOMPONENT_USERINTERFACE_CONFIGURATIONSETTING=ON \

ls $FOLDER_TARGET/lib/libSofa.Component.AnimationLoop.*
FOLDER_SRC=~/NewVolume/Jackson/sofa/src
FOLDER_TARGET=~/NewVolume/Jackson/sofa/build
FOLDER_SP3=$FOLDER_SRC/applications/plugins/SofaPython3

PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PYTHON_EXE=$(which python3)
PYTHON_ROOT_DIR=$CONDA_PREFIX

SOFA_ROOT=~/NewVolume/Jackson/lap_gym/sofa_env/SOFA
SOFAPYTHON3_ROOT=$SOFA_ROOT/plugins/SofaPython3

 python3 "/media/btl/New Volume/Jackson/lap_gym/sofa_zoo/sofa_zoo/envs/deflect_spheres/ppo.py" 
  python3 "/media/btl/New Volume/Jackson/lap_gym/sofa_zoo/sofa_zoo/envs/tissue_dissection/ppo.py"

[ERROR]   [PythonScript] AssertionError: The render_mode must be 'rgb_array', not None
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_zoo/sofa_zoo/envs/tissue_dissection/ppo.py", line 88, in <module>
    model, callback = configure_learning_pipeline(
  File "/media/btl/New Volume/Jackson/lap_gym/sofa_zoo/sofa_zoo/common/sb3_setup.py", line 118, in configure_learning_pipeline
    env = VecVideoRecorder(
  File "/home/btl/NewVolume/Jackson/conda_sofa/lib/python3.10/site-packages/stable_baselines3/common/vec_env/vec_video_recorder.py", line 52, in __init__
    assert self.env.render_mode == "rgb_array", f"The render_mode must be 'rgb_array', not {self.env.render_mode}"

look at line 118 in setup.py for fix: Only wrap in video recorder if render mode is correct


bSofa.Component.ODESolver.Backward.so
[INFO]    [PluginManager] Loaded plugin: /home/btl/NewVolume/Jackson/lap_gym/sofa_env/SOFA/lib/libSofa.Component.LinearSolver.Iterative.so
[INFO]    [PluginManager] Loaded plugin: /home/btl/NewVolume/Jackson/lap_gym/sofa_env/SOFA/lib/libSofa.GL.Component.Rendering3D.so
[INFO]    [PluginManager] Loaded plugin: /home/btl/NewVolume/Jackson/lap_gym/sofa_env/SOFA/lib/libSofa.Component.Topology.Container.Grid.so
[WARNING] [InteractiveCamera(camera)] Too many missing parameters ; taking default ...
[WARNING] [InteractiveCamera(camera)] Too many missing parameters ; taking default ...
[WARNING] [CGLinearSolver(CGLinearSolver)] Required data "iterations" has not been set. Falling back to default value: 25
[WARNING] [CGLinearSolver(CGLinearSolver)] Required data "tolerance" has not been set. Falling back to default value: 1e-05
[WARNING] [CGLinearSolver(CGLinearSolver)] Required data "threshold" has not been set. Falling back to default value: 1e-05
[WARNING] [UniformVelocityDampingForceField(UniformVelocityDampingForceField)] buildStiffnessMatrix not implemented: for compatibility reason, the deprecated API (addKToMatrix) will be used. This compatibility will disapear in the future, and will cause issues in simulations. Please update the code of UniformVelocityDampingForceField to ensure right behavior: the function addKToMatrix has been replaced by buildStiffnessMatrix

