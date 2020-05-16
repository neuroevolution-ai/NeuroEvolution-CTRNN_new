

## quick start

1. (optional) install mujoco: https://github.com/openai/mujoco-py/#install-mujoco
	- on ubuntu you also need to install some requirements `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
	- also the most recent version of mujoco is not compatbile with the most recent version of openaigym. So we use mjpro150 instead of mujoco200. [Download](https://www.roboti.us/download/mjpro150_linux.zip)
2. setup virtual environment:
```bash
git clone git@github.com:neuroevolution-ai/NeuroEvolution-CTRNN.git
cd NeuroEvolution-CTRNN
sudo apt install cmake python3 python3-dev python3-pip swig python3-tk
sudo pip3 install virtualenv --system
virtualenv ~/.venv/neuro --python=python3
. ~/.venv/neuro/bin/activate
pip install scoop pybullet 'gym[atair]' 'gym[box2d]' torch deap matplotlib

# if mujoco is needed, also run this
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chef/.mujoco/mjpro150/bin
pip install mujoco-py 'gym[mujoco]'
```

3. (optional) change configuration `nano Configuration.json`
3. run training
	- this will take some time
	- `LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin python -m scoop CTRNN_ReinforcementLearning_CMA-ES.py`
4. show results:
	- `LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin python CTRNN_Visualisierung.py`

## troubleshooting

#### visualisierung fails with "ERROR: GLEW initalization error: Missing GL version"

solution: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so)`

https://github.com/openai/mujoco-py/issues/268#issuecomment-595177264


