

## quick start

1. (optional) install mujoco: https://github.com/openai/mujoco-py/#install-mujoco
	- on ubuntu you also need to install some requirements `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
	- also the most recent version of mujoco is not compatbile with the most recent version of openaigym. So we use mjpro150 instead of mujoco200. [Download](https://www.roboti.us/download/mjpro150_linux.zip)
2. setup virtual environment:
```bash
git clone git@github.com:neuroevolution-ai/CTRNN_Simulation_Results.git
git clone git@github.com:neuroevolution-ai/NeuroEvolution-CTRNN.git
cd NeuroEvolution-CTRNN

sudo apt install cmake python3 python3-dev python3-pip swig python3-tk
sudo pip3 install virtualenv --system
virtualenv ~/.venv/neuro --python=python3
. ~/.venv/neuro/bin/activate
pip install --upgrade -r requirements.txt

# if mujoco is needed, also run this
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chef/.mujoco/mjpro150/bin
pip install mujoco-py 'gym[mujoco]'
```

3. (optional) change configuration `nano configurations/default.json`
3. run training
	- this will take some time
	- `tmux`
    - `. ~/.venv/neuro/bin/activate`
    - `python -m scoop neuro_evolution_ctrnn/train.py --configuration configurations/default.json`
4. show results:
	- `python neuro_evolution_ctrnn/visualize.py`
5. publish results
    - results are stored in a git submodule, so we need the change cwd before commiting
    - `cd ../CTRNN_Simulation_Results`
    - `git add . && git commit -m "new simulation results" && git pull --rebase && git push`
    

## development

checks: 

```
. $HOME/.venv/neuro/bin/activate
python -m mypy .
PYTHONPATH=neuro_evolution_ctrnn pytest -q --disable-pytest-warnings tests

```

### tricks

### generate statistics 

```
python neuro_evolution_ctrnn/batch_generate_plot.py
cd ../CTRNN_Simulation_Results
python resuts_to_csv.py
```

now you can open `output.csv` in libreoffice-calc. 
You can format the hyperlinks to the plots such that they become clickable links,
which open the plots as images: https://ask.libreoffice.org/en/question/69088/calc-importing-csv-full-of-urls/

Note: make sure the numbers are imported correctly. A comma can be a 
decimal separator, a thousands separator and a column separator. 
A dot can be a decimal-separator and a thousands separator. Sometimes
Libreoffice uses the dot as thousands separator in some columns and as
a decimal separator in others during the same import of an csv.

#### scrolling in tmux: 

``` 
echo "set -g mouse on" >> ~/.tmux.conf
tmux kill-server && tmux
```

#### paramter für scripte

jedes script in diesem repo, 
dass direkt ausgeführt werden kann, hat eine hilfe-funktion, die 
man mit `--help` aufrufen kann

#### Results in IDE anzeigen

Wenn du willst, dass der results-ordner im IDE 
angezeigt wird, kannst du einfach einen softlink darauf ins repo legen: 
`cd NeuroEvolution-CTRNN && ln -s ../CTRNN_Simulation_Results results`

## troubleshooting


#### visualisierung fails with "ERROR: GLEW initalization error: Missing GL version"

solution: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so)`

https://github.com/openai/mujoco-py/issues/268#issuecomment-595177264


