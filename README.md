# Installation

1. Optional: [Install MuJoCo](https://github.com/openai/mujoco-py/#install-mujoco)
	- If you are using Ubuntu you also need to install some requirements:
	
         `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
	- Keep in mind that the the most recent version of MuJoCo is not compatible with the most recent version of the
	OpenAI Gym. Therefore we use [mjpro150](https://www.roboti.us/download/mjpro150_linux.zip) instead of mujoco200. 
2. Setup the virtual environment
    - Clone the repository for the training results as well as the repository for the code
    
        ```bash
        git clone git@github.com:neuroevolution-ai/CTRNN_Simulation_Results.git
        git clone git@github.com:neuroevolution-ai/NeuroEvolution-CTRNN.git
        cd NeuroEvolution-CTRNN
      ```
    - Install system dependencies. If you are running a Linux with a different package manager google the names of the
    packages for your system.
    
        ```bash
        sudo apt install cmake python3 python3-dev python3-pip swig python3-tk tmux
      ```
    
    - Create the virtual environment, activate it and install the required Python packages

        ```bash
        sudo pip3 install virtualenv --system
        virtualenv $HOME/.venv/neuro --python=python3
        . $HOME/.venv/neuro/bin/activate
        pip install --upgrade -r requirements.txt
      ```

    - Optional: If you are using MuJoCo the following steps are required
    
        ```bash
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
        pip install mujoco-py 'gym[mujoco]'
        ``` 

# Quick Start

1. Optional: Change the configuration `nano configurations/default.json`
2. Start the Training
	- Keep in mind that this will take some time, depending on your hardware and the configuration
	- Run the following commands
	
        ```bash
        tmux
        . $HOME/.venv/neuro/bin/activate`
        python neuro_evolution_ctrnn/train.py --configuration configurations/default.json
        ```

3. Visualize the results
    
    `python neuro_evolution_ctrnn/visualize.py --neuron-vis --render` 

4. Optional: Publish the results
    - All results are stored in a git submodule which is located in a different folder
        
        `cd ../CTRNN_Simulation_Results`
    
    - After changing directories, pull recent changes and add the new results
    
        `git pull --rebase && git add .`
        
    - Now commit the results with a corresponding commit message and push them
    
        `git commit -m "New simulation results" && git push`
    

# Development

## Tests and checks

```
. $HOME/.venv/neuro/bin/activate
python -m mypy .
PYTHONPATH=neuro_evolution_ctrnn pytest -q --disable-pytest-warnings tests
```

## Tips and tricks

### PyCharm

  * For running `mypy` simply add a new Python configuration, choose Module name instead of Script path and use `mypy`
   as the module. Then add the absolute path to the directory containing the code as a parameter, like so:
    
    `$HOME/PycharmProjects/NeuroEvolution-CTRNN_new/neuro_evolution_ctrnn`
    
    (Assuming that you cloned the repository to the `PycharmProjects` folder.)
   
   Mypy can now be started by running the configuration.
   
  * Another way can be to use PyCharm plugins
      * https://plugins.jetbrains.com/plugin/11086-mypy
        * Does not work for us, see: https://github.com/leinardi/mypy-pycharm/issues/60
      * https://plugins.jetbrains.com/plugin/13348-mypy-official-
        * Config parameter `Path suffix` has to be set to the `bin` folder of the virtual environment
        * Run command: `dmypy run . -- --follow-imports skip`

### Generate statistics 

```
python neuro_evolution_ctrnn/batch_generate_plot.py
cd ../CTRNN_Simulation_Results
python resuts_to_csv.py
```

`output.csv` can now be opened in LibreOffice Calc. The hyperlinks in the plot column do not get formatted when loading. 
LibreOffice will recognize the hyperlinks when you enter the cell, add a space to the end and leave the cell again. 
When you then click on the hyperlinks, a window with the plot should open.


_Note_: Make sure that the numbers are imported correctly. A comma can be a 
decimal separator, a thousands separator (e.g., 100,000,000) or a column separator. 
A dot can be a decimal separator and a thousands separator. Sometimes
LibreOffice uses the dot as a thousands separator in some columns and as
a decimal separator in other columns during the import of the __same__ CSV.

### Scrolling in tmux: 

``` 
echo "set -g mouse on" >> $HOME/.tmux.conf
tmux kill-server && tmux
```

### Parameters for the scripts

Every script in this repository that can be executed directly has a help function which can be executed by
providing the `--help` parameter.

### JSON formatting

All JSON files are formatted with the following style. Applying different formatting, or none at all, will result in 
a lot of clutter in the git commits.

For PyCharm one can change the formatting of JSON by going to:

File --> Settings --> Editor --> Code Style --> JSON --> Wrapping and Braces

|   |    |
|---|----|
| Hardwrap at| 120 |
| Wrap on typing| yes
| Visual guides| 80, 120
| _Keep when reformatting_|
| Line breaks| ☑ 
| Trailing comma| ☐ 
| Ensure right margin is not exceeded| ☑
| Arrays| Wrap if long
| Objects| Wrap if long
| Align| Do not align

## visualizer
If it is not possible to install pygame via pip, try with apt-manager:
``sudo apt-get install python-pygame``

### Interaktion
You can individualize the visualizer via keyboard and mouse.


## Troubleshooting


### Visualization fails with "ERROR: GLEW initialization error: Missing GL version"

Solution: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so`

Source: https://github.com/openai/mujoco-py/issues/268#issuecomment-595177264


