# Synbols benchmarks


## Requirements
```
$ pip install -r requirements.txt
```

## Add benchmark
Individual benchmarks should be placed into `models`.


## Current benchmarks

### Classification
**Resnet-18 on fonts**

`python3 trainval.py --savedir_base <experiments directory> --exp_group_list font`

### Detection

### Few-shot
**Resnet-18 prototypical on char**

`python3 trainval.py --savedir_base <experiments directory> --exp_group_list fewshot_char`

### Active learning


### Getting Started

To setup a machine learning project for large-scale experimentation, we can follow these 4 steps.

1. [Write the codebase;](#1-writing-the-codebase)
2. [define the hyperparameters;](#2-defining-the-hyperparameters)
3. [run the experiments; and](#3-running-the-experiments)
4. [visualize the results.](#4-visualizing-the-results)

### Examples

The following folders contain example projects built on this framework.

- [Classification](https://github.com/ElementAI/haven/tree/master/examples/classification)
- [Active Learning](https://github.com/ElementAI/haven/tree/master/examples/active_learning)
- [Generative Adversarial Networks](https://github.com/ElementAI/haven/tree/master/examples/gans)
- [Object Counting](https://github.com/ElementAI/haven/tree/master/examples/object_counting)



#### 1. Writing the Codebase

Create a file `trainval.py` with the template below: 

```python
import os
import argparse

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
import pandas as pd


def trainval(exp_dict, savedir_base, reset=False):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print(exp_dict)
    print("Experiment saved in %s" % savedir)

    # Dataset
    # -----------

    # train and val loader
    train_loader = ...
    val_loader = ...
   
    # Model
    # -----------
    model = ...

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.set_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ------------
    print("Starting experiment at epoch %d" % (s_epoch))

    for e in range(s_epoch, exp_dict['max_epoch']):
        score_dict = {}

        # Train the model
        score_dict.update(model.train_on_loader(train_loader))

        # Validate the model
        score_dict.update(model.val_on_loader(val_loader, savedir=os.path.join(savedir_base, exp_dict['dataset']['name'])))
        score_dict["epoch"] = e

        # Visualize the model
        model.vis_on_loader(vis_loader, savedir=savedir+"/images/")

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail())
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

    print('experiment completed')


```



#### 2. Defining the Hyperparameters

Add to `trainval.py` the following dictionary which defines a set of hyperparameters for Mnist. This dictionary defines a `mnist` experiment group.

```python
from haven import haven_utils as hu

# Define exp groups for parameter search
EXP_GROUPS = {'mnist':
                hu.cartesian_exp_group({
                    'lr':[1e-3, 1e-4],
                    'batch_size':[32, 64]})
                }
```

#### 3. Running the Experiments

To run `trainval.py` with the `mnist` experiment group, follow the two steps below.

##### 3.1 Create the 'Main' Script

Add the following script to `trainval.py`. This script allows the user to use the command line to select between experiment groups in order to run them.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # run experiments
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                datadir_base=args.datadir_base,
                reset=args.reset)
```



##### 3.2 Run trainval.py in Command Line

Run the following command in order to launch the mnist experiments and save them under the folder `../results/`.

```
python trainval.py -e mnist -sb ../results -r 1
```

##### 3.3 Using a job manager

You can run all the experiments in parallel using a job scheduler such as the orkestrator. The job scheduler can be used with the following script.

```python
# launch jobs
elif args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hjb
        run_command = ('python trainval.py -ei <exp_id> -sb %s -d %s -nw 1' %  (args.savedir_base, args.datadir_base))
        job_config = {'volume': <volume>,
                    'image': <docker image>,
                    'bid': '1',
                    'restartable': '1',
                    'gpu': '4',
                    'mem': '30',
                    'cpu': '2'}
        workdir = os.path.dirname(os.path.realpath(__file__))

        hjb.run_exp_list_jobs(exp_list, 
                            savedir_base=args.savedir_base, 
                            workdir=workdir,
                            run_command=run_command,
                            job_config=job_config)
```

#### 4. Visualizing the Results
![](examples/4_results.png)

The following two steps will setup the visualization environment.

##### 1. Install Jupyter

Run the following in command line to install a Jupyter server
```bash
mkdir .jupyter_server
cd .jupyter_server

virtualenv -p python3 .
source bin/activate

pip install jupyter notebook
pip install ipywidgets
pip install --upgrade git+https://github.com/ElementAI/haven

jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter notebook --ip 0.0.0.0 --port 9123 \
      --notebook-dir="/home/$USER" --NotebookApp.token="password"
```

##### 2. Create Jupyter

Shown in example.ipynb, run the following script in a Jupyter cell. The script will launch a dashboard from the specified variables


```python
# Specify variables
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu

# please define the path to the experiments
savedir_base = <path_to_saved_experiments>
exp_list = None

# exp_config_name = <exp_config_name>
# exp_list = hu.load_py(exp_config_name).EXP_GROUPS['mnist']

# get specific experiments, for example, {'model':'resnet34'}
filterby_list = None

# group the experiments based on a hyperparameter, for example, ['dataset']
groupby_list = None
verbose = 0

# plot vars
y_metrics='train_loss'
x_metric='epoch'
log_metric_list = ['train_loss']
map_exp_list = []
title_list=['dataset']
legend_list=['model']

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      filterby_list=filterby_list,
                      verbose=verbose
                     )

# launch dashboard
hj.get_dashboard(rm, vars(), wide_display=True)
```

To install Haven from a jupyter cell, run the following script in a cell,

```bash
import sys
!{sys.executable} -m pip install --upgrade  --no-dependencies 'git+https://github.com/ElementAI/haven' --user
```
<!-- /home/issam/Research_Ground/haven/ -->

### Extras

- Create a list of hyperparameters.
- Save a score dictionary at each epoch.
- Launch a score dictionary at each epoch.
- Create and Launch Jupyter.
- View experiment configurations.
- Debug a single experiment
- View scores as a table.
- View scores as a plot.
- View scores as a barchart.




### Contributing

We love contributions!
