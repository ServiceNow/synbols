from haven import haven_utils as hu
import numpy as np

# -------------------------------- SETTINGS -----------------------------------%

fewshot_boilerplate = {
    'benchmark':'fewshot',
    'episodic': True,
    'max_epoch':5000,
    'patience':[10, 25, 50],
    'dataset': hu.cartesian_exp_group({
        'path':'/mnt/datasets/public/research/synbols/balanced-font-chars_n=1000000_2020-May-28.h5py',
        # 'path':'/mnt/datasets/public/research/synbols/default_n=100000_2020-May-20.h5py',
        'name': 'fewshot_synbols',
        "width": 32,
        "height": 32,
        "channels": 3,
        'task': [
                {'train':'char', 'val':'char', 'test':'char', 'ood':'font'},
                {'train':'font', 'val':'font', 'test':'font', 'ood':'char'}
        ],
        'mask': ['stratified_char', 'stratified_font'], 
        # 'mask': ["random", "compositional_char_font", "stratified_char", "stratified_font", "stratified_char_font"]
        'trim_size': [None],
        "z_dim_multiplier": 2*2,
        ## start 5-way 5-shot 15-query
        'nclasses_train': 5, 
        'nclasses_val': 5,
        'nclasses_test': 5,
        'support_size_train': 5,
        'support_size_val': 5,
        'support_size_test': 5,
        'query_size_train': 15,
        'query_size_val': 15,
        'query_size_test':15,
        ## end 5-way 5-shot 5-query
        'train_iters': 5,
        'val_iters': 5,
        'test_iters': 5,
        'ood_iters': 5,
        # 'train_iters': 500,
        # 'val_iters': 500,
        # 'test_iters': 500,
        # 'ood_iters': 500,
    })
}


# -------------------------------- BACKBONES ----------------------------------%

conv4_backbone = {
    'lr':[0.005, 0.002, 0.001, 0.0005, 0.002, 0.0001],
    'batch_size':[1],
    'optimizer':'adam',
    'backbone': hu.cartesian_exp_group({ 
        'name':'conv4',
        'hidden_size': [256, 64],
    })
}

resnet18_backbone = {
    'backbone': hu.cartesian_exp_group({ 
        'name':'resnet18',
        'hidden_size': [64],
        'imagenet_pretraining': [False, True],
    })
}

# -------------------------------- METHODS ------------------------------------%

ProtoNet = {
    'model': "ProtoNet",
}


MAML = {
    'model': "MAML",
    'inner_lr':[0.5, 0.1, 0.01, 0.001, 0.0001],
    'n_inner_iter':[1, 2, 4, 8, 16],
}

RelationNet = {
    'model': "RelationNet",
}

# ------------------------------ EXPERIMENTS ----------------------------------%

n_trials = 10
n_runs = 3

def random_search(hp_lists, n_trials, n_runs=1):
    for i in range(len(hp_lists)):
        if i ==0:
            out = np.random.choice(hu.cartesian_exp_group(
                    hp_lists[i]), n_trials, replace=True).tolist()
        if i == len(hp_lists) - 1:
            out = hu.ignore_duplicates(out)
            print('remove {} duplicates'.format(n_trials-len(out)))
            break
        to_add = np.random.choice(hu.cartesian_exp_group(
                    hp_lists[i+1]), n_trials, replace=True).tolist()
        out = [dict(out[i], **to_add[i]) for i in range(n_trials)]
    ## running multiple 
    if n_runs == 1:
        return out
    else:
        out_n_runs = []
        for i in range(n_runs):
            out_n_runs += [dict(out[j], **{'seed':i}) for j in range(len(out))]
        return out_n_runs

EXP_GROUPS = {}


## Hparam search on few-shot char AND font classification
## https://app.wandb.ai/optimass/synbols_fewshot_hps0/
## https://github.com/ElementAI/synbols/commit/dd34dea5f789a922b48cb98946b506ec31db6ba9
EXP_GROUPS['fewshot_ProtoNet'] = random_search(
    [fewshot_boilerplate, ProtoNet, conv4_backbone], n_trials, n_runs)
 
EXP_GROUPS['fewshot_MAML'] = random_search(
    [fewshot_boilerplate, MAML, conv4_backbone], n_trials, n_runs)

EXP_GROUPS['fewshot_RelationNet'] = random_search(
    [fewshot_boilerplate, RelationNet, conv4_backbone], n_trials, n_runs)

## Safety check
# EXP_GROUPS['fewshot_ProtoNet_resnet'] = random_search(
#     [fewshot_boilerplate, ProtoNet, resnet18_backbone], n_trials)

## Hparam search on few-shot char classification
## https://app.wandb.ai/optimass/synbols_char_hps0/
## https://github.com/ElementAI/synbols/commit/dd34dea5f789a922b48cb98946b506ec31db6ba9
# EXP_GROUPS['fewshot_char_ProtoNet'] = random_search(
#     [fewshot_char_boilerplate, ProtoNet, conv4_backbone], n_trials)
# EXP_GROUPS['fewshot_char_MAML'] = random_search(
#     [fewshot_char_boilerplate, MAML, conv4_backbone], n_trials)
# EXP_GROUPS['fewshot_font_ProtoNet'] = random_search(
#     [fewshot_font_boilerplate, ProtoNet, conv4_backbone], n_trials)
# EXP_GROUPS['fewshot_font_MAML'] = random_search(
#     [fewshot_font_boilerplate, MAML, conv4_backbone], n_trials)

# ------------------------------ MINIIMAGENET ----------------------------------%


fewshot_mini_boilerplate = {
    'benchmark':'fewshot',
    'episodic': True,
    'max_epoch':[1000, 1500],
    'dataset': hu.cartesian_exp_group({
        'path':'/mnt/datasets/public/mini-imagenet/',
        'name': 'miniimagenet',
        "width": 84,
        "height": 84,
        "channels": 3,
        "z_dim_multiplier": 5*5,
        ## start 5-way 5-shot 15-query
        'nclasses_train': 5, 
        'nclasses_val': 5,
        'nclasses_test': 5,
        'support_size_train': 5,
        'support_size_val': 5,
        'support_size_test': 5,
        'query_size_train': 15,
        'query_size_val': 15,
        'query_size_test': 15,
        ## end 5-way 5-shot 15-query
        'train_iters': 500,
        'val_iters': 500,
        'test_iters': 500,
    })
}


## Hparam search on miniimagenet
## https://github.com/ElementAI/synbols/commit/dd34dea5f789a922b48cb98946b506ec31db6ba9
## https://app.wandb.ai/optimass/miniimagenet_hps1/workspace?workspace=user-optimass
EXP_GROUPS['fewshot_mini_ProtoNet'] = random_search(
    [fewshot_mini_boilerplate, ProtoNet, conv4_backbone], n_trials)
 
EXP_GROUPS['fewshot_mini_MAML'] = random_search(
    [fewshot_mini_boilerplate, MAML, conv4_backbone], n_trials)

EXP_GROUPS['fewshot_mini_RelationNet'] = random_search(
    [fewshot_mini_boilerplate, RelationNet, conv4_backbone], n_trials)