from haven import haven_utils as hu
import numpy as np

# -------------------------------- SETTINGS -----------------------------------%

fewshot_char_boilerplate = {
    'benchmark':'fewshot',
    'episodic': True,
    'hidden_size': [64, 256],
    'max_epoch':500,
    'dataset': hu.cartesian_exp_group({
        'path':'/mnt/datasets/public/research/synbols/all_chars_n=1000000_2020-Apr-30.h5py',
        # 'path':'/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py',
        'name': 'fewshot_synbols',
        "width": 32,
        "height": 32,
        "channels": 3,
        # 'task': {'train':'char', 'val':'char'},
        'task': 'char',
        # 'mask': 'random', 
        'mask': ["compositional_char_font", "stratified_char", "stratified_char_font", "random"],
        # 'mask': ["random", "compositional_char_font", "stratified_char", "stratified_font", "stratified_char_font"]
        'trim_size': [100000, None],
        ## start 5-way 5-shot 5-query
        'nclasses_train': 5, 
        'nclasses_val': 5,
        'nclasses_test': 5,
        'support_size_train': 5,
        'support_size_val': 5,
        'support_size_test': 5,
        'query_size_train': 5,
        'query_size_val': 5,
        'query_size_test': 5,
        ## end 5-way 5-shot 5-query
        'train_iters': 500,
        'val_iters': 500,
        'test_iters': 500,
    })
}

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

# -------------------------------- BACKBONES ----------------------------------%

conv4_backbone = {
    'backbone': hu.cartesian_exp_group({ 
        'name':'conv4',
        'hidden_size': [64,],
    })
}

resnet18_backbone = {
    'backbone': hu.cartesian_exp_group({ 
        'name':'resnet18',
        'hidden_size': [64, 128],
        'imagenet_pretraining': [False, True],
    })
}

# -------------------------------- METHODS ------------------------------------%

ProtoNet = {
    'model': "ProtoNet",
    # 'lr':[0.01],
    'lr':[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    'batch_size':[1],
}


MAML = {
'model': "MAML",
'outer_lr':[0.5, 0.1, 0.01, 0.001, 0.0001],
'inner_lr':[0.5, 0.1, 0.01, 0.001, 0.0001],
'n_inner_iter':[1, 2, 4, 8, 16],
'batch_size':[1],
}

RelationNet = {
'lr':[0.1, 0.05, 0.01, 0.001, 0.0001],
'batch_size':[1],
'model': "RelationNet",
}

# ------------------------------ EXPERIMENTS ----------------------------------%

n_trials = 30

# I have to do it this way unless it's too long
def random_search(hp_lists, n_trials):
    for i in range(len(hp_lists)):
        if i ==0:
            out = np.random.choice(hu.cartesian_exp_group(
                    hp_lists[i]), n_trials, replace=True).tolist()
        if i == len(hp_lists) - 1:
            out = hu.ignore_duplicates(out)
            print('remove {} duplicates'.format(n_trials-len(out)))
            return out
        to_add = np.random.choice(hu.cartesian_exp_group(
                    hp_lists[i+1]), n_trials, replace=True).tolist()
        out = [dict(out[i], **to_add[i]) for i in range(n_trials)]




EXP_GROUPS = {}

EXP_GROUPS['fewshot_mini_ProtoNet'] = random_search(
    [fewshot_mini_boilerplate, ProtoNet, conv4_backbone], n_trials)
 
EXP_GROUPS['fewshot_mini_MAML'] = random_search(
    [fewshot_mini_boilerplate, MAML, conv4_backbone], n_trials)
    


# EXP_GROUPS['fewshot_char_ProtoNet'] = hu.cartesian_exp_group(
# dict(fewshot_char_boilerplate, **fewshot_ProtoNet)
# )

# EXP_GROUPS['fewshot_char_MAML'] = np.random.choice(hu.cartesian_exp_group(
# dict(fewshot_char_boilerplate, **fewshot_MAML)),
# 6*16, replace=False).tolist() # random search

# EXP_GROUPS['fewshot_char_RelationNet'] = hu.cartesian_exp_group(
# dict(fewshot_char_boilerplate, **fewshot_RelationNet)
# )

# EXP_GROUPS['fewshot_mini_ProtoNet'] = hu.cartesian_exp_group(
# dict(fewshot_mini_boilerplate, **fewshot_ProtoNet)
# )

# EXP_GROUPS['fewshot_mini_MAML'] = hu.cartesian_exp_group(
# dict(fewshot_mini_boilerplate, **fewshot_MAML)
# )
