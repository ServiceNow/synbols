from haven import haven_utils as hu
import numpy as np

# Define exp groups for parameter search

fewshot_char_boilerplate = {
        'benchmark':'fewshot',
        'episodic': True,
        'hidden_size': 64,
        'max_epoch':1000,
        'dataset': hu.cartesian_exp_group({
                        # 'path':'/mnt/datasets/public/research/synbols/all_chars_n=1000000_2020-Apr-30.h5py',
                        'path':'/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py',
                        'name': 'fewshot_synbols',
                        # 'task': {'train':'char', 'val':'char'},
                        'task': 'char',
                        'mask': 'stratified_char_font', #{"random", "compositional_char_font", "stratified_char", "stratified_font", "stratified_char_font"}
                        # 'mask': ["compositional_char_font", "stratified_char", "stratified_char_font"],
                        # 'mask': ["random", "compositional_char_font", "stratified_char", "stratified_font", "stratified_char_font"]
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
                        'train_iters': 50,
                        'val_iters': 50,
                        'test_iters': 50,
        })
}


fewshot_ProtoNet = {
        # 'lr':[0.5, 0.1, 0.05, 0.01, 0.005],
        'lr':[0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        'batch_size':[1],
        'model': "ProtoNet",
        'backbone': "conv4",
        'imagenet_pretraining': False,
}

fewshot_ProtoNet_resnet = {
        # 'lr':[0.5, 0.1, 0.05, 0.01, 0.005],
        'lr':[0.005, 0.001, 0.0005, 0.0001],
        'batch_size':[1],
        'model': "ProtoNet",
        'backbone': "resnet18",
        'imagenet_pretraining': [False, True],
}

fewshot_MAML = {
        'model': "MAML",
        'outer_lr':[0.5, 0.1, 0.01, 0.001, 0.0001],
        'inner_lr':[0.5, 0.1, 0.01, 0.001, 0.0001],
        'n_inner_iter':[1, 2, 4, 8, 16],
        'batch_size':[1],
        'backbone': "conv4",
        'imagenet_pretraining': False,
}

fewshot_RelationNet = {
        'lr':[0.1, 0.05, 0.01, 0.001, 0.0001],
        'batch_size':[1],
        'model': "RelationNet",
        'backbone': "conv4",
        'max_epoch': 1000,
}


EXP_GROUPS = {}

EXP_GROUPS['fewshot_char_ProtoNet'] = hu.cartesian_exp_group(
        dict(fewshot_char_boilerplate, **fewshot_ProtoNet)
)

EXP_GROUPS['fewshot_char_MAML'] = np.random.choice(hu.cartesian_exp_group(
        dict(fewshot_char_boilerplate, **fewshot_MAML)),
        60, replace=False).tolist() # random search

EXP_GROUPS['fewshot_char_RelationNet'] = hu.cartesian_exp_group(
        dict(fewshot_char_boilerplate, **fewshot_RelationNet)
)


