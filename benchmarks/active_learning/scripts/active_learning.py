from haven import haven_utils as hu

# Define exp groups for parameter search
model_cfg = {
    'lr': [0.001],
    'batch_size': [32],
    'model': "active_learning",
    'seed': [1337, 1338, 1339, 1340],
    'mu': 1e-3,
    'reg_factor': 1e-4,
    'backbone': "vgg16",
    'num_classes': 52,
    'query_size': [100],
    'learning_epoch': 10,
    'heuristic': ['bald', 'random', 'entropy'],
    'iterations': [20],
    'max_epoch': 2000,
    'imagenet_pretraining': [True],
}

EXP_GROUPS = {
    'active_char_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
                'path': '/mnt/datasets/public/research/synbols/active_learning/missing-symbol_n=100000_2020-Apr-30.h5py',
                'name': 'active_learning',
                'task': 'char',
                'initial_pool': 2000,
                'seed': 1337})),
    'active_char_missing_mixup':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='mixup_active_learning',
                                                         mixup_layer_name='classifier.2',
                                                         mixup_alpha=[0.0, 0.7])},
                               dataset={
                                   'path': '/mnt/datasets/public/research/synbols/active_learning/missing-symbol_n=100000_2020-Apr-30.h5py',
                                   'name': 'active_learning',
                                   'task': 'char',
                                   'initial_pool': 2000,
                                   'seed': 1337
                               })),
'active_char_missing':
        hu.cartesian_exp_group(dict(model_cfg,
                               dataset={
                                   'path': '/mnt/datasets/public/research/synbols/active_learning/missing-symbol_n=100000_2020-Apr-30.h5py',
                                   'name': 'active_learning',
                                   'task': 'char',
                                   'initial_pool': 2000,
                                   'seed': 1337
                               })),
    'active_char_label_noise':
        hu.cartesian_exp_group(dict(**model_cfg, dataset={
            'path': '/mnt/datasets/public/research/synbols/default_n=100000_2020-May-20.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337,
            'p': 0.15})),
'active_char_label_noise_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
            'path': '/mnt/datasets/public/research/synbols/default_n=100000_2020-May-20.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337,
            'p': 0.15})),
    'active_char_pixel_noise':
        hu.cartesian_exp_group(dict(**model_cfg, dataset={
            'path': '/mnt/datasets/public/research/synbols/default_n=100000_2020-May-20.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337,
            'pixel_sigma': 0.7,
            'pixel_p': 0.5})),
'active_char_pixel_noise_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
            'path': '/mnt/datasets/public/research/synbols/default_n=100000_2020-May-20.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337,
            'pixel_sigma': 0.7,
            'pixel_p': 0.5})),
    'active_char_default_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
            'path': '/mnt/datasets/public/research/synbols/default_n=100000_2020-May-20.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337})),
'active_char':
        hu.cartesian_exp_group(dict(**model_cfg, dataset={
            'path': '/mnt/datasets/public/research/synbols/default_n=100000_2020-May-20.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337})),
    'active_char_large_trans':
        hu.cartesian_exp_group(dict(**model_cfg, dataset={
            'path': '/mnt/datasets/public/research/synbols/active_learning/large-translation_n=100000_2020-May-06.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337})),
    'active_char_large_trans_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
            'path': '/mnt/datasets/public/research/synbols/active_learning/large-translation_n=100000_2020-May-06.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337})),
    'active_char_partly_occluded':
        hu.cartesian_exp_group(dict(**model_cfg, dataset={
            'path': '/mnt/datasets/public/research/synbols/active_learning/partly-occluded_n=100000_2020-May-05.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337})),
'active_char_partly_occluded_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
            'path': '/mnt/datasets/public/research/synbols/active_learning/partly-occluded_n=100000_2020-May-05.h5py',
            'name': 'active_learning',
            'task': 'char',
            'initial_pool': 2000,
            'seed': 1337})),
'active_char_missing_sghmc':
        hu.cartesian_exp_group(dict(**{**model_cfg, **{'model': 'csghmc_active_learning',
                                                       'learning_epoch': 20,
                                                       'batch_size': 10}},
                               dataset={
                                   'path': '/mnt/datasets/public/research/synbols/active_learning/missing-symbol_n=100000_2020-Apr-30.h5py',
                                   'name': 'active_learning',
                                   'task': 'char',
                                   'initial_pool': 2000,
                                   'seed': 1337
                               })),

}
