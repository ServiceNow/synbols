from haven import haven_utils as hu

# Define exp groups for parameter search
EXP_GROUPS = {'active_font':
    hu.cartesian_exp_group({
        'lr': [0.001],
        'batch_size': [32],
        'model': "active_learning",
        'backbone': "vgg16",
        'num_classes': 1001,
        'query_size': [100],
        'learning_epoch': 10,
        'heuristic': ['bald', 'random', 'entropy'],
        'iterations': [1, 20],
        'max_epoch': 100,
        'imagenet_pretraining': [False],
        'dataset': {'path': '/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                    'name': 'active_learning',
                    'task': 'font',
                    'initial_pool': 1000,
                    'uncertainty_config': {'is_bold': {}}}}),
    'active_font_pretrained': hu.cartesian_exp_group({
        'lr': [0.001],
        'batch_size': [32],
        'model': "active_learning",
        'backbone': "vgg16",
        'num_classes': 1001,
        'learning_epoch': 10,
        'query_size': [100],
        'heuristic': ['bald', 'random', 'entropy'],
        'iterations': [1, 20],
        'max_epoch': 100,
        'imagenet_pretraining': [True],
        'dataset': {'path': '/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                    'name': 'active_learning',
                    'task': 'font',
                    'initial_pool': 1000,
                    'uncertainty_config': {'is_bold': {}}}}),
}
