from haven import haven_utils as hu

# Define exp groups for parameter search
EXP_GROUPS = {'font':
                hu.cartesian_exp_group({
                    'lr':[0.1],
                    'batch_size':[256],
                    'model': "trainer",
                    'backbone': "resnet18",
                    'max_epoch': 100,
                    'imagenet_pretraining': False,
                    'dataset': {'path':'/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                                'name': 'synbols',
                                'task': 'font'}}),
                'font_pretrained': hu.cartesian_exp_group({
                    'lr':[0.01],
                    'batch_size':[256],
                    'model': "trainer",
                    'backbone': "resnet18",
                    'max_epoch': 100,
                    'imagenet_pretraining': [True],
                    'dataset': {'path':'/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                                'name': 'synbols',
                                'task': 'font'}}),
                }