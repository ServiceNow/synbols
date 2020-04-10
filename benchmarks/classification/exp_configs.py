from haven import haven_utils as hu

# Define exp groups for parameter search
EXP_GROUPS = {'font':
                hu.cartesian_exp_group({
                    'lr':[0.1],
                    'batch_size':[256],
                    'model': "classification",
                    'backbone': "resnet18",
                    'max_epoch': 100,
                    'imagenet_pretraining': False,
                    'episodic': False,
                    'dataset': {'path':'/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                                'name': 'synbols',
                                'task': 'font'}}),
                'font_pretrained': hu.cartesian_exp_group({
                    'lr':[0.01],
                    'batch_size':[256],
                    'model': "classification",
                    'backbone': "resnet18",
                    'max_epoch': 100,
                    'episodic': False,
                    'imagenet_pretraining': [True],
                    'dataset': {'path':'/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                                'name': 'synbols',
                                'task': 'font'}}),
                'font_plain':
                    hu.cartesian_exp_group({
                        'lr':[0.2, 0.1],
                        'batch_size':[512],
                        'model': "classification",
                        'backbone': {"name": "resnet18",
                                        "imagenet_pretraining": False},
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': {'path':'/mnt/datasets/public/research/synbols/plain_n=1000000.npz',
                                    'height': 32,
                                    'width': 32,
                                    'augmentation': False,
                                    'name': 'synbols_npz',
                                    'task': 'font'}}),
                    'font_pretrained': hu.cartesian_exp_group({
                        'lr':[0.01],
                        'batch_size':[256],
                        'model': "classification",
                        'backbone': "resnet18",
                        'max_epoch': 100,
                        'episodic': False,
                        'imagenet_pretraining': [True],
                        'dataset': {'path':'/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                                    'name': 'synbols',
                                    'task': 'font'}}),                
                    'char_camouflage': hu.cartesian_exp_group({
                        'lr':[0.2],
                        'batch_size':[512],
                        'model': "classification",
                        'backbone': "resnet18",
                        'max_epoch': 100,
                        'episodic': False,
                        'imagenet_pretraining': [False],
                        'dataset': {'path': '/mnt/datasets/public/research/pau/synbols/camouflage_n=100000', 
                                    'name': 'synbols_folder',
                                    'task': 'char'}}),   
                                                 
                    'camouflage_mlp': hu.cartesian_exp_group({
                        'lr':[0.2],
                        'batch_size':[512],
                        'model': "classification",
                        'backbone': {"name": "mlp",
                                     "depth": 3,
                                     "hidden_size": 4096},
                        'max_epoch': 100,
                        'episodic': False,
                        'imagenet_pretraining': [False],
                        'dataset': {'path': '/mnt/datasets/public/research/synbols/camouflage_n=100000.npz', 
                                    'name': 'synbols_npz',
                                    'width': 32,
                                    'height': 32,
                                    'task': 'char'}}), 
                    'camouflage_vgg': hu.cartesian_exp_group({
                        'lr':[0.2],
                        'batch_size':[128],
                        'model': "classification",
                        'backbone': {"name": "efficientnet",
                                     "type": "efficientnet-b4",
                                       "imagenet_pretraining": True},
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': {'path': '/mnt/datasets/public/research/synbols/camouflage_n=100000.npz', 
                                    'name': 'synbols_npz',
                                    'width': 32,
                                    'height': 32,
                                    'augmentation': True}}),
                    'default_warn': hu.cartesian_exp_group({
                        'lr':[0.0001],
                        'batch_size':[128],
                        'model': "classification",
                        'backbone': {"name": "conv4"},
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': {'path': '/mnt/datasets/public/research/synbols/default_n=100000.npz', 
                                    'name': 'synbols_npz',
                                    'task': 'char',
                                    'width': 32,
                                    'height': 32,
                                    'augmentation': True}}),
                    'mnist': hu.cartesian_exp_group({
                        'lr':[0.2],
                        'batch_size':[256],
                        'model': "classification",
                        'backbone': {"name": "vgg16",
                                     "type": "efficientnet-b4",
                                       "imagenet_pretraining": True},
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': {
                                    'name': 'mnist',
                                    'width': 28,
                                    'height': 28,
                                    'augmentation': True}}),
                }
baselines = []
resnet18 = {"name": "resnet18", "imagenet_pretraining": False}
resnet50 = {"name": "resnet50", "imagenet_pretraining": False}
vgg16 = {"name": "vgg16", "imagenet_pretraining": False}
mlp = {"name": "mlp", "depth": 3, "hidden_size": 256}
warn = {"name": "warn"}
conv4 = {"name": "conv4"}
efficientnet = {"name": "efficientnet",
                "type": "efficientnet-b4"}

for augmentation in [False]:
    mnist = {
        "name": "mnist",
        "width": 32,
        "height": 32,
        "augmentation": augmentation
    }
    default = {
        "name": "synbols_npz",
        "width": 32,
        "height": 32,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000.npz",
        "task": "char",
        "augmentation": augmentation
    }
    camouflage = {
        "name": "synbols_npz",
        "width": 32,
        "height": 32,
        "path": "/mnt/datasets/public/research/synbols/camouflage_n=100000.npz",
        "task": "char",
        "augmentation": augmentation
    }
    plain = {
        "name": "synbols_npz",
        "width": 32,
        "height": 32,
        "path": "/mnt/datasets/public/research/synbols/plain_n=1000000.npz",
        "task": "char",
        "augmentation": augmentation
    }
    for dataset in [plain, default, mnist]:
        for backbone in [resnet50, resnet18, mlp, warn, conv4]:
            for lr in [0.001]:
                baselines += [{'lr':lr,
                            'batch_size': 512,
                            'model': "classification",
                            'backbone': backbone,
                            'max_epoch': 100,
                            'episodic': False,
                            'dataset': dataset}]
        baselines += [{'lr': lr,
                    'batch_size':256,
                    'model': "classification",
                    'backbone': vgg16,
                    'max_epoch': 100,
                    'episodic': False,
                    'dataset': dataset}]
                
EXP_GROUPS["baselines"] = baselines