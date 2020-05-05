from haven import haven_utils as hu

baselines = []
resnet18 = {"name": "resnet18", "imagenet_pretraining": False}
resnet50 = {"name": "resnet50", "imagenet_pretraining": False}
vgg16 = {"name": "vgg16", "imagenet_pretraining": False}
mlp = {"name": "mlp", "depth": 3, "hidden_size": 256}
warn = {"name": "warn"}
conv4 = {"name": "conv4"}
efficientnet = {"name": "efficientnet",
                "type": "efficientnet-b4"}
augmentation = False
for seed in [3, 42, 123]:
    mnist = {
        "name": "mnist",
        "width": 32,
        "height": 32,
        "channels": 1,
        "augmentation": False,
        "mask": None,
        "task": "char"
    }
    svhn = {
        "name": "svhn",
        "width": 32,
        "height": 32,
        "channels": 3,
        "augmentation": False,
        "mask": None,
        "task": "char"
    }
    default_1M = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=1000000_2020-Apr-30.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }
    default = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }
    compositional_char_font = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "compositional_char_font",
    }
    alphabet = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py",
        "task": "alphabet",
        "augmentation": augmentation,
        "mask": "random"
    }
    default_ood = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py",
        "task": "font",
        "mask": "stratified_char",
        "augmentation": augmentation
    }
    default_font = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py",
        "task": "font",
        "mask": "random",
        "augmentation": augmentation
    }
    default_font_ood = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-30.h5py",
        "task": "char",
        "mask": "stratified_font",
        "augmentation": augmentation
    }
    camouflage = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/camouflage_n=100000_2020-Apr-30.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }
    tiny = {
        "name": "synbols_hdf5",
        "width": 8,
        "height": 8,
        "channels": 1,
        "path": "/mnt/datasets/public/research/synbols/tiny_n=10000_2020-Apr-30.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random"
    }
    plain = {
        "name": "synbols_npz",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/plain_n=1000000.npz",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random"
    }

    for lr in [0.001, 0.0001, 0.00001]:
        for dataset in [svhn, mnist, default, camouflage, default_1M, default_font, default_ood, default_font_ood, compositional_char_font]:
            for backbone in [resnet18, resnet50, mlp, conv4, vgg16]:
                    baselines += [{'lr':lr,
                                'batch_size': 512,
                                'min_lr_decay': 1e-3,
                                'amp': 2,
                                "seed": seed,
                                'model': "classification",
                                'backbone': backbone,
                                'max_epoch': 200,
                                'episodic': False,
                                'dataset': dataset}]
            baselines += [{'lr':lr,
                        'batch_size': 128,
                        'min_lr_decay': 1e-3,
                        'amp': 2,
                        "seed": seed,
                        'model': "classification",
                        'backbone': warn,
                        'max_epoch': 200,
                        'episodic': False,
                        'dataset': dataset}]
        for dataset in [tiny]: 
            for backbone in [mlp, conv4]:
                baselines += [{'lr': lr,
                            'batch_size':512,
                            'amp': 2,
                            "seed": seed,
                            'min_lr_decay': 1e-3,
                            'model': "classification",
                            'backbone': backbone,
                            'max_epoch': 200,
                            'episodic': False,
                            'dataset': dataset}]
EXP_GROUPS = {}            
EXP_GROUPS["baselines"] = baselines
EXP_GROUPS["default_font"] = [{'lr': 0.001,
                        'batch_size': 512,
                        'seed': 42,
                        'amp': 1,
                        'min_lr_decay': 1e-3,
                        'model': "classification",
                        'backbone': warn,
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': compositional_char_font}]
EXP_GROUPS["debug"] = [{'lr': 0.001,
                        'batch_size':128,
                        'model': "classification",
                        'backbone': mlp,
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': default_font}]