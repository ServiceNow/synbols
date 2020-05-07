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

for augmentation in [False]:
    mnist = {
        "name": "mnist",
        "width": 32,
        "height": 32,
        "channels": 1,
        "augmentation": augmentation,
        "ood": False,
        "task": "char"
    }
    svhn = {
        "name": "svhn",
        "width": 32,
        "height": 32,
        "channels": 3,
        "augmentation": augmentation,
        "ood": False,
        "task": "char"
    }
    default_1M = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=1000000_2020-Apr-09.h5py",
        "task": "char",
        "augmentation": augmentation,
        "ood": False,
    }
    default = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-09.h5py",
        "task": "char",
        "augmentation": augmentation,
        "ood": False,
    }
    default_ood = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-16.h5py",
        "task": "char",
        "ood": True,
        "augmentation": augmentation
    }
    default_font = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=100000_2020-Apr-16.h5py",
        "task": "font",
        "ood": False,
        "augmentation": augmentation
    }
    default_ood_1M = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/default_n=1000000_2020-Apr-16.h5py",
        "task": "char",
        "ood": True,
        "augmentation": augmentation
    }
    camouflage = {
        "name": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/camouflage_n=100000_2020-Apr-09.h5py",
        "task": "char",
        "augmentation": augmentation
    }
    tiny = {
        "name": "synbols_hdf5",
        "width": 8,
        "height": 8,
        "channels": 1,
        "path": "/mnt/datasets/public/research/synbols/tiny_n=10000_2020-Apr-16.h5py",
        "task": "char",
        "augmentation": augmentation,
        "ood": False,
    }
    plain = {
        "name": "synbols_npz",
        "width": 32,
        "height": 32,
        "channels": 3,
        "path": "/mnt/datasets/public/research/synbols/plain_n=1000000.npz",
        "task": "char",
        "augmentation": augmentation,
        "ood": False,
    }
    for lr in [0.001, 0.0001, 0.00001]:
        for dataset in []:#svhn, mnist, plain, default, camouflage, default_1M, default_font, default_ood, default_ood_1M]:
            for backbone in [resnet18, resnet50, mlp, warn, conv4]:
                    baselines += [{'lr':lr,
                                'batch_size': 256,
                                'amp': 1,
                                'model': "classification",
                                'backbone': backbone,
                                'max_epoch': 100,
                                'episodic': False,
                                'dataset': dataset}]
            baselines += [{'lr': lr,
                        'batch_size':256,
                        'amp': 1,
                        'model': "classification",
                        'backbone': vgg16,
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': dataset}]
        for dataset in [tiny]: 
            for backbone in [mlp, conv4]:
                baselines += [{'lr': lr,
                            'batch_size':512,
                            'model': "classification",
                            'backbone': backbone,
                            'max_epoch': 100,
                            'episodic': False,
                            'dataset': dataset}]
EXP_GROUPS = {}            
EXP_GROUPS["baselines"] = baselines
EXP_GROUPS["default_font"] = [{'lr': 0.001,
                        'batch_size':256,
                        'amp': 3,
                        'model': "classification",
                        'backbone': warn,
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': default_font}]
EXP_GROUPS["debug"] = [{'lr': 0.001,
                        'batch_size':128,
                        'model': "classification",
                        'backbone': mlp,
                        'max_epoch': 100,
                        'episodic': False,
                        'dataset': default_font}]