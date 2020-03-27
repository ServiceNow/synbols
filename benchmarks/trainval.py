import os
import argparse

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc

from datasets import get_dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as tt
from exp_configs import EXP_GROUPS
from models import get_model
import pandas as pd


def trainval(exp_dict, savedir_base, reset=False, wandb=None):
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

    if wandb is not None:
        # https://docs.wandb.com/quickstart
        import wandb as logger
        logger.init(project=wandb)
        logger.config.update(exp_dict)

    # Dataset
    # -----------
    train_dataset = get_dataset('train', exp_dict)
    val_dataset = get_dataset('val', exp_dict)

    # train and val loader
    if exp_dict["episodic"] == False:
        train_loader = DataLoader(train_dataset,
                                    batch_size=exp_dict['batch_size'],
                                    shuffle=True,
                                    num_workers=args.num_workers) 
        val_loader = DataLoader(val_dataset,
                                    batch_size=exp_dict['batch_size'],
                                    shuffle=True,
                                    num_workers=args.num_workers) 
    else: # to support episodes TODO: move inside each model
        from datasets.episodic_dataset import EpisodicDataLoader
        train_loader = EpisodicDataLoader(train_dataset,
                                    batch_size=exp_dict['batch_size'],
                                    shuffle=True,
                                    collate_fn=lambda x: x,
                                    num_workers=args.num_workers) 
        val_loader = EpisodicDataLoader(val_dataset,
                                    batch_size=exp_dict['batch_size'],
                                    shuffle=True,
                                    collate_fn=lambda x: x,
                                    num_workers=args.num_workers) 
                
   
    # Model
    # -----------
    model = get_model(exp_dict)

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
        # model.vis_on_loader(vis_loader, savedir=savedir+"/images/")

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail())
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)
        if wandb is not None:
            for key, values in score_dict.items():
                logger.log({key:values})

    print('experiment completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-v", "--view_experiments", default=None)
    parser.add_argument("-j", "--run_jobs", default=None)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-wb", "--wandb", type=str, default=None)

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
            exp_list += EXP_GROUPS[exp_group_name]


    # Run experiments or View them
    # ----------------------------
    if args.view_experiments:
        # view experiments
        hr.view_experiments(exp_list, savedir_base=args.savedir_base)

    elif args.run_jobs:
        # launch jobs
        # TODO: define experiment-wise
        from haven import haven_jobs as hjb
        run_command = ('python trainval.py -ei <exp_id> -sb %s -nw 1' %  (args.savedir_base))
        job_config = {
            'volume': '/mnt:/mnt',
            'image': 'images.borgy.elementai.net/issam/main',
            'gpu': '1',
            'mem': '16',
            'bid': '1',
            'restartable': '1',
            'cpu': '4',
        }

        workdir = os.path.dirname(os.path.realpath(__file__))
        hjb.run_exp_list_jobs(exp_list, 
                            savedir_base=args.savedir_base, 
                            workdir=workdir,
                            run_command=run_command,
                            job_config=job_config,
                            username='optimass')

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    reset=args.reset,
                    wandb=args.wandb)