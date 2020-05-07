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
import pprint


def trainval(exp_dict, savedir_base, reset=False):
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
    pprint.pprint(exp_dict)
    print("Experiment saved in %s" % savedir)

    # Dataset
    # -----------
    train_dataset = get_dataset('train', exp_dict)
    val_dataset = get_dataset('val', exp_dict)

    # train and val loader
    train_loader = DataLoader(train_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=True,
                                num_workers=args.num_workers) 
    val_loader = DataLoader(val_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=True,
                                num_workers=args.num_workers) 

    # Model
    # -----------
    model = get_model(model_dict=exp_dict['model'], exp_dict=exp_dict, train_set=train_dataset).cuda()

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

    vis_loader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    # model.vis_on_loader(
    #         vis_loader, savedir=os.path.join(savedir, "images"))

    # Train & Val
    # ------------
    print("Starting experiment at epoch %d" % (s_epoch))

    for e in range(s_epoch, exp_dict['max_epoch']):
        # Validate only at the start of each cycle
        score_dict = {}

        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate and Visualize the model
        val_dict = model.val_on_loader(val_loader)
        score_dict.update(val_dict)
        model.vis_on_loader(
            vis_loader, savedir=os.path.join(savedir, "images"))

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = len(score_list)

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

        # Save Best Checkpoint
        if "val_score" in score_df.columns and (score_dict.get("val_score", 0) > score_df["val_score"][:-1].fillna(0).max()):
            hu.save_pkl(os.path.join(
                savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                          model.get_state_dict())
            print("Saved Best: %s" % savedir)

    print('Experiment completed et epoch %d' % e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=None)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)

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
    if args.run_jobs:
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
            'cpu': '4'}
        workdir = os.path.dirname(os.path.realpath(__file__))
        hjb.run_exp_list_jobs(exp_list, 
                            savedir_base=args.savedir_base, 
                            workdir=workdir,
                            run_command=run_command,
                            job_config=job_config)

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    reset=args.reset)