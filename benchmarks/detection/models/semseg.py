import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
from . import metrics
import tqdm
import pylab as plt
import numpy as np
import scipy.sparse as sps
from collections.abc import Sequence
import time
from src import utils as ut
from sklearn.metrics import confusion_matrix
import skimage
from haven import haven_utils as hu
from haven import haven_img as hi
from torchvision import transforms
from src import models
from models import base

from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

# from . import losses, metrics


class SemSeg(torch.nn.Module):
    def __init__(self, exp_dict, train_set, savedir):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = train_set.n_classes
        self.exp_dict = exp_dict

        self.model_base = base.get_base(self.exp_dict['model'].get('base', 'unet2d'),
                                               self.exp_dict, n_classes=self.n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=self.exp_dict.get("lr", 1e-3), betas=(0.99, 0.999))

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=self.exp_dict.get("lr", 1e-3), momentum=0.9, weight_decay=1e-4)

        else:
            raise ValueError

    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    def train_on_loader(model, train_loader):
        model.train()

        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()
    
        for batch in train_loader:
            score_dict = model.train_on_batch(batch)
            
            train_monitor.add(score_dict)
            msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            pbar.update(1)
            
        pbar.close()

        return train_monitor.get_avg_score()

    @torch.no_grad()
    def val_on_loader(self, val_loader):
        self.eval()

        # metrics
        seg_monitor = metrics.SegMonitor()

        n_batches = len(val_loader)
        pbar = tqdm.tqdm(desc="Validating", total=n_batches, leave=False)
        for i, batch in enumerate(val_loader):
            seg_monitor.val_on_batch(self, batch)
            pbar.set_description("Validating: %.4f mIoU" %
                                 (seg_monitor.get_avg_score()['val_score']))
            pbar.update(1)

        pbar.close()
        val_dict = seg_monitor.get_avg_score()
        out_dict = {}
        for c in range(self.n_classes):
            out_dict['iou_group%d' % c] = val_dict['iou'][c]

        return out_dict



    @torch.no_grad()
    def vis_on_loader(self, vis_loader, savedir):
        self.eval()

        os.makedirs(savedir, exist_ok=True)
        # pool_dataset = PoolDataset(self.label_manager, img_ind_list)
        n_batches = len(vis_loader)
        for i, batch in enumerate(vis_loader):
            # print("%d - visualizing %s image - savedir:%s" %
            #       (i, batch["meta"][0]["split"], os.path.abspath(savedir).split("/")[-2]))

            self.vis_on_batch(batch, savedir_image=os.path.join(
                savedir, "%s.jpg" % str(batch["meta"]["index"][0])))
            if i > 5:
                break

    def train_on_batch(self, batch, **extras):
        self.train()

        self.model_base.train()
        self.opt.zero_grad()

        images, labels = batch["images"], batch["masks"]
        images, labels = images.cuda(), labels.cuda()

        if self.exp_dict['model'].get('loss', 'cross_entropy') == 'cross_entropy':
            probs = F.log_softmax(self.model_base(images), dim=1)
            loss = F.nll_loss(
                probs, labels.long(), reduction='mean', ignore_index=255)
        elif self.exp_dict['model']['loss'] == 'dice':
            probs = F.softmax(self.model_base(images), dim=1)
            loss = 0.
            for l in labels.unique():
                if l == 255:
                    continue
                ind = labels != 255
                loss += losses.dice_loss(probs[:, l][ind],
                                         (labels[ind] == l).long()) 
            # loss = F.nll_loss(probs, labels, reduction='mean', ignore_index=255)

        if loss != 0:
            loss.backward()

        self.opt.step()

        return {"train_loss": float(loss)}

    def probs_on_batch(self, batch):
        images = batch["images"].cuda()
        n = images.shape[0]
        logits = self.model_base.forward(images)
        return logits

    def predict_on_batch(self, batch):
        images = batch["images"].cuda()
        n = images.shape[0]
        logits = self.model_base.forward(images)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        # clf
        pred_mask = self.predict_on_batch(batch).cpu()
        # print(pred_mask.sum())
        img = hu.f2l(batch['images'])[0]

        mask_vis = batch["masks"].clone().float()[0][..., None]
        mask_vis[mask_vis == 255] = 0

        pred_mask_vis = pred_mask.clone().float()[0][..., None]
        vmax = 0.1

        fig, ax_list = plt.subplots(ncols=3, nrows=1)
        ax_list[0].imshow(img[:, :, 0], cmap='gray',
                        #   interpolation='sinc', vmin=0, vmax=0.4
                          )

        colors_all = np.array(['black', 'red', 'blue', 'green', 'purple'])
        colors = colors_all[np.unique(mask_vis).astype(int)]

        vis = label2rgb(mask_vis[:, :, 0].numpy(), image=img.numpy(
        ), colors=colors, bg_label=255, bg_color=None, alpha=0.6, kind='overlay')
        vis = mark_boundaries(
            vis, mask_vis[:, :, 0].numpy().astype('uint8'), color=(1, 1, 1))

        ax_list[1].imshow(vis, cmap='gray')

        colors = colors_all[np.unique(pred_mask_vis).astype(int)]
        vis = label2rgb(pred_mask_vis[:, :, 0].numpy(), image=img.numpy(
        ), colors=colors, bg_label=255, bg_color=None, alpha=0.6, kind='overlay')
        vis = mark_boundaries(
            vis, pred_mask_vis[:, :, 0].numpy().astype('uint8'), color=(1, 1, 1))

        ax_list[2].imshow(vis, cmap='gray')

        for i in range(1, self.n_classes):
            plt.plot([None], [None], label='group %d' % i, color=colors_all[i])
        # ax_list[1].axis('off')
        ax_list[0].grid()
        ax_list[1].grid()
        ax_list[2].grid()

        ax_list[0].tick_params(axis='x', labelsize=6)
        ax_list[0].tick_params(axis='y', labelsize=6)

        ax_list[1].tick_params(axis='x', labelsize=6)
        ax_list[1].tick_params(axis='y', labelsize=6)

        ax_list[2].tick_params(axis='x', labelsize=6)
        ax_list[2].tick_params(axis='y', labelsize=6)

        ax_list[0].set_title('Original image', fontsize=8)
        ax_list[1].set_title('Ground-truth',  fontsize=8)
        ax_list[2].set_title('Prediction',  fontsize=8)

        legend_kwargs = {"loc": 2, "bbox_to_anchor": (1.05, 1),
                         'borderaxespad': 0., "ncol": 1}
        ax_list[2].legend(fontsize=6, **legend_kwargs)
        plt.savefig(savedir_image.replace('.jpg', '.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}
