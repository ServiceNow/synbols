import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import numpy as np
import time
from src import utils as ut
from sklearn.metrics import confusion_matrix
import skimage
from src import models
from haven import haven_img as hi
from skimage import morphology as morph
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import cv2
from . import fcn_resnet
from haven import haven_img
from haven import haven_utils as hu


class LCFCN(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.base_model = fcn_resnet.FCN8(n_classes)

    def train_on_loader(self, train_loader):
        return models.train_on_loader(self, train_loader)

    def val_on_loader(self, val_loader):
        val_monitor = LocMonitor()
        return models.val_on_loader(self, val_loader, val_monitor=val_monitor)

    def vis_on_loader(self, vis_loader, savedir_images, **kwargs):
        return models.vis_on_loader(self, vis_loader, savedir_images=savedir_images)

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()
        batch["images"] = torch.stack(batch["images"])
        batch["counts"] = torch.stack(batch["counts"])
        batch["points"] = torch.stack(batch["points"])
        images = batch["images"].cuda()
        counts = batch["counts"].cuda()

        blobs = self.predict_on_batch(batch, method="blobs").squeeze()
        
        points = batch["points"].long().cuda()

        points_numpy = hu.t2n(points).squeeze()
        blob_dict = get_blob_dict_base(self, blobs, points_numpy)

        
        
        self.train()
        logits = self.model.forward(images)
        loss = lc_loss_base(logits, images, points,
                            counts[None], blob_dict)

        if extras.get('mask_bg') is not None:
            loss += F.cross_entropy(logits, torch.from_numpy(1 - extras['mask_bg']).long().cuda()[None], ignore_index=1)


        loss.backward()

        self.opt.step()

        return {"train_loss":loss.item()}

    def train_on_batch_semseg(self, batch, mask_fg,
                               mask_void,
                               mask_bg):
        self.opt.zero_grad()
        # import ipdb;ipdb.set_trace()
        mask_fg[mask_void==1] = 255
        images, labels = batch["images"], torch.LongTensor(mask_fg)[None]
        images, labels = images.cuda(), labels.cuda()
        
        
        probs = F.log_softmax(self.model(images), dim=1)
        loss = F.nll_loss(probs, labels, reduction='mean', ignore_index=255)
        # if np.random.rand() < 0.5:
        #     loss = loss_map[0,mask_fg==0].mean()
        # else:
        #     loss = loss_map[0,mask_fg==1].mean()
        # focal_loss(self.model(images), labels, ignore_index=255)
        loss.backward()
        self.opt.step()

        return {"train_loss":loss.item()}

    def train_on_batch_void(self, batch, mask_void, points, mask_bg, mask_bg_weight=1):
        self.opt.zero_grad()
        self.train()

        images = batch["images"].cuda()
        counts = batch["counts"].cuda()

        blobs = self.predict_on_batch(batch, method="blobs").squeeze()

        blobs = blobs * (1 - mask_void)

        points = points.long().cuda()
        points_numpy = hu.t2n(points).squeeze()

        blob_dict = get_blob_dict_base(self, blobs, points_numpy)

        self.train()
        logits = self.model.forward(images)
        loss = lc_loss_base(logits, images, points,
                            counts[None], blob_dict)
        if mask_bg is not None:
            loss += mask_bg_weight * F.cross_entropy(logits, torch.from_numpy(1 - mask_bg).long().cuda()[None], ignore_index=1)

        loss.backward()

        self.opt.step()

        return {"train_loss":loss.item()}

    def get_state_dict(self):
        state_dict = {"model": self.model.state_dict(),
                      "opt":self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])


    def val_on_batch(self, batch):
        batch["images"] = torch.stack(batch["images"])
        batch["counts"] = torch.stack(batch["counts"])
        batch["points"] = torch.stack(batch["points"])

        pred_counts = self.predict_on_batch(batch, method="counts") 
        pred_blobs = self.predict_on_batch(batch, method="blobs")
        pred_points = blobs2points(pred_blobs[0,0])

        return self.score_on_batch(batch, pred_counts, pred_points)

    def score_on_batch(self, batch, pred_counts, pred_points):
        val_dict = {}
        
        val_dict["val_mae"] = abs(pred_counts.ravel() - batch["counts"].numpy().ravel())
        

        val_dict["val_error"] = val_dict["val_mae"]
        val_dict["val_errors"] = np.minimum(abs(pred_counts.ravel() - batch["counts"].numpy().ravel()), 1)
        
        gt_points = batch["points"].numpy()
        game_mean = 0.

        for L in range(4):
            game_mean += compute_game(pred_points, gt_points, L=L)
            val_dict["val_game_%d" % L] = np.array([game_mean / (L + 1)])

        return val_dict
        
    def predict_on_batch(self, batch, **options):
        self.eval()
        # feat_8s, feat_16s, feat_32s = self.model.extract_features(batch["images"].cuda())

        if options["method"] == "probs":
            images = batch["images"].cuda()
            logits_mask = self.model.forward(images)
            logits_probs = F.softmax(logits_mask, 1)
            
            return logits_probs[:,1][None]

        if options["method"] == "counts":
            images = batch["images"].cuda()
            pred_mask = self.model.forward(images).data.max(1)[1].squeeze().cpu().numpy()

            counts = np.zeros(self.model.n_classes - 1)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs_category = morph.label(pred_mask == category_id)
                n_blobs = (np.unique(blobs_category) != 0).sum()
                counts[category_id - 1] = n_blobs

            return counts[None]

        elif options["method"] == "blobs":
            images = batch["images"].cuda()
            pred_mask = self.model.forward(images).data.max(1)[1].squeeze().cpu().numpy()

            h, w = pred_mask.shape
            blobs = np.zeros((self.model.n_classes - 1, h, w), int)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs[category_id - 1] = morph.label(pred_mask == category_id)

            return blobs[None]

        elif options["method"] == "points":
            images = batch["images"].cuda()
            logits_mask = self.model.forward(images)
            logits_probs = F.softmax(logits_mask, 1)

            pred_mask = logits_probs.data.max(1)[1].squeeze().cpu().numpy()

            h, w = pred_mask.shape
            blobs = np.zeros((self.model.n_classes - 1, h, w), int)
            points = np.zeros((self.model.n_classes - 1, h, w), int)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs_mask = morph.label(pred_mask == category_id)
                blobs[category_id - 1] = blobs_mask
                for blob_id in np.unique(blobs_mask):
                    if blob_id == 0:
                        continue
                    blob_mask = blobs_mask == blob_id
                    best_index = (blob_mask * logits_probs[:, 1].cpu().numpy()).argmax()
                    y, x = np.unravel_index(best_index, pred_mask.shape)
                    points[0, y, x] = category_id


            return points[None]
        
    def vis_on_batch(self, batch, savedir, return_image=False):
        # from skimage.segmentation import mark_boundaries
        # from skimage import data, io, segmentation, color
        # from skimage.measure import label
        batch["images"] = torch.stack(batch["images"])
        batch["counts"] = torch.stack(batch["counts"])
        batch["points"] = torch.stack(batch["points"])

        self.eval()
        pred_counts = self.predict_on_batch(batch, method="counts") 
        pred_blobs = self.predict_on_batch(batch, method="blobs")
        pred_probs = self.predict_on_batch(batch, method="probs")

        # loc 
        pred_count = pred_counts.ravel()[0]
        pred_blobs = pred_blobs.squeeze()
        
        img_org = hu.get_image(batch["images"],denorm="rgb")

        # true points
        y_list, x_list = np.where(batch["points"][0].long().numpy().squeeze())
        img_peaks = haven_img.points_on_image(y_list, x_list, img_org)
        text = "%s ground truth" % (batch["points"].sum().item())
        haven_img.text_on_image(text=text, image=img_peaks)

        # pred points 
        pred_points = blobs2points(pred_blobs).squeeze()
        y_list, x_list = np.where(pred_points.squeeze())
        img_pred = haven_img.points_on_image(y_list, x_list, img_org)
        text = "%s predicted" % (len(y_list))
        haven_img.text_on_image(text=text, image=img_pred)

        # heatmap 
        heatmap = hi.gray2cmap(pred_probs.squeeze().cpu().numpy())
        heatmap = hu.f2l(heatmap)
        haven_img.text_on_image(text="lcfcn heatmap", image=heatmap)
        
        
        img_mask = np.hstack([img_peaks, img_pred, heatmap])
        if return_image:
            return {'gt':img_peaks,'preds':img_pred, 'heatmap':heatmap, 'img_org':img_org}
        hu.save_image(os.path.join(savedir, "%s.jpg" % 
            str(batch["meta"][0]["index"])), 
            img_mask)
      
class GAME:
    def __init__(self):
        super().__init__(higher_is_better=False)

        self.sum = 0.
        self.n_batches = 0.

        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}

        self.game_dict = {}
        for L in range(4):
            self.game_dict[L] = 0.

        self.game = 0.

    def add_batch(self, model, batch, **options):
        pred_blobs = ms.t2n(model.predict(batch, method="blobs")).squeeze()
        assert pred_blobs.ndim == 2
        pred_points = blobs2points(pred_blobs).squeeze()
        gt_points = ms.t2n(batch["points"]).squeeze()

        game_mean = 0.
        for L in range(4):
            game_sum = compute_game(pred_points, gt_points, L=L)
            self.game_dict[L] += game_sum
            game_mean += game_sum

        self.game += game_mean / (L + 1)
        self.n_batches += 1.


    def get_score_dict(self):
        curr_score = self.game/self.n_batches
        self.score_dict["score"] = curr_score

        # The Rest
        for L in range(4):
            self.score_dict["GAME%d"%L] = self.game_dict[L]/self.n_batches

        return self.score_dict


# -----------------------
# Utils
def blobs2points(blobs):
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)

    assert points.ndim == 2

    for r in rps:
        y, x = r.centroid

        points[int(y), int(x)] = 1


    # assert points.sum() == (np.unique(blobs) != 0).sum()
       
    return points

def compute_game(pred_points, gt_points, L=1):
    n_rows = 2**L
    n_cols = 2**L

    pred_points = pred_points.astype(float).squeeze()
    gt_points = gt_points.astype(float).squeeze()
    h, w = pred_points.shape
    se = 0.

    hs, ws = h//n_rows, w//n_cols
    for i in range(n_rows):
        for j in range(n_cols):

            sr, er = hs*i, hs*(i+1)
            sc, ec = ws*j, ws*(j+1)

            pred_count = pred_points[sr:er, sc:ec]
            gt_count = gt_points[sr:er, sc:ec]
            
            se += float(abs(gt_count.sum() - pred_count.sum()))
    return se



# ==========================================================
# Losses
def lc_loss(model, batch):
    model.train()

    blob_dict = get_blob_dict(model, batch)
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    return lc_loss_base(model, images, points, counts, blob_dict)
    # print(images.shape)


def lc_loss_base(logits, images, points, counts, blob_dict):
    N = images.size(0)
    assert N == 1

    S = F.softmax(logits, 1)
    S_log = F.log_softmax(logits, 1)

    # IMAGE LOSS
    loss = compute_image_loss(S, counts)

    # POINT LOSS
    loss += F.nll_loss(S_log, points,
                       ignore_index=0,
                       reduction='sum')
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += compute_fp_loss(S_log, blob_dict)

    # split_mode loss
    if blob_dict["n_multi"] > 0:
        loss += compute_split_loss(S_log, S, points, blob_dict)

    # Global loss
    S_npy = hu.t2n(S.squeeze())
    points_npy = hu.t2n(points).squeeze()
    for l in range(1, S.shape[1]):
        points_class = (points_npy == l).astype(int)

        if points_class.sum() == 0:
            continue

        T = watersplit(S_npy[l], points_class)
        T = 1 - T
        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss / N


# Loss Utils
def compute_image_loss(S, Counts):
    n, k, h, w = S.size()

    # GET TARGET
    ones = torch.ones(Counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones.float(), Counts.float()], 1)
    Target = (BgFgCounts.view(n * k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h * w).max(2)[0].view(-1)

    loss = F.binary_cross_entropy(Smax, Target, reduction='sum')

    return loss


def compute_fp_loss(S_log, blob_dict):
    blobs = blob_dict["blobs"]

    scale = 1.
    loss = 0.
    n_terms = 0
    for b in blob_dict["blobList"]:
        if n_terms > 25:
            break

        if b["n_points"] != 0:
            continue

        T = np.ones(blobs.shape[-2:])
        T[blobs[b["class"]] == b["label"]] = 0

        loss += scale * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                   ignore_index=1, reduction='mean')

        n_terms += 1
    return loss


def compute_bg_loss(S_log, bg_mask):
    loss = F.nll_loss(S_log, torch.LongTensor(bg_mask).cuda()[None],
                      ignore_index=1, reduction='mean')
    return loss


def compute_split_loss(S_log, S, points, blob_dict):
    blobs = blob_dict["blobs"]
    S_numpy = hu.t2n(S[0])
    points_numpy = hu.t2n(points).squeeze()

    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] < 2:
            continue

        l = b["class"] + 1
        probs = S_numpy[b["class"] + 1]

        points_class = (points_numpy == l).astype("int")
        blob_ind = blobs[b["class"]] == b["label"]

        T = watersplit(probs, points_class * blob_ind) * blob_ind
        T = 1 - T

        scale = b["n_points"] + 1
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss


def watersplit(_probs, _points):
    points = _points.copy()

    points[points != 0] = np.arange(1, points.sum() + 1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    seg = watershed(probs, points)

    return find_boundaries(seg)


@torch.no_grad()
def get_blob_dict(model, batch, training=False):
    blobs = model.predict(batch, method="blobs").squeeze()
    points = hu.t2n(batch["points"]).squeeze()

    return get_blob_dict_base(model, blobs, points, training=training)


def get_blob_dict_base(model, blobs, points, training=False):
    if blobs.ndim == 2:
        blobs = blobs[None]

    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0
    total_size = 0

    for l in range(blobs.shape[0]):
        class_blobs = blobs[l]
        points_mask = points == (l + 1)
        # Intersecting
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True)
        uniques = np.delete(np.unique(class_blobs), blob_uniques)

        for u in uniques:
            blobList += [{"class": l, "label": u, "n_points": 0, "size": 0,
                          "pointsList": []}]
            n_fp += 1

        for i, u in enumerate(blob_uniques):
            if u == 0:
                continue

            pointsList = []
            blob_ind = class_blobs == u

            locs = np.where(blob_ind * (points_mask))

            for j in range(locs[0].shape[0]):
                pointsList += [{"y": locs[0][j], "x": locs[1][j]}]

            assert len(pointsList) == blob_counts[i]

            if blob_counts[i] == 1:
                n_single += 1

            else:
                n_multi += 1
            size = blob_ind.sum()
            total_size += size
            blobList += [{"class": l, "size": size,
                          "label": u, "n_points": blob_counts[i],
                          "pointsList": pointsList}]

    blob_dict = {"blobs": blobs, "blobList": blobList,
                 "n_fp": n_fp,
                 "n_single": n_single,
                 "n_multi": n_multi,
                 "total_size": total_size}

    return blob_dict



class LocMonitor:
    def __init__(self):
        self.ae = 0
        self.ae_game_0 = 0
        self.ae_game_1 = 0
        self.ae_game_2 = 0
        self.ae_game_3 = 0
        self.n_samples = 0
        self.errors = 0 

    def add(self, val_dict):
        self.ae += val_dict["val_mae"].sum()

        self.ae_game_0 += val_dict["val_game_0"].sum()
        self.ae_game_1 += val_dict["val_game_1"].sum()
        self.ae_game_2 += val_dict["val_game_2"].sum()
        self.ae_game_3 += val_dict["val_game_3"].sum()

        self.errors += val_dict["val_errors"].sum()
        self.n_samples += val_dict["val_mae"].shape[0]

    def get_avg_score(self):
        val_mae = self.ae/ self.n_samples
        return {"val_mae":val_mae, 

              "val_game_0":self.ae_game_0/ self.n_samples,
              "val_game_1":self.ae_game_1/ self.n_samples,
              "val_game_2":self.ae_game_2/ self.n_samples,
              "val_game_3":self.ae_game_3/ self.n_samples,

              'val_error': val_mae,
              "val_error_rate":self.errors/self.n_samples,
              "val_score" :-val_mae
              }


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8,
        ignore_index=None) -> torch.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    from kornia.utils import one_hot
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    if ignore_index:
        ind = target != ignore_index
        loss_tmp = torch.sum(target_one_hot[ind] * focal[ind], dim=1)
    else:
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss