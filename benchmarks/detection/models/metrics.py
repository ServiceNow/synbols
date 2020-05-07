from collections import defaultdict

from scipy import spatial
import numpy as np
import torch


class SegMonitor:
    def __init__(self):
        self.cf = None
        self.n_samples = 0

    def val_on_batch(self, model, batch):
        masks = batch["masks"]
        self.n_samples += masks.shape[0]
        pred_mask = model.predict_on_batch(batch)
        ind = masks != 255
        masks = masks[ind]
        pred_mask = pred_mask[ind]

        labels = np.arange(model.n_classes)
        cf = confusion_multi_class(pred_mask.float(), masks.cuda().float(),
                                   labels=labels)

        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

    def get_avg_score(self):
        # return -1
        Inter = np.diag(self.cf)
        G = self.cf.sum(axis=1)
        P = self.cf.sum(axis=0)
        union = G + P - Inter

        nz = union != 0
        iou = Inter / np.maximum(union, 1)
        mIoU = np.mean(iou[nz])
        iou[~nz] = np.nan

        val_dict = {'val_score': mIoU}
        val_dict['iou'] = iou

        return val_dict


def confusion_multi_class(prediction, truth, labels):
    """
    cf = confusion_matrix(y_true=prediction.cpu().numpy().ravel(),
            y_pred=truth.cpu().numpy().ravel(),
                    labels=labels)
    """
    nclasses = labels.max() + 1
    cf2 = torch.zeros(nclasses, nclasses, dtype=torch.float,
                      device=prediction.device)
    prediction = prediction.view(-1).long()
    truth = truth.view(-1)
    to_one_hot = torch.eye(int(nclasses), dtype=cf2.dtype,
                           device=prediction.device)
    for c in range(nclasses):
        true_mask = (truth == c)
        pred_one_hot = to_one_hot[prediction[true_mask]].sum(0)
        cf2[:, c] = pred_one_hot

    return cf2.cpu().numpy()


def confusion_binary_class(prediction, truth):
    confusion_vector = prediction / truth

    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    cm = np.array([[tn, fp],
                   [fn, tp]])
    return cm
