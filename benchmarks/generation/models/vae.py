"""
Few-Shot Parallel: trains a model as a series of tasks computed in parallel on multiple GPUs

"""
import numpy as np
import os
import torch
import torch.nn.functional as F
import pylab
import tensorboardX
from utils.losses import get_kl_loss, l1_loss
from utils.metrics import _compute_sap
from tqdm import tqdm
import torch.nn.parallel as parallel 
from apex import amp
from backbones import get_backbone

class MLP(torch.nn.Module):
    def __init__(self, ni, no, nhidden, depth):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            if i == depth - 1:
                setattr(self, "linear%d" %i, torch.nn.Linear(ni, no, bias=False))
            else:
                setattr(self, "linear%d" %i, torch.nn.Linear(ni, nhidden, bias=False))
                setattr(self, "bn%d" %i, torch.nn.BatchNorm1d(nhidden))
            ni = nhidden
    
    def forward(self, x):
        for i in range(self.depth):
            linear = getattr(self, "linear%d" %i)
            x = linear(x)
            if i < self.depth - 1:
                bn = getattr(self, "bn%d" %i)
                x = bn(x)
                x = F.relu(x, True)
        return x

class VAE(torch.nn.Module):
    """Trains a VAE on multiple GPUs"""

    def __init__(self, exp_dict, labelset, writer=None):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        self.model = get_backbone(exp_dict).cuda()
        self.exp_dict = exp_dict 
        self.ngpu = self.exp_dict["ngpu"]
        self.devices = list(range(self.ngpu))
        self.beta = self.exp_dict["beta"]
        self.writer = writer
        self.labelset = labelset
        self.is_categorical = {}
        self.classifiers = {}
        self.optimizers = {}
        for attribute in labelset.keys():
            categorical = len(labelset[attribute]) > 1
            self.is_categorical[attribute] = categorical
            if categorical:
                nclasses = len(labelset[attribute])
            else:
                nclasses = 1
            linear = torch.nn.Linear(self.exp_dict["z_dim"], nclasses).cuda()
            mlp = MLP(self.exp_dict["z_dim"], nclasses, 2 * self.exp_dict["z_dim"], 3).cuda()
            optimizer_linear = torch.optim.Adam(linear.parameters(), lr=exp_dict["lr"])
            optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=exp_dict["lr"])
            self.classifiers["%s_linear" %attribute] = linear
            self.optimizers["%s_linear" %attribute] = optimizer_linear
            self.classifiers["%s_mlp" %attribute] = mlp
            self.optimizers["%s_mlp" %attribute] = optimizer_mlp
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          betas=(0.9, 0.999),
                                          lr=self.exp_dict["lr"])
        if self.exp_dict["amp"] > 0:
            self.model, self.optimizer = amp.initialize(self.model, 
                                                        self.optimizer, opt_level="O%d" % exp_dict["amp"])

    def save_img(self, name, images, idx=0):
        for im in images:
            im[...] = (im - im.min() ) / (im.max() - im.min())
        im = torch.cat(images, 2)
        self.writer.add_image(name, im, idx)

    def get_beta(self, epoch):
        if self.exp_dict["beta"] > 0:
            if self.exp_dict["beta_annealing"] == True:
                cycle_size = (self.exp_dict["max_epoch"] // 4)
                _epoch = epoch % cycle_size
                ratio = max(1e-4, min(1, _epoch / (cycle_size * 0.5)))
                beta = self.beta * ratio
            else:
                beta = self.beta
        else:
            beta = 0
        return beta

    def train_classifiers_on_batch(self, mus, ys):
        ret = {}
        for attribute in ys:
            for classifier_type in ["linear", "mlp"]:
                y = ys[attribute].cuda()
                optimizer = self.optimizers["%s_%s" %(attribute, classifier_type)]
                classifier = self.classifiers["%s_%s" %(attribute, classifier_type)] 
                classifier.train()
                optimizer.zero_grad()
                logits = classifier(mus)
                if self.is_categorical[attribute]:
                    loss = F.cross_entropy(logits, y)
                    ret["train_accuracy_%s_%s" %(attribute, classifier_type)] = float((logits.max(1)[1] == y).float().mean())
                else:
                    logits = logits.squeeze()
                    loss = F.mse_loss(logits, y.float())
                    ret["train_mae_%s_%s" %(attribute, classifier_type)] = float(torch.abs(logits - y).mean())
                loss.backward()
                optimizer.step()
        return ret

    @torch.no_grad()
    def val_classifiers_on_batch(self, mus, ys):
        ret = {}
        for attribute in ys:
            for classifier_type in ["linear", "mlp"]:
                y = ys[attribute].cuda()
                classifier = self.classifiers["%s_%s" %(attribute, classifier_type)] 
                classifier.eval()
                logits = classifier(mus)
                if self.is_categorical[attribute]:
                    loss = F.cross_entropy(logits, y)
                    ret["val_accuracy_%s_%s" %(attribute, classifier_type)] = float((logits.max(1)[1] == y).float().mean())
                else:
                    logits = logits.squeeze()
                    loss = F.mse_loss(logits, y)
                    ret["val_mae_%s_%s" %(attribute, classifier_type)] = float(torch.abs(logits - y).mean())
        return ret

    def train_vae_on_batch(self, epoch, batch_idx, batch, vis_flag):
        x, y, attributes = batch
        b, c, h, w = x.size()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        b = x.size(0)
        if self.ngpu > 1:
            mu, logvar, z, reconstruction = torch.nn.DataParallel(self.model, list(range(self.ngpu)))(x)
        else:
            mu, logvar, z, reconstruction = self.model(x)

        # Encoder
        self.model.zero_grad()
        b = x.size(0)
        kl_loss = get_kl_loss(mu, logvar)
        # pix_mse = ((x - reconstruction) ** 2).sum() / b
        pix_mse = l1_loss(x, reconstruction)
        mse_loss = pix_mse.clone()
        loss = mse_loss
        beta = self.get_beta(epoch)
        loss += kl_loss * beta 

        if self.exp_dict["amp"] > 0:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        ret = dict(train_loss=float(kl_loss + pix_mse),
                    pix_mse=float(pix_mse),
                    kl_loss=float(kl_loss),
                    running_beta=float(beta),
                    mean_mu=float(mu.mean()),
                    mean_logvar=float(logvar.mean()))
        ret.update(self.train_classifiers_on_batch(mu.detach(), attributes))
        return ret
                    

    def val_on_batch(self, epoch, batch_idx, batch, vis_flag):
        x, y, attributes = batch
        b = x.size(0)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        mu, logvar, z, reconstruction = self.model(x)
        kl_loss = get_kl_loss(mu, logvar)
        pix_mse_loss = l1_loss(x, reconstruction)
        if vis_flag and batch_idx == 0:
            # z_2 = torch.randn_like(z)
            # reconstruction_2 = self.model.decode(z_2)
            self.save_img("val_reconstruction", [x[0], reconstruction[0]], epoch)

        ret =  dict(val_kl_loss=float(kl_loss),
                    val_pix_mse_loss=float(pix_mse_loss),
                    val_loss=float(pix_mse_loss + kl_loss),
                    mu=mu.data.cpu().numpy())
        ret.update(self.val_classifiers_on_batch(mu.detach(), attributes))
        return ret

    def predict_on_batch(self, x):
        return self.model(x.cuda()) 

    def train_on_loader(self, epoch, data_loader, vis_flag=False):
        """Iterate over the training set

        Args:
            data_loader: iterable training data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.train()
        self.n_data = len(data_loader.dataset)
        ret = {}
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            res_dict = self.train_vae_on_batch(epoch, batch_idx, batch, False)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        return {k: np.mean(v) for k,v in ret.items()}
        

    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader, vis_flag=True):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        ret = {}
        labels = []
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            x, y, attributes = batch
            labels.append(y.cpu().numpy())
            res_dict = self.val_on_batch(epoch, batch_idx, batch, vis_flag)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        # sap_score = _compute_sap(np.concatenate(ret['mu'], 0), 
        #                   np.concatenate(labels, 0))
        ret = {k: np.mean(v) for k,v in ret.items() if k not in ['mu']}
        # ret["val_sap_score"] = sap_score
        return ret

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(data_loader):
            mse, regularizer, loss = self.val_on_batch(batch_idx, batch, False)
            test_loss_meter.update(float(loss), 1)
        return {"test_loss": test_loss_meter.mean()}

    def get_state_dict(self):
        ret = {}
        ret["model"] = self.model.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        if self.exp_dict["amp"] > 0:
            ret["amp"] = amp.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        if self.exp_dict["amp"] > 0:
            amp.load_state_dict(state_dict["amp"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])
