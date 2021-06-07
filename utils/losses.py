import torch
from torch import nn
from torchvision.models import vgg19
from collections import namedtuple
from operator import mul
from functools import reduce

from utils.general import get_member
VGGOutput = namedtuple(
    "VGGOutput",
    ["input", "relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"],
)

StyleLayers = namedtuple("StyleLayers",["relu1_2","relu2_2","relu3_3", "relu4_3"])


class PerceptualVGG(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.vgg =  vgg19(pretrained=True)
        self.vgg.eval()

        self.vgg_layers = self.vgg.features


        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )
        self.target_layers = {
            "3": "relu1_2",
            "8": "relu2_2",
            "13": "relu3_2",
            "15" : "relu3_3",
            "22": "relu4_2",
            "24" : "relu4_3",
            "31": "relu5_2",
        }

        if weights is None:
            self.loss_weights = {"input":1., "relu1_2": 1.,"relu2_2": 1.,"relu3_2": 1.,"relu3_3": 1.,"relu4_2": 1.,"relu4_3": 1.,"relu5_2": 1. }
        else:
            assert isinstance(weights, dict) and list(weights.keys()) == list(self.target_layers.keys()), f"The weights passed to PerceptualVGG have to be a dict with the keys {list(self.target_layers.keys())}"
            self.loss_weights = weights

    def forward(self, x):
        # IMPORTANT: Input is assumed to be in range [-1,1] here.
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std


        # add also common reconstruction loss in pixel space
        out = {"input": x}

        for name, submodule in self.vgg_layers._modules.items():
            # x = submodule(x)
            if name in self.target_layers:
                x = submodule(x)
                out[self.target_layers[name]] = x
            else:
                x = submodule(x)

        return out

def vgg_loss(custom_vgg:PerceptualVGG, target, pred, weights=None):
    """
    Implements a vgg based perceptual loss, as extensively used for image/video generation tasks
    :param custom_vgg: The vgg feature extractor for the perceptual loss, definition see above
    :param target:
    :param pred:
    :return:
    """
    target_feats = custom_vgg(target)
    pred_feats = custom_vgg(pred)
    target_feats = VGGOutput(**{key: target_feats[key] for key in VGGOutput._fields})
    pred_feats = VGGOutput(**{key: pred_feats[key] for key in VGGOutput._fields})

    names = list(pred_feats._asdict().keys())
    if weights is None:
        losses = {}

        for i, (tf, pf) in enumerate(zip(target_feats, pred_feats)):
            loss = get_member(custom_vgg,"loss_weights")[VGGOutput._fields[i]] * torch.mean(
                torch.abs(tf - pf)
            ).unsqueeze(dim=-1)
            losses.update({names[i]: loss})
    else:

        losses = {
            names[0]: get_member(custom_vgg,"loss_weights")[VGGOutput._fields[0]]
            * torch.mean(weights * torch.abs(target_feats[0] - pred_feats[0]))
            .unsqueeze(dim=-1)
            .to(torch.float)
        }

        for i, (tf, pf) in enumerate(zip(target_feats[1:], pred_feats[1:])):
            loss = get_member(custom_vgg,"loss_weights")[i + 1] * torch.mean(
                torch.abs(tf - pf)
            ).unsqueeze(dim=-1)

            losses.update({names[i + 1]: loss})

    return losses

def vgg_loss_agg(vgg, target, pred, weights=None):
    """
    To aggreagate the vgg losses
    :param vgg:
    :param target:
    :param pred:
    :param weights:
    :return:
    """
    # basic_device = target.get_device()
    # net_device = list(vgg.parameters())[0].get_device()
    # pred = pred.cuda(net_device)
    # target = target.cuda(net_device)
    loss_list = vgg_loss(vgg,target,pred,weights)
    loss_tensor = torch.stack([loss_list[key] for key in loss_list],dim=0,)
    return loss_tensor.sum()#.cuda(basic_device)


class PixelDynamicsLoss(nn.Module):

    def __init__(self, diff_pp=False):
        super().__init__()
        self.diff_pp = diff_pp

    def forward(self,target_t,target_tk,pred_t,pred_tk):
        if self.diff_pp:
            loss = (((target_t-target_tk).abs()-(pred_t.detach()-pred_tk).abs()).mean())**2
        else:
            loss = ((target_t-target_tk).abs().mean()-(pred_t.detach()-pred_tk).abs().mean())**2

        return loss

def pixel_triplet_loss(target_t,target_tk,pred_t, pred_tk,vgg:PerceptualVGG,layerwise = True, detach=True, diff_pp=False):
    """

    :param vgg:
    :param target_t:
    :param target_tk:
    :param pred_t:
    :param pred_tk:
    :param layerwise:
    :param detach: whether or not to detach the predicted feats at time t
    :param diff_pp: whether to consider differences for each spatial location in each channel or average over all (default average)
    :return:
    """
    if layerwise:
        losses = {}
        # old_device = target_tk.get_device()
        # new_device = list(vgg.parameters())[0].get_device()

        # timestep t
        # target_t = target_t.cuda(new_device)
        # pred_t = pred_t.cuda(new_device)
        target_feats_t = vgg(target_t.cuda())
        pred_feats_t = vgg(pred_t.detach() if detach else pred_t)
        target_feats_t = VGGOutput(**{key: target_feats_t[key] for key in VGGOutput._fields})
        pred_feats_t = VGGOutput(**{key: pred_feats_t[key] for key in VGGOutput._fields})

        # timestep tk
        # target_tk = target_tk.cuda(new_device)
        # pred_tk = pred_tk.cuda(new_device)
        target_feats_tk = vgg(target_tk)
        pred_feats_tk = vgg(pred_tk)
        target_feats_tk = VGGOutput(**{key: target_feats_tk[key] for key in VGGOutput._fields})
        pred_feats_tk = VGGOutput(**{key: pred_feats_tk[key] for key in VGGOutput._fields})

        names = list(pred_feats_t._asdict().keys())
        for i, (tft, pft, tftk, pftk) in enumerate(zip(target_feats_t, pred_feats_t,target_feats_tk, pred_feats_tk)):
            if diff_pp:
                loss = get_member(vgg,"loss_weights")[VGGOutput._fields[i]] * torch.mean((torch.abs(tft - tftk) - torch.abs(pft - pftk)) ** 2).unsqueeze(dim=-1)
            else:
                loss = get_member(vgg,"loss_weights")[VGGOutput._fields[i]] * (torch.mean(torch.abs(tft - tftk)).unsqueeze(dim=-1) - torch.mean(torch.abs(pft - pftk)).unsqueeze(dim=-1))**2

            losses.update({names[i]: loss})

        loss_tensor = torch.stack([losses[key] for key in losses], dim=0, )
        ptl = loss_tensor.sum() #.cuda(old_device)

    else:
        ptl = (vgg_loss_agg(vgg, pred_t.detach(), pred_tk) - vgg_loss_agg(vgg, target_t, target_tk)) ** 2

    return ptl

def style_loss(vgg,style_target, pred):
    target_feats = vgg(style_target)
    pred_feats = vgg(pred)
    target_feats = StyleLayers(**{key: target_feats[key] for key in StyleLayers._fields})
    pred_feats = StyleLayers(**{key: pred_feats[key] for key in StyleLayers._fields})

    names = list(pred_feats._asdict().keys())
    style_outs = {}
    # compute gram matrices and take squared frobenius norm
    for i, (tf,pf) in enumerate(zip(target_feats,pred_feats)):
        shape = pf.shape
        pf = pf.reshape(*shape[:2],-1)
        tf = tf.reshape(*shape[:2],-1)
        gram_diff = 1. / (shape[1]*shape[2]*shape[3]) * (torch.matmul(pf,pf.permute(0,2,1)) - torch.matmul(tf,tf.permute(0,2,1)))
        loss = (torch.norm(gram_diff, p="fro",dim=[1,2])**2).mean()
        style_outs.update({names[i]:loss})

    style_outs = torch.stack([style_outs[key] for key in style_outs])
    return style_outs.sum()

class DynamicsLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, config):
        super().__init__()
        self.mse = nn.MSELoss()


    def forward(self, anchor, positive, negative, ):
        if isinstance(anchor,list) and isinstance(positive,list):
            losses = []
            for a,p in zip(anchor,positive):
                losses.append(self.mse(a,p))

            return torch.stack(losses).mean()
        else:
            return self.mse(anchor,positive)

def kl_loss_check(latents):
    """
    Estimates a gaussian from the latents and returns the kl_divergence between this gaussian and the standard normal
    :param latents:
    :return:
    """
    mu = latents[:,:int(latents.shape[1] / 2)]
    sigma = latents[:,int(latents.shape[1] / 2):]

    # reparameterize
    logstd = nn.Sigmoid()(sigma)

    return kl_loss(mu,logstd)




def kl_loss(mu, logstd):
    if len(mu.shape) != 2:
        mu = mu.reshape(mu.shape[0],-1)
        logstd = logstd.reshape(mu.shape[0],-1)

    dim = mu.shape[1]
    std = torch.exp(logstd)
    kl = torch.sum(-logstd + 0.5 * (std ** 2 + mu ** 2), dim=-1) - (0.5 * dim)

    return kl.mean()
