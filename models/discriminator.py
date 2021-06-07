import torch
from torch import nn
from torch.optim import Adam
import functools
from torch.nn.utils import spectral_norm
import math
import numpy as np


from utils.general import get_member
from models.blocks import SPADE

class GANTrainer(object):

    def __init__(self, config, load_fn,logger,spatial_size=128, parallel=False, devices=None, debug=False, temporal=False, sequence_length = None):
        self.config = config
        self.logger = logger
        # disc
        self.logger.info("Load discriminator model")
        self.temporal = temporal
        if self.temporal:
            assert sequence_length is not None
            self.key = "gan_temp"
            self.disc = resnet(config=config[self.key],spatial_size=spatial_size,sequence_length=sequence_length)
            self.load_key = "disc_temp"
            self.postfix = "temp"
            if self.disc.cond:
                self.logger.info(f"Using Conditional temporal discriminator.")
        else:
            self.key = "gan"
            self.disc = PatchDiscriminator(self.config[self.key])
            self.load_key = "disc_patch"
            self.postfix = "patch"
        self.cond = self.config[self.key]["conditional"] if self.temporal and "conditional" in self.config[self.key] else False
        self.logger.info(f"Number of parameters in discriminator_{self.postfix} is {sum(p.numel() for p in self.disc.parameters())}.")
        self.parallel = parallel
        self.devices = devices
        if self.parallel:
            assert self.devices is not None


        # load checkpoint if there's any and it is required
        disc_ckpt = disc_op_ckpt = None
        if self.config["general"]["restart"] and not debug:
            disc_ckpt, disc_op_ckpt = load_fn(key=self.load_key)
            if disc_ckpt is not None:
                self.logger.info(f"Resuming training of discriminator...loading weights.")
                self.disc.load_state_dict(disc_ckpt)

        if self.parallel:
            self.disc = nn.DataParallel(self.disc,device_ids=self.devices)
            self.disc.cuda(self.devices[0])
        else:
            self.disc.cuda()
        self.logger.info("Discriminator on gpu!")



        # disc optimizer
        self.disc_opt = Adam(self.disc.parameters(), lr=self.config["training"]["lr"])
        if self.config["general"]["restart"] and disc_op_ckpt is not None:
            self.disc_opt.load_state_dict(disc_op_ckpt)

        # scheduler for disc optimizer
        milestones = [int(self.config["training"]["n_epochs"] * t) for t in self.config["training"]["tau"]]
        self.disc_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.disc_opt, milestones=milestones, gamma=self.config["training"]["lr_reduce"])


    def train_step(self, x_in_true, x_in_fake, cond=None):
        # predict

        cond = cond if self.cond else None
        self.disc.train()

        # if self.parallel:
        #     x_in_true = x_in_true.cuda(self.devices[0])
        #     x_in_fake = x_in_fake.cuda(self.devices[0])
        # set gradient to zero
        self.disc_opt.zero_grad()

        # real examples
        x_in_true.requires_grad_()

        pred_true, _ = self.disc(x_in_true, cond)
        loss_real = get_member(self.disc,"loss")(pred_true, real=True)
        if self.config[self.key]["gp_weight"] > 0:
            loss_real.backward(retain_graph=True)
            # gradient penalty
            loss_gp = get_member(self.disc,"gp")(pred_true, x_in_true).mean()
            gp_weighted = self.config[self.key]["gp_weight"] * loss_gp
            gp_weighted.backward()
        else:
            loss_real.backward()

        # fake examples
        pred_fake, _ = self.disc(x_in_fake.detach(),cond)
        loss_fake = get_member(self.disc,"loss")(pred_fake, real=False)
        loss_fake.backward()

        # optmize parameters
        self.disc_opt.step()

        loss_disc = ((loss_real + loss_fake) / 2.).item()
        out_dict = {f"loss_disc_{self.postfix}": loss_disc, f"p_true_{self.postfix}": torch.sigmoid(pred_true).mean().item(), f"p_fake_{self.postfix}": torch.sigmoid(pred_fake).mean().item(),
                    f"loss_gp_{self.postfix}": loss_gp.item() if self.config[self.key]["gp_weight"] > 0 else 0 }

        # train generator
        pred_fake, fmap_fake = self.disc(x_in_fake,cond)
        _, fmap_true = self.disc(x_in_true,cond)
        if get_member(self.disc,"bce_loss"):
            loss_gen = get_member(self.disc,"bce")(pred_fake, torch.ones_like(pred_fake))
        else:
            loss_gen = -torch.mean(pred_fake)



        loss_fmap = get_member(self.disc,"fmap_loss")(fmap_fake, fmap_true)
        # if self.parallel:
        #     loss_fmap = loss_fmap.cuda(self.devices[0])
        #     loss_gen = loss_gen.cuda(self.devices[0])

        return out_dict, loss_gen, loss_fmap


# code taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, config, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        deep_disc = config["deep_disc"] if "deep_disc" in config else False
        input_nc = 6 if config["pixel_dynamics"] else 3
        n_deep_layers = config["deep_layers"]
        ndf = 64
        n_layers = config["n_layers"]
        self.bce_loss = config["bce_loss"]
        if self.bce_loss:
            self.bce = nn.BCEWithLogitsLoss()

        kw = 4
        padw = 1
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.in_conv = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        nf_mult = 1
        nf_mult_prev = 1
        self.act_fn = nn.LeakyReLU(0.2, True)
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norms.append(norm_layer(ndf * nf_mult))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        self.norms.append(norm_layer(ndf * nf_mult))
        n_d = ndf * nf_mult
        if deep_disc:
            n_max = 1024
            for i in range(n_deep_layers):
                # add one layer to the original patch discrminator to make it more powerful
                self.layers.append(nn.Conv2d(n_d, min(n_max, n_d*2), kernel_size=kw, stride=1, padding=padw, bias=use_bias))
                self.norms.append(norm_layer(min(n_max, n_d*2)))
                n_d = min(n_max, n_d*2)
        self.out_conv = nn.Conv2d(n_d, 1, kernel_size=kw, stride=1, padding=padw)  # output 1 channel prediction map


    def forward(self, input,cond=None):
        """Standard forward."""
        x = self.act_fn(self.in_conv(input))
        fmap = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.act_fn(self.norms[i](x))
            fmap.append(x)
        x = self.out_conv(x)
        return x, fmap

    def loss(self, pred, real):
        if self.bce_loss:
            # vanilla gan loss
            return self.bce(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))
        else:
            # hinge loss
            if real:
                l = torch.mean(torch.nn.ReLU()(1.0 - pred))
            else:
                l = torch.mean(torch.nn.ReLU()(1.0 + pred))
            return l

    def gp(self, pred_fake, x_fake):
        batch_size = x_fake.size(0)
        grad_dout = torch.autograd.grad(
            outputs=pred_fake.sum(), inputs=x_fake,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_fake.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def fmap_loss(self, fmap1, fmap2, loss="l1"):
        recp_loss = 0
        for idx in range(len(fmap1)):
            if loss == "l1":
                recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
            if loss == "l2":
                recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
        return recp_loss / len(fmap1)



######################################################################################################
###3D-ConvNet Implementation from https://github.com/tomrunia/PyTorchConv3D ##########################
def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def conv3x3x3(in_planes, out_planes, stride=1, stride_t=1):
    # 3x3x3 convolution with padding
    return spectral_norm(nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 3),
        stride=[stride_t, stride, stride],
        padding=[1, 1, 1],
        bias=False))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, stride_t=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, stride_t)
        self.bn1 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 spatial_size,
                 sequence_length,
                 config):
        super(ResNet, self).__init__()
        # spatial_size = config["spatial_size"]
        self.inplanes = 64
        self.bce_loss = config["bce_loss"]
        min_spatial_size = int(spatial_size / 8)
        #sample_duration = dic.Network['sequence_length']-1
        self.max_channels = config["max_channels"] if "max_channels" in config else 256
        self.conv1 = spectral_norm(nn.Conv3d(
            3,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False))
        self.gn1      = nn.GroupNorm(num_groups=16, num_channels=64)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

        self.layers = nn.ModuleList()
        self.patch_temp = config["patch_temp_disc"]
        self.spatio_temporal = config["spatio_temporal"] if"spatio_temporal" in config else False
        if self.patch_temp:
            self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
            self.layers.append(self._make_layer(block, 128, layers[1], stride=1, stride_t=1))
            self.layers.append(self._make_layer(block, 128, layers[2], stride=2, stride_t=1))
            self.layers.append(self._make_layer(block, 256, layers[3], stride=2, stride_t=1))
            last_size = int(math.ceil(spatial_size / 16))
            last_duration = 1
            self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
            self.cond = config["conditional"] if "conditional" in config else False
            if self.cond:
                self.spade_emb = SPADE(norm_nc=block.expansion * 256, label_nc=2, config=config)
            self.fc = nn.Linear(256 * block.expansion, config["num_classes"], bias=False)
        else:
            spatial_size /= 2
            self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
            n_channels = 64
            if "conditional" in config and config["conditional"]:
                raise ValueError("If non-patch-gan temporal discriminator is used, conditional must not be True!")
            self.cond = False
            n = 0
            while sequence_length > 1:
                blocks = layers[n] if n<sequence_length-1 else layers[-1]
                n_channels = min(2*n_channels,self.max_channels)
                stride = 1 if spatial_size <= min_spatial_size else 2
                spatial_size = int(spatial_size / stride)
                stride_t = 1 if self.spatio_temporal else (2 if sequence_length > 1 else 1)
                self.layers.append(self._make_layer(block,n_channels,blocks,stride=stride,stride_t=stride_t))
                sequence_length = int(math.ceil(sequence_length / 2))
                n += 1

            self.final = nn.Conv2d(n_channels,1,3,padding=1)


        print(f"Temporal discriminator has {len(self.layers)} layers")


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.orthogonal_(m.weight)
    def _make_layer(self, block, planes, blocks, stride=1, stride_t=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or stride_t != 1:
            downsample = nn.Sequential(
                spectral_norm(nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=[3, 3, 3],
                    stride=[stride_t, stride, stride],
                    padding=[1, 1, 1],
                    bias=False)),
                nn.GroupNorm(num_channels=planes * block.expansion, num_groups=16))
        layers = []
        layers.append(block(self.inplanes, planes, stride, stride_t, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x, cond=None):
        out = []
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        out.append(x)

        for n in range(len(self.layers)):
            x = self.layers[n](x)
            out.append(x)


        if self.patch_temp:
            if self.cond:
                x_norm = []
                for i in range(x.size(2)):
                    x_norm.append(self.spade_emb(x[:,:,i],cond))
                x_norm = torch.stack(x_norm,2)
            else:
                x_norm = x
            x1 = self.avgpool(x_norm)
            output = []
            for i in range(x1.size(2)):
                output.append(self.fc(x1[:,:,i].reshape(x1.size(0), -1)))
            return torch.cat(output, dim=1), out
        else:

            output = self.final(x.squeeze(2))
            return output, out




    def loss(self, pred, real):
        if self.bce_loss:
            # vanilla gan loss
            return self.bce(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))
        else:
            # hinge loss
            if real:
                l = torch.mean(torch.nn.ReLU()(1.0 - pred))
            else:
                l = torch.mean(torch.nn.ReLU()(1.0 + pred))
            return l

    def gp(self, pred_fake, x_fake):
        batch_size = x_fake.size(0)
        grad_dout = torch.autograd.grad(
            outputs=pred_fake.sum(), inputs=x_fake,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_fake.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def fmap_loss(self, fmap1, fmap2, loss="l1"):
        recp_loss = 0
        for idx in range(len(fmap1)):
            if loss == "l1":
                recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
            if loss == "l2":
                recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
        return recp_loss / len(fmap1)

#         return output, out, mu
if __name__ == '__main__':
    ## Test 3dconvnet with dummy input
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    config = {"num_classes": 1, "patch_temp_disc": True,"spatial_size": 128, "bce_loss": False, "conditional": True}

    dummy = torch.rand((2, 3, 6, 128, 128)).cuda()
    dummy_cond = torch.rand((2, 2, 128, 128)).cuda()
    model = resnet(config=config,spatial_size=128, sequence_length=dummy.shape[2]).cuda()
    print("Number of parameters in generator", sum(p.numel() for p in model.parameters()))

    if config["conditional"]:
        out, out2 = model(dummy,dummy_cond)
    else:
        out, out2,= model(dummy)
    test = 1