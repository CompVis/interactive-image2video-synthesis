import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

from models.blocks import Conv2dBlock, ResBlock, AdaINLinear, NormConv2d,ConvGRU

class OscillatorModel(nn.Module):
    def __init__(self,spatial_size,config,n_no_motion=2, logger=None):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16

        self.reparamterize = config["reparameterize_poke"] if "reparameterize_poke" in config else False
        self.norm_layer = config["norm_layer"] if "norm_layer" in config else "in"
        self.layers = config["layers"]
        self.n_gru_layers = config["n_gru_layers"] if "n_gru_layers" in config else 3
        self.n_no_motion = n_no_motion
        self.n_stages = len(self.layers)
        assert self.n_no_motion is not None




        nf_first_shape_enc = int(max(32, config["nf_deep"] / (2 ** self.n_stages)))
        self.shape_enc = SkipConnectionEncoder(nf_in=3, nf_max=self.layers[-1], n_stages=self.n_stages, n_skip_stages=self.n_stages
                                               ,nf_first=nf_first_shape_enc, norm_layer=self.norm_layer,layers=self.layers)

        self.dynamics_enc = Encoder(2, nf_max=self.layers[-1], n_stages=self.n_stages,
                                    variational=self.reparamterize, norm_layer=self.norm_layer, layers=self.layers)
        ups = [False] * self.n_gru_layers
        #input_sizes = [self.layers[-1]*2] + (len(self.layers)-1) * [self.layers[-1]]
        self.fusion_block = ConvGRU(input_size=self.layers[-1] * 2, hidden_sizes=self.layers[-1], kernel_sizes=3, n_layers=self.n_gru_layers,
                                    upsampling=ups,)

        self.dec = SkipConnectionDecoder(nf_in=self.layers[-1], in_channels=self.shape_enc.depths, n_skip_stages=len(self.layers),
                                         disentanglement=False, norm_layer=self.norm_layer, layers=self.layers)

        if logger is not None:
            logger.info("Constructed OscillatorModel")
            logger.info(f"Layers of OscillatorModel is {self.layers}")
            logger.info(f"Encoder channels of oscillator model is {self.layers}")


    def forward(self,input_img, poke, len,n_ref,target_img=None):

        imgs = []
        sigmas_hat_out = []


        if target_img==None:
            target_img = input_img

        if self.reparamterize:
            delta, mu, _ = self.dynamics_enc(poke)
        else:
            delta = self.dynamics_enc(poke)[0]
        # only first time shape encoding
        # if self.poke_scale_mode and not poke_linear:
        sigmas = self.shape_enc(input_img)
        sigma_tgt = self.shape_enc(target_img)[-1]

        sigma_dyn = sigmas.pop()
        pred = [sigma_dyn] * self.n_gru_layers
        pred_out = pred[-1]

        for n in range(len):
            # apply fusion block: input is delta, hidden states are the sigma_n

            # get inputs for network
            delta_in = delta * (1. - float(n)/(n_ref-1)) if n <= n_ref else torch.zeros_like(delta)
            if self.training:
                sigma_diff = pred_out - sigma_tgt if n < len - self.n_no_motion else torch.zeros_like(pred_out)
            else:
                sigma_diff = pred_out - sigma_tgt
            delta_in = torch.cat([delta_in,sigma_diff],1)

            # predict object encoding at next time step
            pred = self.fusion_block(delta_in, pred)
            pred_out = pred[-1]
            sigmas.append(pred_out)

            # decode
            x = self.dec(sigmas, [], del_shape=False)
            imgs.append(x)
            sigmas_hat_out.append(pred_out)
            #remove pred
            sigmas.pop()

        imgs = torch.stack(imgs, dim=1)
        #sigmas_hat_out[-1].reverse()

        return imgs, sigmas_hat_out



class SAVPArchModel(nn.Module):

    def __init__(self, spatial_size, config):
        super().__init__()

        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.poke_every_t = config["poke_every_t"] if "poke_every_t" in config else True

        self.dynamics_enc = Encoder(nf_in=2, nf_max=64, n_stages=self.n_stages)
        self.gen = SAVPGenerator(self.poke_every_t)


    def forward(self,img,poke,len):

        # encode dynamics
        delta = self.dynamics_enc(poke)[0]

        out = self.gen(img,delta,len)

        return out

class SAVPGenerator(nn.Module):
    def __init__(self, poke_every):
        super().__init__()

        self.poke_every_t = poke_every
        # encoder stuff
        self.conv_e1 = Conv2dBlock(3, 32, 3, 2, norm="in", padding=1, activation="relu")
        # ssize 32
        self.rnn_e1 = ConvGRU(32, 32, 3, 1)
        self.conv_e2 = Conv2dBlock(32, 64, 3, 2, norm="in", padding=1, activation="relu")
        # ssize 16
        self.rnn_e2 = ConvGRU(64, 64, 3, 1)

        # bottleneck
        self.conv_bn = Conv2dBlock(128, 128, 3, 2, norm="in", padding=1, activation="relu")
        # ssize 8
        self.rnn_bn = ConvGRU(128, 64, 3, 1)

        # decoder stuff
        self.up1 = nn.Upsample((16, 16), mode="bilinear")
        # ssize 16
        self.conv_d1 = Conv2dBlock(128, 64, 3, 1, norm="in", padding=1, activation="relu")
        self.rnn_d1 = ConvGRU(64, 32, 3, 1)
        self.up2 = nn.Upsample((32, 32), mode="bilinear")
        # ssize 32
        self.conv_d2 = Conv2dBlock(64, 32, 3, 1, norm="in", padding=1, activation="relu")
        self.rnn_d2 = ConvGRU(32, 32, 3, 1)
        self.up3 = nn.Upsample((64, 64), mode="bilinear")

        self.conv_out = Conv2dBlock(32, 3, 3, 1, 1, norm="none", activation="tanh")

    def forward(self,img,delta,len):

        x = img
        out_imgs = []
        for t in range(len):
            x1e = self.conv_e1(x)
            x1er = self.rnn_e1(x1e,[x1er] if t>0 else None)[0]

            x2e = self.conv_e2(x1er)
            x2er = self.rnn_e2(x2e,[x2er] if t>0 else None)[0]

            if t > 0 and not self.poke_every_t:
                delta = np.zeros_like(delta)

            xbn = torch.cat([x2er,delta],dim=1)
            xbn = self.conv_bn(xbn)
            xbnr = self.rnn_bn(xbn,[xbnr] if t>0 else None)[0]

            x1d = self.up1(xbnr)
            x1d = self.conv_d1(torch.cat([x1d,x2er],1))
            x1dr = self.rnn_d1(x1d,[x1dr] if t>0 else None)[0]

            x2d = self.up2(x1dr)
            x2d = self.conv_d2(torch.cat([x2d,x1er],1))
            x2dr = self.rnn_d2(x2d,[x2dr] if t > 0 else None)[0]

            x = self.conv_out(self.up3(x2dr))
            out_imgs.append(x)

        out_imgs = torch.stack(out_imgs,1)
        return out_imgs


class ForegroundBackgroundModel(nn.Module):

    def __init__(self, spatial_size,config):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.cat_poke_img = config["poke_and_img"]
        self.zeroflow_baseline = config["zeroflow_baseline"]
        self.variational = config["variational"] if "variational" in config else False

        foreground_background_div = config["foreground_background_div"]
        assert foreground_background_div >= 1.

        nf_first_shape_enc = int(max(32, config["nf_deep"] / (2 ** self.n_stages)))

        if self.variational:
            self.shape_enc = VariationalSkipConnectionEncoderFGBG(nf_in=3,nf_max=config["nf_deep"],n_stages=self.n_stages, n_skip_stages=self.n_stages,
                                                                  nf_first=nf_first_shape_enc)
        else:
            self.shape_enc = SkipConnectionEncoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages,
                                               n_skip_stages=self.n_stages, nf_first=nf_first_shape_enc,
                                               fg_bg=True, div=foreground_background_div)

        self.dynamics_enc = Encoder(nf_in=5 if self.cat_poke_img else 2, nf_max=config["nf_deep"], n_stages=self.n_stages)
        hidden_sizes = [int(config["nf_deep"]/foreground_background_div)] + [int(d/foreground_background_div) for d in self.shape_enc.depths]
        ups = [False] + self.shape_enc.downs
        self.fusion_block = ConvGRU(input_size=config["nf_deep"], hidden_sizes=hidden_sizes, kernel_sizes=3,
                                    n_layers=self.n_stages + 1, upsampling=ups)

        self.dec = SkipConnectionDecoder(nf_in=config["nf_deep"], in_channels=self.shape_enc.depths,
                                         n_skip_stages=self.n_stages, disentanglement=False)

    def forward(self,fg_img,bg_img,poke,len):

        x = fg_img

        mus = logstds = None

        if len > 0:

            if self.zeroflow_baseline:
                poke = torch.zeros_like(poke)

            if self.cat_poke_img:
                poke = torch.cat([poke, x], dim=1)

            imgs = []
            sigmas_fg = []
            sigmas_bg = []
            # infer dynamics input
            delta = self.dynamics_enc(poke)[0]
            # only first time shape encoding
            sigma_n = self.shape_enc(x)[0]
            sigma_bg = self.shape_enc(bg_img)[1]

            for n in range(len):
                # apply fusion block: input is delta, hidden states are the sigma_n
                sigma_n.reverse()
                sigma_n = self.fusion_block(delta, sigma_n)
                sigma_n.reverse()

                sigma_cat = [torch.cat([sfg,sbg],dim=1) for sfg,sbg in zip(sigma_n,sigma_bg)]
                x = self.dec(sigma_cat, None, del_shape=True)
                imgs.append(x)
                # output foreground representation
                sigmas_fg.append(sigma_n)
                # out
                sigmas_bg.append(sigma_bg)
            imgs = torch.stack(imgs, dim=1)
            sigmas_fg[-1].reverse()
        else:
            if self.variational:
                sigmas_fg, sigmas_bg1, mus, logstds = self.shape_enc(x)
                _, sigmas_bg2, *_ = self.shape_enc(bg_img)
            else:
                sigmas_fg, sigmas_bg1 = self.shape_enc(x)
                _, sigmas_bg2 = self.shape_enc(bg_img)


            sigmas_bg = (sigmas_bg1,sigmas_bg2)
            sigmas = [torch.cat([sfg,sbg],dim=1) for sfg,sbg in zip(sigmas_fg,sigmas_bg2)]
            imgs = self.dec(sigmas, None, del_shape=True)
            sigmas_fg.reverse()

        return imgs, sigmas_fg, sigmas_bg, mus, logstds


class SkipSequenceModel(nn.Module):
    def __init__(self,spatial_size,config,n_no_motion=None):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.min_spatial_size = config["min_spatial_size"] if "min_spatial_size" in config else 16
        self.n_stages = int(np.log2(spatial_size[0] // self.min_spatial_size))
        print(f"number of stages in model is {self.n_stages}")
        self.disentanglement = config["disentanglement"] if "disentanglement" in config else False
        self.cat_poke_img = config["poke_and_img"]
        self.zeroflow_baseline = config["zeroflow_baseline"]
        self.poke_every_t = config["poke_every_t"] if "poke_every_t" in config else True
        use_spectral_norm = config["spectnorm_decoder"] if "spectnorm_decoder" in config else False
        self.reparamterize = config["reparameterize_poke"] if "reparameterize_poke" in config else False
        self.norm_layer = config["norm_layer"] if "norm_layer" in config else "in"
        self.layers = config["layers"] if "layers" in config and len(config["layers"])>0 else None
        self.poke_scale_mode = config["poke_scale"] if "poke_scale" in config else False
        # self.n_no_motion = n_no_motion
        # if self.poke_scale_mode:
        #     assert self.n_no_motion is not None
        # self.multiscale_fusion_block = config["multiscale_dynamics"]
        # default is dynamics model
        # if self.disentanglement:
        #     self.appearance_enc = Encoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, prepare_adain=True,
        #                                   resnet_down=config["resnet_down"] if "resnet_down" in config else False)
        # n_skip_stages = min(config["n_skip_stages"], self.n_stages) if "n_skip_stages" in config else self.n_stages
        nf_first_shape_enc = int(max(32, config["nf_deep"] / (2 ** self.n_stages)))
        self.shape_enc = SkipConnectionEncoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, n_skip_stages=self.n_stages
                                               ,nf_first=nf_first_shape_enc, norm_layer=self.norm_layer,layers=self.layers)

        self.dynamics_enc = Encoder(nf_in=5 if self.cat_poke_img else 2, nf_max=config["nf_deep"], n_stages=self.n_stages,
                                    variational=self.reparamterize, norm_layer=self.norm_layer, layers=self.layers)
        hidden_sizes = [config["nf_deep"]]+self.shape_enc.depths
        ups = [False] + self.shape_enc.downs
        self.fusion_block = ConvGRU(input_size=config["nf_deep"], hidden_sizes=hidden_sizes, kernel_sizes=3, n_layers=self.n_stages+1 if self.layers is None else len(self.layers)+1,
                                    upsampling=ups,)

        self.dec = SkipConnectionDecoder(nf_in=config["nf_deep"], in_channels=self.shape_enc.depths, n_skip_stages=self.n_stages if self.layers is None else len(self.layers),
                                         disentanglement=False, spectral_norm=use_spectral_norm, norm_layer=self.norm_layer, layers=self.layers)

    def forward(self, app_img, shape_img, poke, len, poke_linear=False,delta_scaling = None, n_zero_frames=0, invert_poke=False, poke_jump=False):
        # if self.disentanglement:
        #     alpha, *_ = self.appearance_enc(app_img)
        # else:
        #     alpha = None
        # sigma = self.shape_enc(shape_img)


        x = shape_img

        if len > 0:

            if self.zeroflow_baseline:
                poke = torch.zeros_like(poke)

            if self.cat_poke_img:
                poke = torch.cat([poke, app_img], dim=1)

            imgs = []
            sigmas_hat_out = []
            sigmas_out = []
            # infer dynamics input
            if self.reparamterize:
                delta, mu, _  = self.dynamics_enc(poke)
            else:
                delta = self.dynamics_enc(poke)[0]
                sigmas_out = delta
            # only first time shape encoding
            #if self.poke_scale_mode and not poke_linear:
            sigma_n = self.shape_enc(x)

            for n in range(len):
                # apply fusion block: input is delta, hidden states are the sigma_n
                sigma_n.reverse()
                if self.poke_scale_mode:
                    if poke_linear:
                        if invert_poke:
                            delta_in = delta *  (1 - float(n) / int(len/2)) if n < int(len/2) else delta * (float(n - int(len/2)) / int(math.ceil(float(len)/2)) - 1)
                        else:
                            delta_in = (1 - float(n) / (len-n_zero_frames)) * delta if n <= len - n_zero_frames else torch.zeros_like(delta)
                    else:
                        delta_in = delta_scaling[n] * delta_in

                else:
                    if poke_jump:
                        delta_in = delta if n < len -n_zero_frames else torch.zeros_like(delta)
                    else:
                        delta_in = delta if self.poke_every_t else (delta if n == 0 else torch.zeros_like(delta))
                sigma_n = self.fusion_block(delta_in, sigma_n)
                sigma_n.reverse()

                x = self.dec(sigma_n, [], del_shape=False)
                imgs.append(x)
                sigmas_hat_out.append(sigma_n)
            imgs = torch.stack(imgs, dim=1)
            sigmas_hat_out[-1].reverse()
        else:
            sigmas = self.shape_enc(x)
            sigmas_out = sigmas
            sigmas_hat_out = None
            imgs = self.dec(sigmas, [], del_shape=False)
            sigmas_out.reverse()

        return imgs, sigmas_out, sigmas_hat_out, []


class SingleScaleBaseline(nn.Module):
    def __init__(self, spatial_size, config, n_no_motion=None):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.disentanglement = config["disentanglement"] if "disentanglement" in config else False
        self.cat_poke_img = config["poke_and_img"]
        self.zeroflow_baseline = config["zeroflow_baseline"]
        self.poke_scale_mode = config["poke_scale"] if "poke_scale" in config else False
        self.poke_every_t = config["poke_every_t"] if "poke_every_t" in config else True
        self.n_no_motion = n_no_motion
        if self.poke_scale_mode:
            assert self.n_no_motion is not None

        print("Initialize SingleScaleBaseline")
        # self.multiscale_fusion_block = config["multiscale_dynamics"]
        # default is dynamics model
        if self.disentanglement:
            self.appearance_enc = Encoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, prepare_adain=True,
                                          resnet_down=config["resnet_down"] if "resnet_down" in config else False)

        self.shape_enc = SkipConnectionEncoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, n_skip_stages=0)

        self.dynamics_enc = Encoder(nf_in=5 if self.cat_poke_img else 2, nf_max=config["nf_deep"], n_stages=self.n_stages)
        self.n_gru_layers = 3
        self.fusion_block = ConvGRU(input_size=config["nf_deep"], hidden_sizes=config["nf_deep"], kernel_sizes=3, n_layers=config["n_gru_layers"])

        self.dec = SkipConnectionDecoder(nf_in=config["nf_deep"], in_channels=self.shape_enc.depths, n_skip_stages=0, disentanglement=self.disentanglement)

    def forward(self, app_img, shape_img, poke, len, poke_linear=False,delta_scaling = None, n_zero_frames=0, invert_poke=False,poke_jump=False):
        if self.disentanglement:
            alpha, *_ = self.appearance_enc(app_img)
        else:
            alpha = None
        # sigma = self.shape_enc(shape_img)
        if self.zeroflow_baseline:
            poke = torch.zeros_like(poke)
        if self.cat_poke_img:
            poke = torch.cat([poke, app_img], dim=1)

        x = shape_img

        if len > 0:
            imgs = []
            sigmas_hat_out = []
            sigmas_out = []
            # infer dynamics input
            delta = self.dynamics_enc(poke)[0]
            sigma_n = self.shape_enc(x)[0]
            sigma_n = torch.stack([sigma_n] * self.n_gru_layers)

            for n in range(len):
                # delta scaling
                delta_in = delta if self.poke_every_t else (delta if n == 0 else torch.zeros_like(delta))
                if self.poke_scale_mode:
                    if poke_linear:
                        if invert_poke:
                            delta_in = delta * (1 - float(n) / int(len / 2)) if n < int(len / 2) else delta * (float(n - int(len / 2)) / int(math.ceil(float(len) / 2)) - 1)
                        else:
                            delta_in = (1 - float(n) / (len - n_zero_frames)) * delta if n <= len - n_zero_frames else torch.zeros_like(delta)
                    else:
                        delta_in = delta_scaling[n] * delta_in

                # apply fusion block
                sigma_n = self.fusion_block(delta_in, sigma_n)

                # residual connection
                sigma_n1 = sigma_n[-1]

                x = self.dec([sigma_n1], alpha, del_shape=False)
                imgs.append(x)
                sigmas_hat_out.append(sigma_n1)
            #sigmas_hat_out = torch.stack(sigmas_hat_out)
            imgs = torch.stack(imgs, dim=1)
        else:
            sigmas = self.shape_enc(x)
            sigmas_out = sigmas[-1]
            sigmas_hat_out = None
            imgs = self.dec(sigmas, alpha, del_shape=False)

        return imgs, sigmas_out, sigmas_hat_out, alpha



class ResidualSequenceBaseline(nn.Module):
    def __init__(self,spatial_size,config):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.disentanglement = config["disentanglement"] if "disentanglement" in config else False
        self.cat_poke_img = config["poke_and_img"]
        self.zeroflow_baseline = config["zeroflow_baseline"]
        #self.multiscale_fusion_block = config["multiscale_dynamics"]
        # default is dynamics model
        if self.disentanglement:
            self.appearance_enc = Encoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, prepare_adain=True,
                                          resnet_down=config["resnet_down"] if "resnet_down" in config else False)

        self.shape_enc = SkipConnectionEncoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, n_skip_stages=0)


        self.dynamics_enc = Encoder(nf_in=5 if self.cat_poke_img else 2, nf_max=config["nf_deep"], n_stages=self.n_stages)
        self.n_gru_layers = 3
        self.fusion_block = ConvGRU(input_size=config["nf_deep"], hidden_sizes=config["nf_deep"],kernel_sizes=3, n_layers=3)

        self.dec = SkipConnectionDecoder(nf_in=config["nf_deep"], in_channels=self.shape_enc.depths, n_skip_stages=0, disentanglement=self.disentanglement)


    def forward(self,app_img,shape_img,poke,len):
        if self.disentanglement:
            alpha, *_ = self.appearance_enc(app_img)
        else:
            alpha = None
        #sigma = self.shape_enc(shape_img)
        if self.zeroflow_baseline:
            poke = torch.zeros_like(poke)
        if self.cat_poke_img:
            poke = torch.cat([poke,app_img],dim=1)

        x = shape_img

        if len>0:
            imgs = []
            sigmas_hat_out = []
            sigmas_out = []
            # infer dynamics input
            delta = self.dynamics_enc(poke)[0]
            delta = torch.stack([delta]*self.n_gru_layers)

            for n in range(len):
                # shape encoding
                sigma_n = self.shape_enc(x)[0]

                # apply fusion block
                delta = self.fusion_block(sigma_n,delta)

                # residual connection
                sigma_n1 = sigma_n + delta[-1]

                x = self.dec([sigma_n1],alpha)
                imgs.append(x)
                sigmas_hat_out.append(sigma_n1)
            sigmas_hat_out = torch.stack(sigmas_hat_out,)
            imgs = torch.stack(imgs,dim=1)
        else:
            sigmas = self.shape_enc(x)
            sigmas_out = sigmas[-1]
            sigmas_hat_out = None
            imgs = self.dec(sigmas,alpha,del_shape=False)

        return imgs, sigmas_out, sigmas_hat_out, alpha

class DynamicSkipModel(nn.Module):
    def __init__(self, spatial_size,config):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.disentanglement = config["disentanglement"] if "disentanglement" in config else False
        self.cat_poke_img = config["poke_and_img"]
        self.zeroflow_baseline = config["zeroflow_baseline"]
        self.multiscale_fusion_block = config["multiscale_dynamics"]
        # default is dynamics model
        if self.disentanglement:
            self.appearance_enc = Encoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, prepare_adain=True,
                                          resnet_down=config["resnet_down"] if "resnet_down" in config else False)
        n_skip_stages = min(config["n_skip_stages"],self.n_stages) if "n_skip_stages" in config else self.n_stages
        self.shape_enc = SkipConnectionEncoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages,n_skip_stages=n_skip_stages)

        if config["multiscale_dynamics"]:
            self.dynamics_enc = SkipConnectionEncoder(nf_in=5 if self.cat_poke_img else 2, nf_max=config["nf_deep"], n_stages=self.n_stages, n_skip_stages=n_skip_stages)
            self.fusion_block = FusionBlockMultiscale(nf_in=config["nf_deep"],nfs=self.shape_enc.depths,n_blocks=config["n_blocks"])
        else:
            self.dynamics_enc = Encoder(nf_in=5 if self.cat_poke_img else 2, nf_max=config["nf_deep"], n_stages=self.n_stages)
            self.fusion_block = LearnedFusionBlock(nf=config["nf_deep"], n_blocks=config["n_blocks"])


        self.dec = SkipConnectionDecoder(nf_in=config["nf_deep"],in_channels=self.shape_enc.depths, n_skip_stages=n_skip_stages,disentanglement=self.disentanglement)

    def forward(self,app_img,shape_img,poke, apply_dynamics = False):
        if self.disentanglement:
            alpha, *_ = self.appearance_enc(app_img)
        else:
            alpha = None
        sigma = self.shape_enc(shape_img)
        if self.zeroflow_baseline:
            poke = torch.zeros_like(poke)
        if self.cat_poke_img:
            poke = torch.cat([poke,app_img],dim=1)
        delta = self.dynamics_enc(poke) if self.multiscale_fusion_block else self.dynamics_enc(poke)[0]


        if apply_dynamics:
            sigma_hat = self.fusion_block(sigma if self.multiscale_fusion_block else sigma.pop(), delta)
            if self.multiscale_fusion_block:
                sigma_in_dec = sigma_hat
            else:
                sigma_in_dec = sigma + [sigma_hat]
            sigma_out = sigma_hat
        else:
            sigma_out = sigma if self.multiscale_fusion_block else sigma[-1]
        img = self.dec(sigma_in_dec if apply_dynamics else sigma,alpha,del_shape=False)

        return img, sigma_out, alpha

class DisentangledModelWithoutDynamics(nn.Module):
    def __init__(self, spatial_size,config):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.adain = config["adain"]
        # default is dynamics model
        self.latent_fusion = config["latent_fusion"] if "latent_fusion" in config else None
        self.appearance_enc = Encoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, prepare_adain=self.adain,
                                      resnet_down=config["resnet_down"] if "resnet_down" in config else False)
        self.shape_enc = Encoder(nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages, variational=config["ib_shape"])

        if self.adain:
            self.dec = AdaINDecoderDisentangled(nf_in=config["nf_deep"], n_stages=self.n_stages, latent_fusion=self.latent_fusion, nf_in_bn=self.appearance_enc.nf_in_bn)
        else:
            self.dec = DecoderEntangled(nf_in=2 * config["nf_deep"],n_stages=self.n_stages)

    def forward(self,app_img,shape_img):
        # appearance representation
        alpha, alpha_spatial, *_ = self.appearance_enc(app_img)
        # shape representation
        sigma, shape_mean, shape_logstd = self.shape_enc(shape_img)
        # decode
        img = self.dec(alpha, sigma, alpha_spatial)

        return img, alpha, sigma, shape_mean,shape_logstd

class BasicDisentangledModel(nn.Module):
    def __init__(self, spatial_size,config):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.zero_flow_baseline = config["zero_flow_baseline"]
        self.adain = config["adain"]
        self.cat_poke_img = config["poke_and_img"]
        # default is dynamics model
        self.latent_fusion = config["latent_fusion"] if "latent_fusion" in config else None



        self.appearance_enc = Encoder(nf_in=3,nf_max = config["nf_deep"], n_stages=self.n_stages, prepare_adain=self.adain,
                                      resnet_down=config["resnet_down"] if "resnet_down" in config else False)
        self.shape_enc = Encoder(nf_in=3, nf_max = config["nf_deep"],n_stages=self.n_stages, variational=config["ib_shape"])

        self.dynamics_enc = Encoder(nf_in=5 if self.cat_poke_img else 2, nf_max=config["nf_deep"], n_stages=self.n_stages)
        self.fusion_block = LearnedFusionBlock(nf=config["nf_deep"],n_blocks=config["n_blocks"])

        if self.adain:
            self.dec = AdaINDecoderDisentangled(nf_in=config["nf_deep"],n_stages=self.n_stages,latent_fusion=self.latent_fusion,nf_in_bn=self.appearance_enc.nf_in_bn)
        else:
            self.dec = DecoderEntangled(nf_in=2 * config["nf_deep"],n_stages=self.n_stages)

    def forward(self, app_img, shape_img, poke, apply_dynamics = False):
        # appearance representation
        alpha, alpha_spatial, *_ = self.appearance_enc(app_img)
        # shape representation
        sigma, shape_mean, shape_logstd = self.shape_enc(shape_img)
        # dynamics representation
        if self.zero_flow_baseline:
            poke = torch.zeros_like(poke)
        if self.cat_poke_img:
            poke = torch.cat([poke,app_img],dim=1)
        delta, *_ = self.dynamics_enc(poke)
        if self.zero_flow_baseline:
            delta = torch.zeros_like(delta)
        # apply dynamics to shape represenation
        sigma_hat = self.fusion_block(sigma,delta)
            # decode
        if apply_dynamics:
            img = self.dec(alpha,sigma_hat,alpha_spatial)
        else:
            img = self.dec(alpha,sigma,alpha_spatial)


        return img, sigma, sigma_hat, alpha, delta, shape_mean, shape_logstd

class LearnedFusionBlock(nn.Module):
    def __init__(self,nf,n_blocks):
        super().__init__()
        assert n_blocks >= 1
        blocks = [ResBlock(2*nf,nf)]

        for i in range(1,n_blocks):
            blocks.append(ResBlock(nf,nf))

        self.model = nn.Sequential(*blocks)
    def forward(self,sigma,delta):
        x = torch.cat([sigma,delta],dim=1)
        x = self.model(x)
        return x



class BasicModel(nn.Module):
    def __init__(self, spatial_size, config):
        super().__init__()
        # number of downsampling layers; always such that spatial bottleneck size is 16x16
        self.n_stages = int(np.log2(spatial_size[0] // 16))
        self.zero_flow_baseline = config["zero_flow_baseline"]
        self.adain = config["adain"]
        self.obj_enc = Encoder(
            nf_in=3, nf_max=config["nf_deep"], n_stages=self.n_stages,variational=config["variational"])
        self.flow_enc = Encoder(
            nf_in=2,
            nf_max=config["nf_deep"],
            n_stages=self.n_stages,
            prepare_adain=self.adain
        )
        if self.adain:
            self.dec = AdaINDecoderEntangled(
                nf_in=config["nf_deep"], n_stages=self.n_stages,latent_fusion=config["latent_fusion"]
            )
        else:
            self.dec = DecoderEntangled(nf_in=2*config["nf_deep"],n_stages=self.n_stages)

    def forward(self, image, flow,sample_prior=False):

        # get object code and variational paramezers if model is variational
        object_code, mean, logstd = self.obj_enc(image,sample_prior)

        # get dynamics codes
        dynamics_code1, dynamics_code2, _ = self.flow_enc(flow)
        if self.zero_flow_baseline:
            # this is without flow usage, to measure the impacts of the flow as adain input
            dynamics_code1 = torch.zeros_like(dynamics_code1)
            dynamics_code2 = torch.zeros_like(dynamics_code2)
        # decode
        if self.adain:
            img = self.dec(object_code, dynamics_code1,dynamics_code2)
        else:
            img = self.dec(object_code,dynamics_code1)

        return img, object_code, dynamics_code1, mean, logstd

class VariationalSkipConnectionEncoderFGBG(nn.Module):

    def __init__(self,nf_in,nf_max, n_stages, n_skip_stages, act = "relu", nf_first=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.n_stages = n_stages
        self.depths = []
        self.downs = []
        if nf_first is None:
            nf = 64
        else:
            nf = nf_first

        # required

        self.blocks.append(
            NormConv2d(
                nf_in, int(1.5 * nf), 3, 2, padding=1
            )
        )
        self.n_skip_stages = n_skip_stages
        self.depths.append(nf)
        for n in range(self.n_stages - 1):
            self.blocks.append(
                NormConv2d(
                    nf,
                    min(nf * 3, int(1.5*nf_max)),
                    3,
                    2,
                    padding=1,
                )
            )
            nf = min(nf * 2, nf_max)
            self.depths.insert(0, nf)
            self.downs.insert(0, True)

        self.bottleneck = ResBlock(nf, int(1.5 * nf_max), activation=act, stride=1)
        self.downs.insert(0, False)
        self.squash = nn.Sigmoid()

    def _reparameterize(self,codes):

        mu = codes[:,:int(codes.shape[1]/2)]
        logstd = codes[:,int(codes.shape[1]/2):]
        logstd = self.squash(logstd)

        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu, mu, logstd

    def forward(self,x):

        out_fg = []
        out_bg = []
        out_logstd = []
        out_mu = []

        for i in range(self.n_stages):
            x = self.blocks[i](x)
            if i >= self.n_stages - self.n_skip_stages:
                act_div = int(x.shape[1] * 2. / 3.)
                sample, mu, logstd = self._reparameterize(x[:,:act_div])
                out_fg.append(sample)
                bg = x[:,act_div:]
                out_bg.append(bg)
                out_mu.append(mu)
                out_logstd.append(logstd)

                x = torch.cat([mu,bg], dim=1)

        x = self.bottleneck(x)


        act_div = int(x.shape[1] * 2. / 3.)
        sample, mu, logstd = self._reparameterize(x[:, :act_div])
        out_fg.append(sample)
        bg = x[:, act_div:]
        out_bg.append(bg)
        out_mu.append(mu)
        out_logstd.append(logstd)

        return out_fg, out_bg, out_mu, out_logstd


class SkipConnectionEncoder(nn.Module):
    def __init__(self,nf_in,nf_max, n_stages, n_skip_stages, act = "elu", nf_first=None, fg_bg = False, div= None, norm_layer="in", layers=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.n_stages = n_stages if layers is None else len(layers)
        self.depths = []
        self.downs = []
        if nf_first is None:
            nf = 32
        else:
            nf = nf_first

        if layers is not None:
            nf = layers[0]

        self.fg_bg = fg_bg
        if self.fg_bg:
            assert div is not None
            self.div = div

        self.blocks.append(
            Conv2dBlock(
                nf_in, nf, 3, 2, norm=norm_layer, activation=act, padding=1
            )
        )
        self.n_skip_stages = n_skip_stages if layers is None else len(layers)
        self.depths.append(nf)
        for n in range(self.n_stages - 1):
            self.blocks.append(
                Conv2dBlock(
                    nf,
                    min(nf * 2, nf_max) if layers is None else layers[n+1],
                    3,
                    2,
                    norm=norm_layer,
                    activation=act,
                    padding=1,
                )
            )
            nf = min(nf * 2, nf_max) if layers is None else layers[n+1]
            self.depths.insert(0,nf)
            self.downs.insert(0,True)

        self.bottleneck = ResBlock(nf, nf_max, activation=act, stride=1,norm=norm_layer)
        self.downs.insert(0,False)


    def forward(self,x):
        if self.fg_bg:
            out_fg = []
            out_bg = []
        else:
            out = []

        for i in range(self.n_stages):
            x = self.blocks[i](x)
            if i >= self.n_stages - self.n_skip_stages:
                if self.fg_bg:
                    act_div = int(x.shape[1] / self.div)
                    out_fg.append(x[:,:act_div])
                    out_bg.append(x[:,act_div:])
                else:
                    out.append(x)

        x = self.bottleneck(x)

        if self.fg_bg:
            act_div = int(x.shape[1] / self.div)
            out_fg.append(x[:,:act_div])
            out_bg.append(x[:,act_div:])
            return out_fg, out_bg
        else:
            out.append(x)
            return out


class Encoder(nn.Module):
    def     __init__(self, nf_in, nf_max, n_stages, prepare_adain=False, variational=False, resnet_down=False, norm_layer = "in", layers=None):
        super().__init__()

        self.prepare_adain = prepare_adain
        self.variational = variational
        if self.prepare_adain:
            assert not self.variational, "Encoder should not be variational if adain is prepared"

        if self.prepare_adain:
            self.final_linear = nn.Linear(nf_max, nf_max)

        act = "elu" #if self.variational else "relu"

        blocks = []
        bottleneck = []
        nf = 32 if layers is None else layers[0]
        blocks.append(
            Conv2dBlock(
                nf_in, nf, 3, 2, norm=norm_layer, activation=act, padding=1
            )
        )
        n_stages = n_stages if layers is None else len(layers)
        for n in range(n_stages - 1):
            blocks.append(
                Conv2dBlock(
                    nf,
                    min(nf * 2, nf_max) if layers is None else layers[n+1],
                    3,
                    2,
                    norm=norm_layer,
                    activation=act,
                    padding=1,
                )
            )
            nf = min(nf * 2, nf_max) if layers is None else layers[n+1]

        self.resnet_down = resnet_down and self.prepare_adain
        self.nf_in_bn = nf
        bottleneck.append(ResBlock(nf, nf_max,activation=act, stride=2 if self.resnet_down else 1, norm=norm_layer))
        if layers is None:
            bottleneck.append(ResBlock(nf_max, nf_max,activation=act, stride=2 if self.resnet_down else 1, norm=norm_layer))

        if self.resnet_down:
            self.make_vector = Conv2dBlock(nf_max,nf_max,4,1,0)


        if self.variational:
            self.make_mu = NormConv2d(nf_max,nf_max,3, padding=1)
            self.make_sigma = NormConv2d(nf_max,nf_max,3, padding=1)
            self.squash = nn.Sigmoid()

        self.model = nn.Sequential(*blocks)
        self.bottleneck = nn.Sequential(*bottleneck)

    def forward(self, input, sample_prior=False):
        out = self.model(input)
        mean = out
        out = self.bottleneck(out)
        logstd = None
        if self.prepare_adain:
            # mean is a false name here, this is the raw channels of the conv model
            # mean = out
            if self.resnet_down:
                # in this case, mean has spatial_size 4x4
                out = self.make_vector(out).squeeze(-1).squeeze(-1)
            else:
                out = F.avg_pool2d(out, out.size(2), padding=0)
                out = out.squeeze(-1).squeeze(-1)
            # no activation for the first trial, as relu would not allow for values < 0
            out = self.final_linear(out)
        elif self.variational:
            mean = self.make_mu(out)
            # normalize sigma in between
            logstd = self.squash(self.make_sigma(out))
            if sample_prior:
                out = torch.randn_like(mean)
            else:
                out = self.reparametrize(mean,logstd)

        return out, mean, logstd

    def reparametrize(self,mean,logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std) + mean

class AdaINDecoderEntangled(nn.Module):
    """
    We sample up from spatial resolution 16x16, given quadratic images
    """

    def __init__(self, nf_in, n_stages,latent_fusion):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.affines = nn.ModuleList()
        self.n_stages = n_stages
        self.latent_fusion = latent_fusion

        # results latent fusion results in a deeper model
        nf = nf_in * 2 if self.latent_fusion else nf_in
        self.in_block = ResBlock(nf, nf)



        for n in range(self.n_stages):
            self.affines.append(AdaINLinear(nf_in, int(nf // 2)))
            # upsampling adain layers
            self.blocks.append(
                ResBlock(nf, int(nf // 2), norm="adain", upsampling=True)
            )
            nf = int(nf // 2)

        self.out_conv = Conv2dBlock(
            nf, 3, 3, 1, padding=1, norm="none", activation="tanh"
        )

    def forward(self, object_code, dynamics_linear,dynamics_spatial):
        if self.latent_fusion:
            in_code = torch.cat([object_code,dynamics_spatial],dim=1)
        else:
            in_code = object_code
        x = self.in_block(in_code)
        for n in range(self.n_stages):
            adain_params = self.affines[n](dynamics_linear)
            x = self.blocks[n](x, adain_params)

        x = self.out_conv(x)
        return x

class AdaINDecoderDisentangled(nn.Module):
    def __init__(self,nf_in, n_stages, latent_fusion = None, nf_in_bn = 0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.affines = nn.ModuleList()
        self.n_stages = n_stages
        self.latent_fusion = False if latent_fusion is None else latent_fusion
        if self.latent_fusion:
            assert nf_in_bn > 0

        # results latent fusion results in a deeper model
        nf = nf_in + nf_in_bn if self.latent_fusion else nf_in
        # self.bottleneck_adain = bottleneck_adain
        self.in_block = ResBlock(nf, nf,)

        for n in range(self.n_stages):
            self.affines.append(AdaINLinear(nf_in, int(nf // 2)))
            # upsampling adain layers
            self.blocks.append(
                ResBlock(nf, int(nf // 2), norm="adain", upsampling=True)
            )
            nf = int(nf // 2)

        self.out_conv = Conv2dBlock(
            nf, 3, 3, 1, padding=1, norm="none", activation="tanh"
        )

    def forward(self,alpha,sigma,alpha_spatial=None):
        if self.latent_fusion:
            assert alpha_spatial is not None
            in_code = torch.cat([sigma, alpha_spatial],dim=1)
        else:
            in_code = sigma
        x = self.in_block(in_code)
        for n in range(self.n_stages):
            adain_params = self.affines[n](alpha)
            x = self.blocks[n](x, adain_params)

        x = self.out_conv(x)
        return x


class SkipConnectionDecoder(nn.Module):

    def __init__(self,nf_in, in_channels, n_skip_stages, disentanglement=False, spectral_norm=False, norm_layer="in",layers=None):
        super().__init__()
        self.n_stages = len(in_channels)
        self.disentanglement = disentanglement
        self.n_skip_stages = n_skip_stages

        self.blocks = nn.ModuleList()
        if self.disentanglement:
            self.affines = nn.ModuleList()

        nf = nf_in
        self.in_block = ResBlock(nf,in_channels[0], snorm=spectral_norm, norm=norm_layer)

        for i,nf in enumerate(in_channels):
            if layers is None:
                n_out = int(nf // 2) if i < len(in_channels) - 1 else nf
            if self.disentanglement:
                self.affines.append(AdaINLinear(nf_in,n_out))
            nf_in_dec = 2 * nf if i < self.n_skip_stages else nf
            if layers is not None:
                nf_in_dec = 2 * nf
                n_out = in_channels[i+1] if i < len(in_channels) -1 else nf
            self.blocks.append(ResBlock(nf_in_dec, n_out , norm="adain" if self.disentanglement else norm_layer, upsampling=True,snorm=spectral_norm))

        self.out_conv = Conv2dBlock(nf,3,3,1,1,norm="none",activation="tanh")

    def forward(self,shape, appearance = None, del_shape=True):
        x = self.in_block(shape.pop() if del_shape else shape[-1])
        for n in range(self.n_stages):
            if n < self.n_skip_stages:
                x = torch.cat([x,shape.pop() if del_shape else shape[self.n_skip_stages-1-n]],1)
            if self.disentanglement:
                adain_params = self.affines[n](appearance)
                x = self.blocks[n](x,adain_params)
            else:
                x = self.blocks[n](x)

        if del_shape:
            assert not shape
        out = self.out_conv(x)
        return out
class DecoderEntangled(nn.Module):
    """
    We sample up from spatial resolution 16x16, given quadratic images
    """

    def __init__(self, nf_in, n_stages):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.n_stages = n_stages

        nf = nf_in
        self.in_block = ResBlock(nf, nf)

        for n in range(self.n_stages):
            self.blocks.append(
                ResBlock(nf, int(nf // 2), norm="in", upsampling=True)
            )
            nf = int(nf // 2)

        self.out_conv = Conv2dBlock(
            nf, 3, 3, 1, padding=1, norm="none", activation="tanh"
        )

    def forward(self, object_code, dynamics_code,*args):

        in_code = torch.cat([object_code,dynamics_code],dim=1)

        x = self.in_block(in_code)
        for n in range(self.n_stages):
            x = self.blocks[n](x)

        x = self.out_conv(x)
        return x

class FusionBlockMultiscale(nn.Module):

    def __init__(self,nf_in,nfs,n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.n_stages = len(nfs) + 1

        nf = nf_in
        for n in range(self.n_stages):
            self.blocks.append(LearnedFusionBlock(nf,n_blocks))
            if n < len(nfs):
                nf = nfs[n]

    def forward(self,sigmas,deltas):
        out = []
        for i,n in enumerate(range(len(sigmas)-1,-1,-1))    :
            out.insert(0,self.blocks[i](sigmas[n],deltas[n]))

        return out
