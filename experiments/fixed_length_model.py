import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Average, MetricUsage
import numpy as np
import wandb
from functools import partial
from lpips import LPIPS
from tqdm import tqdm


from experiments.experiment import Experiment
from data import get_dataset
from data.samplers import FixedLengthSampler
from models.latent_flow_net import SingleScaleBaseline,SkipSequenceModel
from models.discriminator import GANTrainer
from utils.losses import PerceptualVGG,vgg_loss_agg,DynamicsLoss, style_loss
from utils.testing import make_flow_grid, make_img_grid, make_video, make_plot
from utils.metrics import metric_fid, FIDInceptionModel, metric_lpips, psnr_lightning, ssim_lightning
from utils.general import linear_var, get_member, get_patches



class FixedLengthModel(Experiment):
    def __init__(self, config, dirs, device):
        super().__init__(config, dirs, device)
        self.datakeys = ["images","poke"]
        if self.config["architecture"]["disentanglement"]:
            self.datakeys.append("img_aT")
            self.datakeys.append("app_img_random")

        # used for efficient metrics computation
        self.fid_feats_real_per_frame = {}
        self.fid_feats_fake_per_frame = {}
        self.psnrs = {"t": [], "tk": [], "pl" : []}
        self.ssims = {"t": [], "tk": [], "pl" : []}
        self.lpips = {"t": [], "tk": []}

        self.use_gan = self.config["gan"]["use"]
        self.use_temp_disc = self.config["gan_temp"]["use"]
        if self.use_temp_disc:
            if not self.config["gan_temp"]["patch_temp_disc"]:
                assert not self.config["gan_temp"]["conditional"]
        #self.pixel_decoder_loss = self.config["training"]["pixel_dynamics_weight"] > 0
        self.lr_dec_t = 0
        self.target_dev = None

        # metrics for each frame
        self.ssims_per_frame = {}
        self.lpips_per_frame = {}
        self.psnrs_per_frame = {}
        # self.ssims_per_frame_pl = {}
        # self.psnrs_per_frame_pl = {}
        self.lpips_avg = None

        self.custom_sampler = self.config["training"]["custom_sampler"] if "custom_sampler" in self.config["training"] else False

        self.poke_jump = self.config["training"]["poke_jump"] if "poke_jump" in self.config["training"] else False
        self.poke_scale_mode = self.config["architecture"]["poke_scale"] if "poke_scale" in self.config["architecture"] else False

        if self.poke_jump:
            assert not self.poke_scale_mode

    def __clear_metric_arrs(self):
        [self.psnrs[key].clear() for key in self.psnrs]
        [self.ssims[key].clear() for key in self.ssims]
        [self.lpips[key].clear() for key in self.lpips]
        self.lpips_per_frame = {}
        self.psnrs_per_frame = {}
        self.ssims_per_frame = {}
        self.fid_feats_real_per_frame = {}
        self.fid_feats_fake_per_frame = {}
        # self.ssims_per_frame_pl = {}
        # self.psnrs_per_frame_pl = {}

    def train(self):
        ########## checkpoints ##########
        if self.config["general"]["restart"] and not self.is_debug:
            mod_ckpt, op_ckpts = self._load_ckpt("reg_ckpt", single_opt=False)
            op_ckpt_dis = op_ckpts["optimizer_dis"]
            op_ckpt_dyn = op_ckpts["optimizer_dyn"]
        else:
            mod_ckpt = op_ckpt_dis = op_ckpt_dyn = None

        # get datasets for training and testing
        def w_init_fn(worker_id):
            return np.random.seed(np.random.get_state()[1][0] + worker_id)

        if not self.poke_scale_mode:
            del self.config["data"]["n_ref_frames"]

        dataset, transforms = get_dataset(config=self.config["data"])
        train_dataset = dataset(transforms, self.datakeys, self.config["data"], train=True)
        test_datakeys = self.datakeys + ["app_img_random"] if self.config["testing"]["eval_app_transfer"] and "app_img_random" not in self.datakeys else self.datakeys
        test_datakeys.append("flow")
        test_dataset = dataset(transforms, test_datakeys, self.config["data"], train=False)
        if self.custom_sampler:
            train_sampler = FixedLengthSampler(train_dataset, self.config["training"]["batch_size"],shuffle=True,
                                               weighting=train_dataset.obj_weighting,drop_last=True,zero_poke=True, zero_poke_amount=self.config["training"]["zeropoke_amount"])
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,num_workers=0 if self.is_debug else self.config["data"]["num_workers"],
                worker_init_fn=w_init_fn,)

            test_sampler = FixedLengthSampler(test_dataset, batch_size=self.config["training"]["batch_size"], shuffle=True,
                                              drop_last=True, weighting=test_dataset.obj_weighting,zero_poke=True,zero_poke_amount=self.config["training"]["zeropoke_amount"])
            test_loader = DataLoader(
                test_dataset,
                batch_sampler=test_sampler,
                num_workers=0 if self.is_debug else self.config["data"]["num_workers"],  #
                worker_init_fn=w_init_fn,
            )


            eval_sampler = FixedLengthSampler(test_dataset,batch_size=self.config["testing"]["test_batch_size"],shuffle=True,
                                          drop_last=True,weighting=test_dataset.obj_weighting,zero_poke=False)
            eval_loader = DataLoader(test_dataset,
                                     batch_sampler=eval_sampler,
                                     num_workers=0 if self.is_debug else self.config["data"]["num_workers"],
                                     worker_init_fn=w_init_fn,)
            self.logger.info("Using custom fixed length sampler.")
        else:
            self.logger.info("Using standard pytorch random sampler")
            train_sampler = RandomSampler(train_dataset)

            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=self.config["training"]["batch_size"],
                num_workers=0 if self.is_debug else self.config["data"]["num_workers"],
                worker_init_fn=w_init_fn,
                drop_last=True
            )
            test_sampler = RandomSampler(test_dataset,)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config["training"]["batch_size"],
                sampler=test_sampler,
                num_workers=0 if self.is_debug else self.config["data"]["num_workers"],
                worker_init_fn=w_init_fn,
                drop_last=True
            )

            # no zeropoke for evaluation as zeropoke is only to ensure no reaction when poking outside
            eval_sampler = SequentialSampler(test_dataset,)
            eval_loader = DataLoader(test_dataset,
                                     sampler=eval_sampler,
                                     batch_size=self.config["testing"]["test_batch_size"],
                                     num_workers=0 if self.is_debug else self.config["data"]["num_workers"],
                                     worker_init_fn=w_init_fn,
                                     drop_last=True)

        # define model
        self.logger.info(f"Load model...")

        #net_model = SkipSequenceModel if self.config["architecture"]["use_skip_model"] else ResidualSequenceBaseline
        net = SkipSequenceModel(spatial_size=self.config["data"]["spatial_size"],config=self.config["architecture"]) if self.config["architecture"]["use_skip_model"] else \
            SingleScaleBaseline(spatial_size=self.config["data"]["spatial_size"],
                        config=self.config["architecture"], )

        self.logger.info(
            f"Number of trainable parameters in model is {sum(p.numel() for p in net.parameters())}"
        )
        if self.config["general"]["restart"] and mod_ckpt is not None:
            self.logger.info("Load pretrained paramaters and resume training.")
            net.load_state_dict(mod_ckpt)

        if self.parallel:
            net = torch.nn.DataParallel(net, device_ids=self.all_devices)
        net.cuda(self.all_devices[0])
        self.logger.info("Model on gpu!")

        # log weights and gradients
        wandb.watch(net, log="all")

        # define optimizers
        # appearance and shape disentanglement
        dis_params = [{"params": get_member(net,"shape_enc").parameters(), "name": "shape_encoder"},
                      {"params": get_member(net,"dec").parameters(), "name": "decoder"}
            ]

        optimizer_dis = Adam(dis_params, lr=self.config["training"]["lr"])
        if self.config["general"]["restart"] and op_ckpt_dis is not None:
            self.logger.info("Load state_dict of optimizer.")
            optimizer_dis.load_state_dict(op_ckpt_dis)
        milestones = [int(self.config["training"]["n_epochs"] * t) for t in self.config["training"]["tau"]]
        scheduler_dis = torch.optim.lr_scheduler.MultiStepLR(optimizer_dis, milestones=milestones, gamma=self.config["training"]["lr_reduce"])
        # dynamics
        dyn_params = [{"params": get_member(net,"dynamics_enc").parameters(), "name": "dynamics_encoder", },
                      {"params": get_member(net,"fusion_block").parameters(), "name": "fusion_block",},]
        if self.config["training"]["decoder_update_tk"]:
            dyn_params.append({"params": get_member(net,"dec").parameters(), "name": "decoder"})

        if "singlestage" in self.config["training"] and self.config["training"]["singlestage"]:
            dyn_params.append({"params": get_member(net, "shape_enc").parameters(), "name": "shape_encoder"})
        optimizer_dyn = Adam(dyn_params, lr = self.config["training"]["lr"])

        if self.config["general"]["restart"] and op_ckpt_dyn is not None:
            self.logger.info("Load state_dict of optimizer.")
            optimizer_dyn.load_state_dict(op_ckpt_dyn)
        milestones = [int(self.config["training"]["n_epochs"] * t) for t in self.config["training"]["tau"]]
        scheduler_dyn = torch.optim.lr_scheduler.MultiStepLR(optimizer_dyn, milestones=milestones, gamma=self.config["training"]["lr_reduce"])

        # initialize disc if gan mode is enabled
        if self.use_gan:
            gan_trainer = GANTrainer(self.config, self._load_ckpt, self.logger,spatial_size=self.config["data"]["spatial_size"][0] ,
                                     parallel=self.parallel, devices=self.all_devices, debug=self.is_debug)
        if self.use_temp_disc:
            gan_trainer_temp = GANTrainer(self.config, self._load_ckpt,self.logger,spatial_size=self.config["data"]["spatial_size"][0],
                                          parallel=self.parallel,devices=self.all_devices, debug=self.is_debug,temporal=True, sequence_length=train_dataset.max_frames)
        # set start iteration and epoch in case model training is resumed
        start_it = 0
        start_epoch = 0
        n_epoch_train = self.config["training"]["n_epochs"]
        if self.config["general"]["restart"] and op_ckpts is not None:
            start_it = list(optimizer_dis.state_dict()["state"].values())[-1]["step"]
            start_epoch = int(np.floor(start_it / len(train_loader)))
            assert self.config["training"]["n_epochs"] > start_epoch
            n_epoch_train = self.config["training"]["n_epochs"] - start_epoch

        #
        lr_dec_rec = partial(linear_var,start_it=0,
            end_it=self.config["training"]["lr_dec_end_it"],
            start_val=self.config["training"]["lr"],
            end_val=self.config["training"]["lr_dec_end_val"],
            clip_min=0,
            clip_max=self.config["training"]["lr"],)

        self.lr_dec_t = lr_dec_rec(start_it)

        # losses
        self.logger.info("Load VGG")
        self.vgg = PerceptualVGG()
        if self.parallel:
            self.vgg = torch.nn.DataParallel(self.vgg,device_ids=self.all_devices)
        self.vgg.cuda(self.all_devices[0])
        self.logger.info("VGG on gpu")
        # from torchsummary import summary
        # summary(vgg.vgg,(3,224,224))




        self.logger.info("Initialize persistent losses")
        latent_dynamics_loss = DynamicsLoss(config=self.config["training"])
        self.logger.info("Finished initializing persistent losses.")
        def train_step(engine,batch):
            net.train()

            # prepare data
            weights=None
            loss_dis = 0
            out_dict = {}
            if train_dataset.flow_weights:
                poke = batch["poke"][0].cuda(self.all_devices[0])
                weights = batch["poke"][1].cuda(self.all_devices[0])
            else:
                poke = batch["poke"].cuda(self.all_devices[0])
            x_t = batch["images"][:, 0].cuda(self.all_devices[0])
            x_seq = batch["images"][:, 1:].cuda(self.all_devices[0])
            if self.config["architecture"]["disentanglement"]:
                shape_img = batch["img_aT"].cuda(self.all_devices[0])
                # apply style loss
                app_img_tr = batch["app_img_random"].cuda(self.all_devices[0])
                x_trans, *_ = net(app_img_tr,x_t,poke,len=0)
                loss_style = style_loss(self.vgg,app_img_tr,x_trans)
                loss_dis = self.config["training"]["style_loss_weight"] * loss_style
                out_dict.update({"style_loss": loss_style.item()})
            else:
                 shape_img = x_t

            x_t_hat_i, sigma_t, _ , alpha = net(x_seq[:,-1],shape_img,poke,len=0)
            n_ref_frames = self.config["data"]["n_ref_frames"] - 1 if self.poke_scale_mode else train_dataset.max_frames -1


            # static loss to obtain fixed image state space
            if "singlestage" not in self.config["training"] or not self.config["training"]["singlestage"]:
                loss_dis = loss_dis + vgg_loss_agg(self.vgg, x_t, x_t_hat_i)


                #optimize parameter of appearance, shape encoders and decoder
                optimizer_dis.zero_grad()
                loss_dis.backward()
                optimizer_dis.step()

                out_dict.update({"loss_dis" : loss_dis.item()})

            #optimize in alternating gradient descent as this results in equal results than training the static/dynamic model in two completely seperate stages
            # however, it performs significantly better than training both models jointly with a single optimizer step (see ablations or run the model with 'singlestage' set to true)

            # forward pass for training of dynamics part of  the model
            # dynamics losses
            seq_len = x_seq.shape[1]
            seq_rec, mu_delta, sigmas_hat, logstd_delta = net(x_t,shape_img,poke,len=seq_len,
                                                              poke_linear=self.poke_scale_mode,
                                                              n_zero_frames=seq_len-n_ref_frames-1, poke_jump=self.poke_jump)
            sigmas_gt = []
            ll_loss_dyn = []
            rec_imgs = []

            if weights is not None:
                seq_rec = get_patches(seq_rec,weights,self.config["data"],train_dataset.weight_value_flow, logger=self.logger)
                x_seq = get_patches(x_seq,weights,self.config["data"],train_dataset.weight_value_flow, logger=self.logger)



            for n in range(seq_len):
                x_hat_tn,s_tn,*_ = net(x_seq[:,n],x_seq[:,n],poke,len=0)
                sigmas_gt.append(s_tn)
                rec_imgs.append(x_hat_tn)
                w = 1. if n != n_ref_frames else self.config["training"]["target_weight"]
                ll_dyn_n =w  *  vgg_loss_agg(self.vgg,x_seq[:,n],seq_rec[:,n])
                ll_loss_dyn.append(ll_dyn_n)
            ll_loss_dyn = torch.stack(ll_loss_dyn,dim=0).mean()
            rec_imgs  = torch.stack(rec_imgs,1)

            #latent dynamics
            dyn_losses = []
            for s_tk,s_hat_tk in zip(sigmas_gt,sigmas_hat):
                dyn_losses.append(latent_dynamics_loss(s_hat_tk,s_tk,[]))
            latent_loss_dyn = torch.stack(dyn_losses).mean()
            loss_dyn = self.config["training"]["vgg_dyn_weight"] * ll_loss_dyn + self.config["training"]["latent_dynamics_weight"] * latent_loss_dyn


            if self.use_gan and engine.state.iteration >= self.config["gan"]["start_iteration"]:
                if self.config["gan"]["pixel_dynamics"]:
                    offsets = np.random.choice(np.arange(max(1,x_seq.shape[1]-train_dataset.max_frames)),size=x_seq.shape[0])
                    true_exmpls = torch.stack([seq[o:o+train_dataset.max_frames] for seq, o in zip(x_seq,offsets)],dim=0)
                    fake_exmpls = torch.stack([seq[o:o+train_dataset.max_frames] for seq, o in zip(seq_rec, offsets)], dim=0)
                    x_true = torch.cat([true_exmpls[:,1:],true_exmpls[:,:-1]],dim=2).reshape(-1,2*true_exmpls.shape[2],*true_exmpls.shape[3:])
                    x_fake = torch.cat([fake_exmpls[:, 1:], true_exmpls[:, :-1]], dim=2).reshape(-1, 2 * fake_exmpls.shape[2], *fake_exmpls.shape[3:])
                else:
                    true_exmpls = np.random.choice(np.arange(x_seq.shape[0]*x_seq.shape[1]),self.config["gan"]["n_examples"])
                    fake_exmpls = np.random.choice(np.arange(seq_rec.shape[0]*seq_rec.shape[1]), self.config["gan"]["n_examples"])
                    x_true = x_seq.view(-1,*x_seq.shape[2:])[true_exmpls]
                    x_fake = seq_rec.view(-1,*seq_rec.shape[2:])[fake_exmpls]

                disc_dict, loss_gen, loss_fmap = gan_trainer.train_step(x_true, x_fake)
                loss_dyn = loss_dyn + self.config["gan"]["gen_weight"] * loss_gen + self.config["gan"]["fmap_weight"] * loss_fmap

            if self.use_temp_disc and engine.state.iteration >= self.config["gan_temp"]["start_iteration"]:
                seq_len_act = x_seq.shape[1]
                offset = int(np.random.choice(np.arange(max(1,seq_len_act-train_dataset.max_frames)),1))
                # offset_fake = int(np.random.choice(np.arange(max(1,seq_len_act-seq_len_temp_disc)), 1))
                x_fake_tmp = seq_rec[:,offset:offset+train_dataset.max_frames].permute(0,2,1,3,4)
                x_true_tmp = x_seq[:, offset:offset+train_dataset.max_frames].permute(0,2,1,3,4)


                if self.config["gan_temp"]["conditional"]:
                    cond = get_patches(poke,weights,self.config["data"],test_dataset.weight_value_flow,self.logger) if test_dataset.flow_weights else poke
                else:
                    cond = None
                disc_dict_temp, loss_gen_temp, loss_fmap_temp = gan_trainer_temp.train_step(x_true_tmp,x_fake_tmp,cond)
                loss_dyn = loss_dyn + self.config["gan_temp"]["gen_weight"] * loss_gen_temp + self.config["gan_temp"]["fmap_weight"] * loss_fmap_temp

            # optimize parameters of dynamics part
            optimizer_dyn.zero_grad()
            loss_dyn.backward()
            optimizer_dyn.step()

            out_dict.update({"loss_dyn":loss_dyn.item() ,"vgg_loss_dyn" : ll_loss_dyn.item(), "latent_loss_dyn": latent_loss_dyn.item(), "lr_dec_t": self.lr_dec_t})
            if self.use_gan and engine.state.iteration >= self.config["gan"]["start_iteration"]:
                out_dict.update(disc_dict)
                out_dict.update({"loss_gen_patch" :loss_gen.item(), "loss_fmap_patch": loss_fmap.item()})

            if self.use_temp_disc and engine.state.iteration >= self.config["gan_temp"]["start_iteration"]:
                out_dict.update(disc_dict_temp)
                out_dict.update({"loss_gen_temp" :loss_gen_temp.item(), "loss_fmap_temp": loss_fmap_temp.item()})


            return out_dict

        self.logger.info("Initialize inception model...")
        self.inception_model = FIDInceptionModel()
        self.logger.info("Finished initialization of inception model...")

        # note that lpips is exactly vgg-cosine similarity as proposed in the google papers and savp
        self.lpips_fn = LPIPS(net="vgg")

        def eval_step(engine, eval_batch):
            net.eval()
            out_dict = {}
            with torch.no_grad():
                # prepare data
                weights = None
                if test_dataset.flow_weights:
                    poke = eval_batch["poke"][0].cuda(self.all_devices[0])
                    weights = eval_batch["poke"][1].cuda(self.all_devices[0])
                else:
                    poke = eval_batch["poke"].cuda(self.all_devices[0])
                x_t = eval_batch["images"][:,0].cuda(self.all_devices[0])
                x_seq_gt = eval_batch["images"][:,1:].cuda(self.all_devices[0])

                if self.config["architecture"]["disentanglement"]:
                    app_img_tr = eval_batch["app_img_random"].cuda(self.all_devices[0])
                    x_trans, *_ = net(app_img_tr, x_t, poke,len=0)
                    loss_style = style_loss(self.vgg, app_img_tr, x_trans)
                    out_dict.update({"style_loss_eval": loss_style.item()})

                n_ref_frames = self.config["data"]["n_ref_frames"] - 1 if self.poke_scale_mode else train_dataset.max_frames -1


                # eval forward passes
                seq_len = x_seq_gt.shape[1]
                x_t_hat, sigma_t, _, alpha = net(x_t,x_t,poke,len=0)
                x_seq_hat, _, sigmas_hat,_ = net(x_t, x_t, poke,len=seq_len,poke_linear=self.poke_scale_mode,
                                                 n_zero_frames=seq_len-n_ref_frames-1,poke_jump=self.poke_jump)

                if weights is not None and self.config["testing"]["metrics_on_patches"]:
                    x_seq_hat = get_patches(x_seq_hat,weights,self.config["data"],test_dataset.weight_value_flow, logger=self.logger)
                    x_seq_gt = get_patches(x_seq_gt, weights, self.config["data"], test_dataset.weight_value_flow, logger=self.logger)

                sigmas_gt = []
                ll_loss_dyn = []
                rec_imgs = []
                for n in range(seq_len):
                    x_hat_tn, s_tn, *_ = net(x_seq_gt[:, n], x_seq_gt[:, n], poke, len=0)
                    sigmas_gt.append(s_tn)
                    rec_imgs.append(x_hat_tn)
                    ll_dyn_n = vgg_loss_agg(self.vgg, x_seq_gt[:, n], x_seq_hat[:, n])
                    ll_loss_dyn.append(ll_dyn_n)
                ll_loss_tk_eval = torch.stack(ll_loss_dyn,dim=0).mean()
                rec_imgs = torch.stack(rec_imgs,1)

                if weights is not None and self.config["testing"]["metrics_on_patches"]:
                    rec_imgs = get_patches(rec_imgs, weights, self.config["data"], test_dataset.weight_value_flow, logger=self.logger)


                # apply inception model for fid calculation at all timesteps
                for t in range(x_seq_gt.shape[1]):
                    real_features_t = self.inception_model(x_seq_gt[:, t]).cpu().numpy()
                    fake_features_t = self.inception_model(x_seq_hat[:, t]).cpu().numpy()
                    if t not in self.fid_feats_fake_per_frame:
                        self.fid_feats_fake_per_frame.update({t: fake_features_t})
                        self.fid_feats_real_per_frame.update({t: real_features_t})
                    else:
                        self.fid_feats_fake_per_frame[t] = np.concatenate([self.fid_feats_fake_per_frame[t], fake_features_t], axis=0)
                        self.fid_feats_real_per_frame[t] = np.concatenate([self.fid_feats_real_per_frame[t], real_features_t], axis=0)


                # evaluate training losses
                # ll_loss_tk_eval = vgg_loss_agg(self.vgg, x_tk, x_tk_hat)
                ll_loss_t_i_eval = vgg_loss_agg(self.vgg, x_t, x_t_hat)
                dyn_losses = []
                for s_tk, s_hat_tk in zip(sigmas_gt, sigmas_hat):
                    dyn_losses.append(latent_dynamics_loss(s_hat_tk, s_tk, []))
                latent_loss_dyn_eval = torch.stack(dyn_losses).mean()


                out_dict.update({"vgg_loss_dyn_eval": ll_loss_tk_eval.item(), "loss_dis_i_eval": ll_loss_t_i_eval.item(), "latent_loss_dyn_eval": latent_loss_dyn_eval.item()})

                # compute metrics
                ssim_t = ssim_lightning(x_t, x_t_hat)
                psnr_t = psnr_lightning(x_t, x_t_hat)
                lpips_t = metric_lpips(x_t,x_t_hat, self.lpips_fn, reduce=False)



                ssim_tk, ssim_per_frame = ssim_lightning(x_seq_gt, x_seq_hat, return_per_frame=True)
                psnr_tk, psnr_per_frame = psnr_lightning(x_seq_gt, x_seq_hat, return_per_frame=True)
                lpips_avg, lpips_per_frame = metric_lpips(x_seq_gt, x_seq_hat,self.lpips_fn,reduce=False,return_per_frame=True)
                # ssim_pl, ssim_pl_per_frame = ssim_lightning(x_seq_gt,x_seq_hat,return_per_frame=True)
                # psnr_pl, psnr_pl_per_frame = psnr_lightning(x_seq_gt, x_seq_hat, return_per_frame=True)


                # append to arrays
                self.lpips["t"].append(lpips_t)
                self.psnrs["t"].append(psnr_t)
                self.ssims["t"].append(ssim_t)
                self.psnrs["tk"].append(psnr_tk)
                self.ssims["tk"].append(ssim_tk)
                self.lpips["tk"].append(lpips_avg)
                #self.ssims["pl"].append(ssim_pl)
                #self.psnrs["pl"].append(psnr_pl)


                # append the values of the respective sequence length
                [self.ssims_per_frame[key].append(ssim_per_frame[key]) if key in self.ssims_per_frame else self.ssims_per_frame.update({key:[ssim_per_frame[key]]}) for key in ssim_per_frame]
                [self.psnrs_per_frame[key].append(psnr_per_frame[key]) if key in self.psnrs_per_frame else self.psnrs_per_frame.update({key:[psnr_per_frame[key]]}) for key in psnr_per_frame]
                [self.lpips_per_frame[key].append(lpips_per_frame[key]) if key in self.lpips_per_frame else self.lpips_per_frame.update({key:[lpips_per_frame[key]]}) for key in lpips_per_frame]
                #[self.ssims_per_frame_pl[key].append(ssim_pl_per_frame[key]) if key in self.ssims_per_frame_pl else self.ssims_per_frame_pl.update({key: [ssim_pl_per_frame[key]]}) for key in ssim_pl_per_frame]
                #[self.psnrs_per_frame_pl[key].append(psnr_pl_per_frame[key]) if key in self.psnrs_per_frame_pl else self.psnrs_per_frame_pl.update({key: [psnr_pl_per_frame[key]]}) for key in psnr_pl_per_frame]

                return out_dict

        # test_it steps are performed while generating test_imgs, there n_test_img is overall number divided by number of test iterations
        n_test_img = int(self.config["testing"]["n_test_img"] // self.config["testing"]["test_it"])

        def eval_visual(engine, eval_batch):
            net.eval()
            with torch.no_grad():
                # prepare data
                if test_dataset.flow_weights:
                    poke = eval_batch["poke"][0].cuda(self.all_devices[0])
                    weights = eval_batch["poke"][1]
                else:
                    poke = eval_batch["poke"].cuda(self.all_devices[0])
                x_t = eval_batch["images"][:, 0].cuda(self.all_devices[0])
                x_seq_gt = eval_batch["images"][:, 1:].cuda(self.all_devices[0])
                flow = eval_batch["flow"]
                if self.config["architecture"]["disentanglement"]:
                    shape_img = eval_batch["img_aT"].cuda(self.all_devices[0])
                else:
                    shape_img = x_t

                n_ref_frames = self.config["data"]["n_ref_frames"] - 1 if self.poke_scale_mode else train_dataset.max_frames -1

                seq_len = x_seq_gt.shape[1]
                x_seq_hat, *_ = net(x_t,x_t, poke, len=seq_len,poke_linear=self.poke_scale_mode,n_zero_frames=seq_len-n_ref_frames-1, poke_jump = self.poke_jump)
                x_t_hat , *_ = net(x_seq_gt[:,-1],shape_img,poke,len=0)


                grid_dis = make_img_grid(x_seq_gt[:,-1],shape_img, x_t_hat,x_t, n_logged=n_test_img)
                grid_dyn = make_flow_grid(x_t, poke, x_seq_hat[:,-1], x_seq_gt[:,-1], n_logged=n_test_img, flow=flow)
                seq_vis_hat = torch.cat([x_t.unsqueeze(1), x_seq_hat], 1)
                seq_vis_gt = torch.cat([x_t.unsqueeze(1), x_seq_gt], 1)
                grid_anim = make_video(x_t,poke,seq_vis_hat,seq_vis_gt,n_logged=n_test_img,flow=flow, display_frame_nr=True)
                it = engine.state.iteration

                log_dict = {"Last Frame Comparison Test data": wandb.Image(grid_dyn, caption=f"Last frames test grid #{it}."),
                            "Disentanglement Grid Test Data": wandb.Image(grid_dis, caption=f"Test grid disentanglement #{it}."),
                            "Video Grid Test Data": wandb.Video(grid_anim,caption=f"Test Video Grid #{it}.",fps=5)}

                if self.config["testing"]["eval_app_transfer"]:
                    app_img_unrelated = eval_batch["app_img_random"].cuda(self.all_devices[0])
                    x_transferred, *_ = net(app_img_unrelated,x_t, poke,len=0)
                    transfer_grid = make_img_grid(app_img_unrelated,x_t,x_transferred)


                    log_dict.update({"Appearance transfer grid Test Data": wandb.Image(transfer_grid, caption=f"Test_grid appearance transfer #{it}")})

                wandb.log(log_dict)
                return None

        self.logger.info("Initialize engines...")
        trainer = Engine(train_step)
        evaluator = Engine(eval_step)
        test_img_generator = Engine(eval_visual)
        self.logger.info("Finish engine initialization...")


        # checkpointing
        self.logger.info("Add checkpointing and pbar...")
        n_saved = 10
        self.logger.info(f"Checkpoint saving window is {n_saved}")
        ckpt_handler = ModelCheckpoint(self.dirs["ckpt"], "reg_ckpt", n_saved=n_saved, require_empty=False)
        save_dict = {"model": net, "optimizer_dis": optimizer_dis, "optimizer_dyn": optimizer_dyn}
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=self.config["testing"]["ckpt_intervall"]),
                                  ckpt_handler,
                                  save_dict)

        if self.use_gan:
            ckpt_handler_disc = ModelCheckpoint(self.dirs["ckpt"], gan_trainer.load_key, n_saved=10, require_empty=False)
            save_dict_disc = {"model": gan_trainer.disc, "optimizer": gan_trainer.disc_opt}
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=self.config["testing"]["ckpt_intervall"]),
                                      ckpt_handler_disc,
                                      save_dict_disc)

        if self.use_temp_disc:
            ckpt_handler_disc_temp = ModelCheckpoint(self.dirs["ckpt"], gan_trainer_temp.load_key, n_saved=10, require_empty=False)
            save_dict_disc_temp = {"model": gan_trainer_temp.disc, "optimizer": gan_trainer_temp.disc_opt}
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=self.config["testing"]["ckpt_intervall"]),
                                      ckpt_handler_disc_temp,
                                      save_dict_disc_temp)
        pbar = ProgressBar(ascii=True)
        pbar.attach(trainer, output_transform=lambda x: x)
        pbar.attach(evaluator, output_transform=lambda x: x)


        #reduce the learning rate of the decoder for the image reconstruction task, such that the model focusses more on t --> tk
        @trainer.on(Events.ITERATION_COMPLETED)
        def update_lr(engine):
            self.lr_dec_t = lr_dec_rec(engine.state.iteration)
            for g in optimizer_dis.param_groups:
                if g["name"] == "decoder":
                    g["lr"] = self.lr_dec_t

        @trainer.on(Events.ITERATION_COMPLETED(every=self.config["testing"]["log_intervall"]))
        def log(engine):
            it = engine.state.iteration
            wandb.log({"iteration": it})

            # log losses
            for key in engine.state.output:
                wandb.log({key: engine.state.output[key]})

            data = engine.state.batch
            if test_dataset.flow_weights:
                poke = data["poke"][0].cuda(self.all_devices[0])
            else:
                poke = data["poke"].cuda(self.all_devices[0])
            x_t = data["images"][:, 0].cuda(self.all_devices[0])
            x_seq_gt = data["images"][:, 1:].cuda(self.all_devices[0])

            if self.config["architecture"]["disentanglement"]:
                shape_img = data["img_aT"].cuda(self.all_devices[0])
            else:
                shape_img = x_t

            n_ref_frames = self.config["data"]["n_ref_frames"] - 1 if self.poke_scale_mode else train_dataset.max_frames -1

            net.eval()
            seq_len = x_seq_gt.shape[1]
            with torch.no_grad():
                x_seq_hat, *_ = net(x_t, x_t, poke, len=seq_len, poke_linear=self.poke_scale_mode, n_zero_frames=seq_len-n_ref_frames-1, poke_jump=self.poke_jump)
                x_t_hat, *_ = net(x_seq_gt[:,-1], shape_img, poke,len=0)
                #x_t_hat_e, *_ = net(img_aT, img_sT, poke)

            grid_dis_i = make_img_grid(x_seq_gt[:,-1], shape_img, x_t_hat, x_t, n_logged=n_test_img)
            grid_dyn = make_flow_grid(x_t, poke, x_seq_hat[:,-1], x_seq_gt[:,-1], n_logged=n_test_img)
            seq_vis_hat = torch.cat([x_t.unsqueeze(1),x_seq_hat],1)
            seq_vis_gt = torch.cat([x_t.unsqueeze(1), x_seq_gt], 1)
            grid_anim = make_video(x_t,poke,seq_vis_hat,seq_vis_gt,n_logged=n_test_img, display_frame_nr=True)
            wandb.log({"Last Frame Comparison Train Data": wandb.Image(grid_dyn, caption=f"Last frames train grid after {it} train steps."),
                       "Disentanglement Grid Invariance Train Data": wandb.Image(grid_dis_i, caption=f"Invariance Disentanglement Grid on train set after {it} train steps."),
                       "Video Grid Train Data": wandb.Video(grid_anim, caption=f"Train Video Grid after {it} train steps",fps=5)})
                       #"Disentanglement Grid Equivariance Train Data": wandb.Image(grid_dis_e, caption=f"Eqiuvariance Disentanglement Grid on train set after {it} train steps.")

        self.logger.info("Initialize metrics...")
        # compute loss average over epochs
        # Average(output_transform=lambda x: x["loss_dis"]).attach(trainer, "loss_dis-epoch_avg")
        if "singlestage" not in self.config["training"] or not self.config["training"]["singlestage"]:
            Average(output_transform=lambda x: x["loss_dis"]).attach(trainer, "loss_dis-epoch_avg")
            Average(output_transform=lambda x: x["loss_dis_i_eval"]).attach(evaluator, "loss_dis_i_eval")

        Average(output_transform=lambda x: x["vgg_loss_dyn"]).attach(trainer, "vgg_loss_dyn-epoch_avg")
        Average(output_transform=lambda x: x["latent_loss_dyn"]).attach(trainer, "latent_loss_dyn-epoch_avg")

        if "disentanglement" in self.config["architecture"] and self.config["architecture"]["disentanglement"]:
            Average(output_transform=lambda x: x["style_loss"]).attach(trainer, "style_loss-epoch_avg")
            Average(output_transform=lambda x: x["style_loss_eval"]).attach(evaluator, "style_loss_eval")

        if self.use_temp_disc or self.use_gan:
            def gan_training_started(engine,epoch, key="gan"):
                return engine.state.iteration >= self.config[key]["start_iteration"]

        if self.use_gan:
            use_patchgan_metrics = MetricUsage(started=Events.EPOCH_STARTED(event_filter=gan_training_started),
                                          completed=Events.EPOCH_COMPLETED(event_filter=gan_training_started),
                                          iteration_completed=Events.ITERATION_COMPLETED(event_filter=gan_training_started))
            # gan losses
            Average(output_transform=lambda x: x["loss_gen_patch"]).attach(trainer, "loss_gen_patch-epoch_avg",usage=use_patchgan_metrics)
            Average(output_transform=lambda x: x["loss_fmap_patch"]).attach(trainer, "loss_fmap_patch-epoch_avg",usage=use_patchgan_metrics)
            Average(output_transform=lambda x: x["loss_disc_patch"]).attach(trainer, "loss_disc_patch-epoch_avg",usage=use_patchgan_metrics)
            #if self.config["gan"]["gp_weighflow_video_generatort"] > 0:
            Average(output_transform=lambda x: x["loss_gp_patch"]).attach(trainer, "loss_gp_patch-epoch_avg",usage=use_patchgan_metrics)
            Average(output_transform=lambda x: x["p_""true_patch"]).attach(trainer, "p_true_patch-epoch_avg",usage=use_patchgan_metrics)
            Average(output_transform=lambda x: x["p_fake_patch"]).attach(trainer, "p_fake_patch-epoch_avg",usage=use_patchgan_metrics)

            @trainer.on(Events.EPOCH_COMPLETED(event_filter=gan_training_started))
            def gan_stuff(engine):
                gan_trainer.disc_scheduler.step()

        if self.use_temp_disc:


            use_tmpgan_metrics = MetricUsage(started=Events.EPOCH_STARTED(event_filter=partial(gan_training_started,key="gan_temp")),
                                          completed=Events.EPOCH_COMPLETED(event_filter=partial(gan_training_started,key="gan_temp")),
                                          iteration_completed=Events.ITERATION_COMPLETED(event_filter=partial(gan_training_started,key="gan_temp")))
            # gan losses
            Average(output_transform=lambda x: x["loss_gen_temp"]).attach(trainer, "loss_gen_temp-epoch_avg",usage=use_tmpgan_metrics)
            Average(output_transform=lambda x: x["loss_fmap_temp"]).attach(trainer, "loss_fmap_temp-epoch_avg",usage=use_tmpgan_metrics)
            Average(output_transform=lambda x: x["loss_disc_temp"]).attach(trainer, "loss_disc_temp-epoch_avg",usage=use_tmpgan_metrics)
            #if self.config["gan"]["gp_weight"] > 0:
            Average(output_transform=lambda x: x["loss_gp_temp"]).attach(trainer, "loss_gp_temp-epoch_avg",usage=use_tmpgan_metrics)
            Average(output_transform=lambda x: x["p_true_temp"]).attach(trainer, "p_true_temp-epoch_avg",usage=use_tmpgan_metrics)
            Average(output_transform=lambda x: x["p_fake_temp"]).attach(trainer, "p_fake_temp-epoch_avg",usage=use_tmpgan_metrics)

            @trainer.on(Events.EPOCH_COMPLETED(event_filter=gan_training_started))
            def temp_disc_stuff(engine):
                gan_trainer_temp.disc_scheduler.step()


        # evaluation losses
        Average(output_transform=lambda x: x["vgg_loss_dyn_eval"]).attach(evaluator, "vgg_loss_dyn_eval")
        Average(output_transform=lambda x: x["latent_loss_dyn_eval"]).attach(evaluator, "latent_loss_dyn_eval")

        self.logger.info("Finish metric initialization.")


        @trainer.on(Events.EPOCH_COMPLETED(every=self.config["testing"]["n_epoch_metrics"]))
        def metrics(engine):
            # set incpetion model to cpu
            self.inception_model.eval()
            self.inception_model.cuda(self.all_devices[0])
            self.lpips_fn.cuda(self.all_devices[0])
            self.lpips_fn.eval()
            if self.config["gan_temp"]["use"]:
                gan_trainer_temp.disc.cpu()
            if self.config["gan"]["use"]:
                gan_trainer.disc.cpu()


            # compute metrics over an epoch
            self.logger.info(f"Computing metrics after epoch #{engine.state.epoch}")
            batch_size = eval_sampler.batch_size if self.config["training"]["custom_sampler"] else eval_loader.batch_size
            bs = 20 if self.is_debug else (int(8000 / batch_size) if len(test_dataset) > 8000 else len(eval_loader))
            evaluator.run(eval_loader, max_epochs=1, epoch_length=bs)
            [wandb.log({key: evaluator.state.metrics[key]}) for key in evaluator.state.metrics]
            # compute metrics
            test = np.stack(self.ssims["t"], axis=0)
            ssim_t = np.mean(np.stack(self.ssims["t"], axis=0))
            psnr_t = np.mean(np.stack(self.psnrs["t"], axis=0))
            lpips_t = np.mean(np.concatenate(self.lpips["t"], axis=0))
            ssim_tk = np.mean(np.stack(self.ssims["tk"], axis=0))
            psnr_tk = np.mean(np.stack(self.psnrs["tk"], axis=0))
            lpips_avg = np.mean(np.concatenate(self.lpips["tk"], axis=0))
            self.lpips_avg = lpips_avg

            fid_per_frame = {}
            for key in tqdm(self.fid_feats_real_per_frame, desc="Computing FID per frame"):
                fid_per_frame[key] = metric_fid(self.fid_feats_real_per_frame[key], self.fid_feats_fake_per_frame[key])

            #fid_tk = metric_fid(self.features_real_fid["tk"], self.features_fake_fid["tk"])

            fid_avg = np.mean([fid_per_frame[key] for key in fid_per_frame])


            log_dict = {"ssim-t": ssim_t, "psnr-t": psnr_t, "lpips-t": lpips_t,"ssim-tk": ssim_tk, "psnr-tk": psnr_tk, "fid-tk": fid_avg, "lpips-avg": lpips_avg}

            # add histograms for per-frame-metrics
            self.lpips_per_frame = {key: np.concatenate(self.lpips_per_frame[key], axis=0).mean() for key in self.lpips_per_frame}
            self.ssims_per_frame = {key: np.stack(self.ssims_per_frame[key], axis=0).mean() for key in self.ssims_per_frame}
            self.psnrs_per_frame = {key: np.stack(self.psnrs_per_frame[key], axis=0).mean() for key in self.psnrs_per_frame}
            # self.ssims_per_frame_pl = {key: np.stack(self.ssims_per_frame_pl[key], axis=0).mean() for key in self.ssims_per_frame_pl}
            # self.psnrs_per_frame_pl = {key: np.stack(self.psnrs_per_frame_pl[key], axis=0).mean() for key in self.psnrs_per_frame_pl}

            x = [k + 1 for k in self.lpips_per_frame]
            make_plot(x, list(self.lpips_per_frame.values()), "LPIPS of predicted frames", ylabel="Average LPIPS")
            make_plot(x, list(self.ssims_per_frame.values()), "SSIM of predicted frames", ylabel="Average SSIM")
            make_plot(x, list(self.psnrs_per_frame.values()), "PSNR of predicted frames", ylabel="Average PSNR")
            make_plot(x, list(fid_per_frame.values()), "FIDs of predicted frames", ylabel="FID")

            wandb.log(log_dict)
            # clear collection arrays
            self.__clear_metric_arrs()

            self.inception_model.cpu()
            self.lpips_fn.cpu()
            if self.config["gan_temp"]["use"]:
                gan_trainer_temp.disc.cuda(self.all_devices[0])

            if self.config["gan"]["use"]:
                gan_trainer.disc.cuda(self.all_devices[0])
            # set
            #toggle_gpu(True)

        @trainer.on(Events.ITERATION_COMPLETED(every=self.config["testing"]["test_img_intervall"]))
        def make_test_grid(engine):
            test_img_generator.run(test_loader, max_epochs=1, epoch_length=self.config["testing"]["test_it"])

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_train_avg(engine):
            wandb.log({"epoch": engine.state.epoch})
            [wandb.log({key: engine.state.metrics[key]}) for key in engine.state.metrics]
            # also perform scheduler step
            scheduler_dis.step()
            scheduler_dyn.step()

        def score_fn(engine):
            assert self.lpips_avg is not None
            return -self.lpips_avg

        # define best ckpt
        best_ckpt_handler = ModelCheckpoint(self.dirs["ckpt"],filename_prefix="ckpt_metric" ,score_function=score_fn,score_name="lpips",n_saved=5,require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.config["testing"]["n_epoch_metrics"]),best_ckpt_handler,save_dict)

        @trainer.on(Events.STARTED)
        def set_start_it(engine):
            self.logger.info(f'Engine starting from iteration {start_it}, epoch {start_epoch}')
            engine.state.iteration = start_it
            engine.state.epoch = start_epoch

        # run everything
        n_step_per_epoch = 10 if self.is_debug else len(train_loader)
        self.logger.info("Start training...")
        trainer.run(train_loader, max_epochs=n_epoch_train, epoch_length=n_step_per_epoch)
        self.logger.info("End training.")


    def test(self):
        from tqdm import tqdm
        import cv2
        from os import makedirs,path

        mod_ckpt, _ = self._load_ckpt("reg_ckpt", single_opt=False)

        dataset, transforms = get_dataset(config=self.config["data"])
        test_dataset = dataset(transforms, self.datakeys, self.config["data"], train=False)

        # get datasets for training and testing
        def w_init_fn(worker_id):
            return np.random.seed(np.random.get_state()[1][0] + worker_id)

        if self.custom_sampler:
            test_sampler = FixedLengthSampler(test_dataset, batch_size=self.config["testing"]["test_batch_size"], shuffle=True,
                                              drop_last=True, weighting=test_dataset.obj_weighting, zero_poke=False)
            test_loader = DataLoader(
                test_dataset,
                batch_sampler=test_sampler,
                num_workers=0 if self.is_debug else self.config["data"]["num_workers"],  #
                worker_init_fn=w_init_fn,
            )
            self.logger.info("Using custom data sampler")
        else:
            test_sampler = RandomSampler(test_dataset, )
            test_loader = DataLoader(test_dataset,
                                     sampler=test_sampler,
                                     batch_size=16,
                                     num_workers=self.config["data"]["num_workers"],
                                     worker_init_fn=w_init_fn,
                                     drop_last=True)
            self.logger.info("Using common torch sampler")
        # define model
        self.logger.info(f"Sequence length is {test_dataset.max_frames}")

        self.logger.info(f"Load model...")

        net_model = SkipSequenceModel if self.config["architecture"]["use_skip_model"] else SingleScaleBaseline
        net = net_model(spatial_size=self.config["data"]["spatial_size"],
                        config=self.config["architecture"], )

        weights = [5, 5, 0]
        self.logger.info(
            f"Number of trainable parameters in model is {sum(p.numel() for p in net.parameters())}"
        )
        net.load_state_dict(mod_ckpt)

        net.cuda(self.all_devices[0])
        self.logger.info("Model on gpu!")

        net.eval()

        if self.config["testing"]["mode"] == "metrics":

            fid_feats_real_per_frame = {}
            fid_feats_fake_per_frame = {}

            def metric_step(engine, eval_batch):
                net.eval()
                out_dict = {}
                with torch.no_grad():
                    # prepare data
                    weights = None
                    if test_dataset.flow_weights:
                        poke = eval_batch["poke"][0].cuda(self.all_devices[0])
                        weights = eval_batch["poke"][1].cuda(self.all_devices[0])
                    else:
                        poke = eval_batch["poke"].cuda(self.all_devices[0])
                    x_t = eval_batch["images"][:, 0].cuda(self.all_devices[0])
                    x_seq_gt = eval_batch["images"][:, 1:].cuda(self.all_devices[0])
                    
                    n_ref_frames = self.config["data"]["n_ref_frames"] -1 if "n_ref_frames" in self.config["data"] else self.config["data"]["max_frames"]

                    # eval forward passes
                    seq_len = x_seq_gt.shape[1]
                    x_t_hat, sigma_t, _, alpha = net(x_t, x_t, poke, len=0)
                    x_seq_hat, _, sigmas_hat, _ = net(x_t, x_t, poke, len=seq_len, poke_linear=self.poke_scale_mode,
                                                              n_zero_frames=seq_len-n_ref_frames-1, poke_jump=self.poke_jump)

                    if weights is not None and self.config["testing"]["metrics_on_patches"]:
                        x_seq_hat = get_patches(x_seq_hat, weights, self.config["data"], test_dataset.weight_value_flow, logger=self.logger)
                        x_seq_gt = get_patches(x_seq_gt, weights, self.config["data"], test_dataset.weight_value_flow, logger=self.logger)

                    # apply inception model for fid calculation at time t+k
                    for t in range(x_seq_gt.shape[1]):
                        real_features_t = self.inception_model(x_seq_gt[:, t]).cpu().numpy()
                        fake_features_t = self.inception_model(x_seq_hat[:, t]).cpu().numpy()
                        if t not in fid_feats_fake_per_frame:
                            fid_feats_fake_per_frame.update({t: fake_features_t})
                            fid_feats_real_per_frame.update({t: real_features_t})
                        else:
                            fid_feats_fake_per_frame[t] = np.concatenate([fid_feats_fake_per_frame[t], fake_features_t], axis=0)
                            fid_feats_real_per_frame[t] = np.concatenate([fid_feats_real_per_frame[t], real_features_t], axis=0)

                    ssim_tk, ssim_per_frame = ssim_lightning(x_seq_gt, x_seq_hat, return_per_frame=True)
                    psnr_tk, psnr_per_frame = psnr_lightning(x_seq_gt, x_seq_hat, return_per_frame=True)
                    lpips_avg, lpips_per_frame = metric_lpips(x_seq_gt, x_seq_hat, self.lpips_fn, reduce=False, return_per_frame=True)

                    # append to arrays
                    self.psnrs["tk"].append(psnr_tk)
                    self.ssims["tk"].append(ssim_tk)
                    self.lpips["tk"].append(lpips_avg)

                    # append the values of the respective sequence length
                    [self.ssims_per_frame[key].append(ssim_per_frame[key]) if key in self.ssims_per_frame else self.ssims_per_frame.update({key: [ssim_per_frame[key]]}) for key in ssim_per_frame]
                    [self.psnrs_per_frame[key].append(psnr_per_frame[key]) if key in self.psnrs_per_frame else self.psnrs_per_frame.update({key: [psnr_per_frame[key]]}) for key in psnr_per_frame]
                    [self.lpips_per_frame[key].append(lpips_per_frame[key]) if key in self.lpips_per_frame else self.lpips_per_frame.update({key: [lpips_per_frame[key]]}) for key in lpips_per_frame]

                    return out_dict

            evaluator = Engine(metric_step)

            self.logger.info("Initialize inception model...")
            self.inception_model = FIDInceptionModel()
            self.logger.info("Finished initialization of inception model...")

            # note that lpips is exactly vgg-cosine similarity as proposed in the google papers and savp
            self.lpips_fn = LPIPS(net="vgg")

            pbar = ProgressBar(ascii=True)
            pbar.attach(evaluator, output_transform=lambda x: x)

            # set incpetion model to cpu
            self.inception_model.eval()
            self.inception_model.cuda(self.all_devices[0])
            self.lpips_fn.cuda(self.all_devices[0])
            self.lpips_fn.eval()

            # compute metrics over an epoch
            self.logger.info(f"Start metrics computation.")
            batch_size = test_sampler.batch_size if self.custom_sampler else test_loader.batch_size
            el = (int(8000 / batch_size) if len(test_dataset) > 8000 else len(test_loader))
            evaluator.run(test_loader, max_epochs=1, epoch_length=el)
            # [wandb.log({key: evaluator.state.metrics[key]}) for key in evaluator.state.metrics]
            # compute metrics
            ssim_tk = np.mean(np.stack(self.ssims["tk"], axis=0))
            psnr_tk = np.mean(np.stack(self.psnrs["tk"], axis=0))
            lpips_avg = np.mean(np.concatenate(self.lpips["tk"], axis=0))

            assert list(fid_feats_real_per_frame.keys()) == list(fid_feats_fake_per_frame.keys())

            fid_per_frame = {}
            for key in tqdm(fid_feats_real_per_frame, desc="Computing FID per frame"):
                fid_per_frame[key] = metric_fid(fid_feats_real_per_frame[key], fid_feats_fake_per_frame[key])

            # fid_tk = metric_fid(self.features_real_fid["tk"], self.features_fake_fid["tk"])

            fid_avg = np.mean([fid_per_frame[key] for key in fid_per_frame])
            log_dict = {"ssim-avg-temp": ssim_tk, "psnr-avg_temp": psnr_tk, "fid-avg_temp": fid_avg, "lpips-avg-temp": lpips_avg}

            # add histograms for per-frame-metrics
            self.lpips_per_frame = {key: np.concatenate(self.lpips_per_frame[key], axis=0).mean() for key in self.lpips_per_frame}
            self.ssims_per_frame = {key: np.stack(self.ssims_per_frame[key], axis=0).mean() for key in self.ssims_per_frame}
            self.psnrs_per_frame = {key: np.stack(self.psnrs_per_frame[key], axis=0).mean() for key in self.psnrs_per_frame}

            savedir = path.join(self.dirs["generated"], "metric_summaries")
            makedirs(savedir, exist_ok=True)
            x = [k + 1 for k in self.lpips_per_frame]
            make_plot(x, list(self.lpips_per_frame.values()), "LPIPS of predicted frames", ylabel="Average LPIPS", savename=path.join(savedir, "lpips.svg"))
            make_plot(x, list(self.ssims_per_frame.values()), "SSIM of predicted frames", ylabel="Average SSIM", savename=path.join(savedir, "ssim.svg"))
            make_plot(x, list(self.psnrs_per_frame.values()), "PSNR of predicted frames", ylabel="Average PSNR", savename=path.join(savedir, "psnr.svg"))
            make_plot(x, list(fid_per_frame.values()), "FIDs of predicted frames", ylabel="FID", savename=path.join(savedir, "fid.svg"))

            self.logger.info("Averaged metrics: ")
            for key in log_dict:
                self.logger.info(f'{key}: {log_dict[key]}')

        elif self.config["testing"]["mode"] == "fvd":
            batch_size = test_sampler.batch_size if self.custom_sampler else test_loader.batch_size
            el = (int(1000 / batch_size) if len(test_dataset) > 1000 else len(test_loader))

            real_samples = []
            fake_samples = []
            real_samples_out = []

            def generate_vids(engine, eval_batch):
                net.eval()
                with torch.no_grad():
                    # prepare data
                    if test_dataset.flow_weights:
                        poke = eval_batch["poke"][0].cuda(self.all_devices[0])
                    else:
                        poke = eval_batch["poke"].cuda(self.all_devices[0])

                    if engine.state.iteration < el:
                        x_t = eval_batch["images"][:, 0].cuda(self.all_devices[0])
                        x_seq_gt = eval_batch["images"][:, 1:].cuda(self.all_devices[0])

                        n_ref_frames = self.config["data"]["n_ref_frames"] -1 if "n_ref_frames" in self.config["data"] else self.config["data"]["max_frames"]
                        # eval forward passes
                        seq_len = x_seq_gt.shape[1]
                        x_seq_hat, *_ = net(x_t, x_t, poke, len=seq_len,poke_linear=self.poke_scale_mode,
                                                                  n_zero_frames=seq_len-n_ref_frames-1, poke_jump=self.poke_jump)

                        real_batch = ((x_seq_gt + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
                        fake_batch = ((x_seq_hat + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)

                        real_samples.append(real_batch)
                        fake_samples.append(fake_batch)
                    else:
                        real_batch = ((eval_batch["images"][:, 1:] + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
                        real_samples_out.append(real_batch)

            generator = Engine(generate_vids)
            pbar = ProgressBar(ascii=True)
            pbar.attach(generator, output_transform=lambda x: x)

            self.logger.info(f"Start collecting sequences for fvd computation...")


            generator.run(test_loader, max_epochs=1, epoch_length=el)

            savedir = path.join(self.dirs["generated"], "samples_fvd")
            savedir_exmpls = path.join(savedir,"vid_examples")
            makedirs(savedir, exist_ok=True)
            makedirs(savedir_exmpls, exist_ok=True)

            real_samples = np.stack(real_samples, axis=0)
            fake_samples = np.stack(fake_samples, axis=0)
            real_samples_out = np.stack(real_samples_out, axis=0)

            n_ex = 0
            self.logger.info(f"Generating example videos")
            for i,(r,f) in enumerate(zip(real_samples,fake_samples)):
                savename = path.join(savedir_exmpls,f"sample{i}.mp4")
                r = np.concatenate([v for v in r],axis=2)
                f = np.concatenate([v for v in f],axis=2)
                all = np.concatenate([r,f],axis=1)

                writer = cv2.VideoWriter(
                    savename,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    5,
                    (all.shape[2], all.shape[1]),
                )

                # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

                for frame in all:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                writer.release()

                n_ex+=1

                if n_ex > 20:
                    break


            self.logger.info(f"Saving samples to {savedir}")
            np.save(path.join(savedir, "real_samples.npy"), real_samples)
            np.save(path.join(savedir, "fake_samples.npy"), fake_samples)

            self.logger.info(f'Finish generation of vid samples.')