import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMessageBox
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2
from array2gif import write_gif
import torch
import argparse
from glob import glob
from os import listdir,path
import time
import os
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from models.latent_flow_net import SkipSequenceModel,ResidualSequenceBaseline,ForegroundBackgroundModel
from utils.general import LoggingParent
from utils.testing import put_text_to_video_row
from data import get_dataset
from data.flow_dataset import GoogleImgDataset


class Form(QtWidgets.QDialog,LoggingParent):
    def __init__(self, config, dir_structure):
        QtWidgets.QDialog.__init__(self)
        LoggingParent.__init__(self)
        self.config = config
        self.dirs = dir_structure
        self.display_image_w, self.display_image_h = self.config["ui"]["display_size"], self.config["ui"]["display_size"]
        self.dataset, self.transforms = self.init_dataset()
        self.target_img_size =self.dataset.config["spatial_size"][0] if self.dataset.scale_poke_to_res else 256
        self.input_w, self.input_h = self.dataset.config["spatial_size"]
        self.scale_w, self.scale_h = self.input_w/self.display_image_w, self.input_h/self.display_image_h
        self.spacing = 20
        self.fps = self.config["ui"]["fps"]
        self.input_seq_length = self.config["ui"]["seq_length_to_generate"]
        self.show_id = self.config["ui"]["show_id"]
        self.interactive = self.config["ui"]["interactive"] if "interactive" in self.config["ui"] else False
        #self.actual_seq_len = self.dataset.min_frames
        #self.mag2len = {self.dataset.seq}
        self.current_video = None
        self.actual_id = None
        self.actual_length = None
        self.same_img_count = 0
        self.start_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.logger.info(f"Startup date and time: {self.start_time}")


        self.actual_torch_image, self.actual_image = self.load_next_image()
        self.old_torch_image = self.actual_torch_image
        self.init_images()
        self.net = None
        self.__get_net_model()

    def numpy_to_qImage(self, np_image):
        np_image2 = np_image.astype(np.uint8)
        qimage = QtGui.QImage(np_image2,
                              np_image2.shape[1],
                              np_image2.shape[0],
                              QtGui.QImage.Format_RGB888)
        return qimage

    def _load_ckpt(self, key, dir=None,name=None):
        if not path.isdir(dir):
            dir = self.dirs["ckpt"]

        if not path.isfile(path.join(dir,name)):
            if len(listdir(dir)) > 0:
                ckpts = glob(path.join(dir,"*.pt"))

                # load latest stored checkpoint
                ckpts = [ckpt for ckpt in ckpts if key in ckpt.split("/")[-1]]
                if len(ckpts) == 0:
                    raise FileNotFoundError(f'No checkpoints found under "{dir}"')

                ckpts = {float(x.split("_")[-1].split(".")[0]): x for x in ckpts}
                latest_ckpt = ckpts[max(list(ckpts.keys()))]
                ckpt = torch.load(
                    latest_ckpt, map_location="cpu"
                )

                if "model" in ckpt:
                    mod_ckpt = ckpt["model"]
                else:
                    raise KeyError(f'"Model"-key not in checkpoint, but required for loading the model.')

                if mod_ckpt is not None:
                    self.logger.info(f"*************Restored model with key {key} from checkpoint****************")
                else:
                    self.logger.info(f"*************No ckpt for model with key {key} found, not restoring...****************")

            else:
                raise FileNotFoundError(f'No checkpoints found under "{dir}"')

            return mod_ckpt

        else:
            # fixme add checkpoint loading for best performing models
            ckpt_path = path.join(dir,name)
            if not path.isfile(ckpt_path):
                self.logger.info(f"*************No ckpt for model and optimizer found under {ckpt_path}, not restoring...****************")
                mod_ckpt = None
            else:
                # if "epoch_ckpts" in ckpt_path:
                #     mod_ckpt = torch.load(
                #         ckpt_path, map_location="cpu"
                #     )
                #     op_path = ckpt_path.replace("model@","opt@")
                #     op_ckpt = torch.load(op_path,map_location="cpu")
                #     return mod_ckpt,op_ckpt

                ckpt = torch.load(ckpt_path, map_location="cpu")
                mod_ckpt = ckpt["model"] if "model" in ckpt else None

                if mod_ckpt is not None:
                    self.logger.info(f"*************Restored model under {ckpt_path} ****************")
                else:
                    self.logger.info(f"*************No ckpt for model found under {ckpt_path}, not restoring...****************")

            return mod_ckpt

    def forward(self,img,poke,length):
        with torch.no_grad():
            if self.config["general"]["experiment"] == "sequence_poke_model" or self.config["general"]["experiment"] == "two_stage_model":
                seq, *_ = self.net(img, img, poke, len=length)
            elif self.config["general"]["experiment"] == "poke_scale_model":
                seq, *_ = self.net(img,img, poke, len=length, poke_linear=True,)
            elif self.config["general"]["experiment"] == "fixed_length_model":
                seq, *_ = self.net(img,img,poke,len=length,)
            else:
                raise NotImplementedError(f'No forward for {self.config["general"]["experiment"]}-experiment implemented yet.')

        return seq

    def __get_net_model(self):
        if self.config["general"]["experiment"] =="sequence_poke_model" or self.config["general"]["experiment"] =="two_stage_model":
            self.net = SkipSequenceModel(self.config["data"]["spatial_size"], self.config["architecture"])
        elif self.config["general"]["experiment"] =="poke_scale_model":
            self.net = SkipSequenceModel(self.config["data"]["spatial_size"], self.config["architecture"],n_no_motion=self.config["training"]["n_no_motion"])

        elif self.config["general"]["experiment"] == "fixed_length_model":
            self.net = SkipSequenceModel(self.config["data"]["spatial_size"], self.config["architecture"])
        else:
            raise ValueError(f'The "{self.config["general"]["experiment"]}"-experiment is invalid for usage in UI.')

        self.logger.info(f'Load net on device {self.config["gpu"]}')

        if not self.config["ui"]["debug"]:
            ckpt = self._load_ckpt(key="reg_ckpt",dir=self.config["ui"]["ckpt_dir"],name=self.config["ui"]["ckpt_name"])
            self.net.load_state_dict(ckpt)
            self.net.eval()
            self.net.to(self.config["gpu"])

    def load_next_image(self):
        if self.config["ui"]["target_id"] is None:
            actual_id = int(np.random.choice(np.arange(self.dataset.datadict["img_path"].shape[0]),1))
        else:
            if isinstance(self.config["ui"]["target_id"],int):
                actual_id = self.config["ui"]["target_id"]
            else:
                assert isinstance(self.config["ui"]["target_id"],list)
                actual_id = int(np.random.choice(self.config["ui"]["target_id"],1))
        self.actual_id = actual_id
        actual_img_path = self.dataset.datadict["img_path"][actual_id]
        actual_image = cv2.imread(actual_img_path)
        actual_image = cv2.resize(
            actual_image, self.dataset.config["spatial_size"], cv2.INTER_LINEAR
        )
        actual_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB)
        actual_torch_image = self.transforms(actual_image).unsqueeze(0).to(self.config["gpu"])
        actual_image = cv2.resize(actual_image, (self.display_image_w, self.display_image_h))
        return actual_torch_image, actual_image

    def update_gt_img(self):
        self.same_img_count = 0
        self.actual_torch_image, self.actual_image = self.load_next_image()
        self.old_torch_image = self.actual_torch_image
        self.gt.set_image(self.actual_image,id=self.actual_id if self.show_id else None)
        self.update_pd(self.actual_image,id=self.actual_id if self.show_id else None)

    def reset_gt_img(self):
        self.gt.set_image(self.actual_image, id=self.actual_id if self.show_id else None)
        self.actual_torch_image = self.old_torch_image
        self.update_pd(self.actual_image,id=self.actual_id if self.show_id else None)


    def generate_gt_poke_vid(self,basepath):
        self.logger.info("Generating GT poke video....")


        # if self.dataset.var_sequence_length:
        idx = self.actual_id
        length  = self.actual_length
        ids = (idx, length - self.dataset.min_frames -1)

        # get only source image
        img = self.dataset._get_imgs(ids, sample_idx=idx,sample=True).to(self.config["gpu"]).unsqueeze(0)
        # set mask, if required
        self.dataset.mask = {}
        self.dataset._get_mask(ids)

        # sample defined number of pokes for which a video will be synthesized
        for i in tqdm(range(self.config["ui"]["n_gt_pokes"]),desc=f'Generating {self.config["ui"]["n_gt_pokes"]} gt pokes for id {idx}'):
            if self.dataset.flow_weights:
                poke, _, poke_targets = self.dataset._get_poke(ids,yield_poke_target=True)
            else:
                poke, poke_targets = self.dataset._get_poke(ids,yield_poke_target=True)
            poke = poke.to(self.config["gpu"]).unsqueeze(0)
            vid = self.forward(img,poke,length)
            vid = ((vid + 1.) * 127.5).squeeze(0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            savename = path.join(basepath, f'gt_poke_vid_{i}.mp4')
            writer = cv2.VideoWriter(
                savename,
                cv2.VideoWriter_fourcc(*"MP4V"),
                5,
                (vid.shape[2], vid.shape[1]),
            )

            for frame in vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()

            if self.config["ui"]["make_enrollment"]:
                ds = self.config["ui"]["disc_step"]

                arrow_start = poke_targets[0][0]
                arrow_dir = poke_targets[0][1]


                # arrow_start = (int(self.gt.source.x() * self.scale_w), int(self.gt.source.y() * self.scale_w))
                arrow_start = [int(c.detach().cpu().item()) for c in arrow_start]
                arrow_end = (arrow_start[0] + int(arrow_dir[0].detach().cpu().item()),
                             arrow_start[1] + int(arrow_dir[1].detach().cpu().item()))

                savename_en = savename[:-4] + "_enrollment"
                if self.config["ui"]["draw_arrow"]:
                    self.logger.info("Drawing arrow...")
                    s1 = cv2.arrowedLine(deepcopy(self.actual_image), tuple(arrow_start), arrow_end, (255, 0, 0),
                                         thickness=int(max(int(self.actual_image.shape[1] / 64), 1)))
                    s1 = cv2.resize(s1, dsize=self.config["data"]["spatial_size"], interpolation=cv2.INTER_LINEAR)
                    sn = np.stack([cv2.UMat.get(cv2.circle(cv2.UMat(s), arrow_end, 3, (255, 0, 0), -1)) for s in vid[::ds]], axis=0)
                else:
                    s1 = cv2.resize(deepcopy(self.actual_image),dsize=self.config["data"]["spatial_size"],interpolation=cv2.INTER_LINEAR)
                    sn = [s for s in vid[::ds]]

                en = np.concatenate([s1, *sn], axis=1)

                en = cv2.cvtColor(en, cv2.COLOR_BGR2RGB)
                cv2.imwrite(savename_en + ".png", en)

        # else:
        #     pass




    def save_video(self):
        if self.current_video is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No video was ever generated...")
            msg.setInformativeText("You cannot save an image before generating one!")
            msg.setWindowTitle("No video generated...")
            msg.exec_()
        else:
            self.logger.info("saving video...")
            basepath = path.join(self.dirs["generated"],"gui",f'id_{self.actual_id}',f'{self.start_time}')

            os.makedirs(basepath,exist_ok=True)
            savename = path.join(basepath,f'vid_{self.same_img_count}.mp4')
            writer = cv2.VideoWriter(
                savename,
                cv2.VideoWriter_fourcc(*"MP4V"),
                5,
                (self.current_video.shape[2], self.current_video.shape[1]),
            )

            for frame in self.current_video:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()

            if self.config["ui"]["prepare_parts"]:
                parts_path = path.join(self.dirs["generated"],"gui",f'id_{self.actual_id}')
                save_nr = self.same_img_count
                if any(map(lambda x: x.endswith(".mp4"),filter(lambda x: path.isfile(x),os.listdir(parts_path)))):
                    vid_names = filter(lambda x: x.endswith(".mp4"), filter(lambda x: path.isfile(x), os.listdir(parts_path)))
                    save_nr = int(max([int(v.split("_")[-1][:-4]) for v in vid_names])) + 1

                self.logger.info(f"actual save nr for part_vids is {save_nr}")

                savename_vids_parts = path.join(parts_path, f'vid_{save_nr}.mp4')
                writer = cv2.VideoWriter(
                    savename_vids_parts,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    5,
                    (self.current_video.shape[2], self.current_video.shape[1]),
                )

                for frame in self.current_video:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                writer.release()




            if self.config["ui"]["gt_poke"]:
                self.generate_gt_poke_vid(basepath)


            if self.config["ui"]["make_enrollment"]:
                ds = self.config["ui"]["disc_step"]
                self.logger.info("Making enrollment plot...")
                x_diff, y_diff = float(self.gt.target.x() - self.gt.source.x()) / self.gt.max_amplitude, float(self.gt.target.y() - self.gt.source.y()) / self.gt.max_amplitude

                if self.dataset.normalize_flows:
                    maxval = 20
                    x_poke, y_poke = x_diff * maxval, y_diff * maxval
                else:
                    self.logger.info("Generating poke in ablolute pixels")
                    x_poke = float(self.gt.target.x() - self.gt.source.x()) / (self.target_img_size / self.display_image_w)
                    y_poke = float(self.gt.target.y() - self.gt.source.y()) / (self.target_img_size / self.display_image_h)

                arrow_start = (int(self.gt.source.x() * self.scale_w),int(self.gt.source.y() * self.scale_w))
                arrow_end = (arrow_start[0]+int(x_poke),arrow_start[1]+int(y_poke))

                savename_en = savename[:-4] + "_enrollment"
                if self.config["ui"]["draw_arrow"]:
                    self.logger.info("Drawing arrow...")
                    s1 = cv2.arrowedLine(deepcopy(self.actual_image), arrow_start, arrow_end, (255, 0, 0),
                                         thickness=int(max(int(self.actual_image.shape[1] / 64), 1)))
                    s1 = cv2.resize(s1, dsize=self.config["data"]["spatial_size"], interpolation=cv2.INTER_LINEAR)
                    sn = np.stack([cv2.UMat.get(cv2.circle(cv2.UMat(s), arrow_end, 3, (255, 0, 0), -1)) for s in self.current_video[::ds]],axis=0)
                else:
                    s1 = deepcopy(self.actual_image)
                    s1 = cv2.resize(s1, dsize=self.config["data"]["spatial_size"], interpolation=cv2.INTER_LINEAR)
                    sn = [s for s in self.current_video[::ds]]


                en = np.concatenate([s1, *sn], axis=1)
                en = cv2.cvtColor(en,cv2.COLOR_BGR2RGB)
                cv2.imwrite(savename_en + ".png",en)

                if self.same_img_count == 0:
                    self.logger.info("save ground truth enrollment....")
                    l = self.actual_length if self.actual_length is not None else self.dataset.max_frames
                    gts = self.dataset.datadict["img_path"][self.actual_id:self.actual_id + self.dataset.subsample_step * l + 1:self.dataset.subsample_step]
                    gts = [cv2.imread(g) for g in gts]
                    gts = [cv2.resize(g,self.config["data"]["spatial_size"],interpolation=cv2.INTER_LINEAR) for g in gts]

                    writer = cv2.VideoWriter(
                        path.join(basepath, "gt_vid.mp4"),
                        cv2.VideoWriter_fourcc(*"MP4V"),
                        5,
                        (self.current_video.shape[2], self.current_video.shape[1]),
                    )

                    for frame in gts:
                        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame)

                    writer.release()

                    if self.config["ui"]["draw_arrow"]:
                        gts = [cv2.cvtColor(g, cv2.COLOR_BGR2RGB) for g in gts]
                        s1 = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(gts[0]), (int(self.gt.source.x()), int(self.gt.source.y())), (int(self.gt.target.x()), int(self.gt.target.y())), (255, 0, 0),
                                            thickness=int(max(int(self.actual_image.shape[1] / 64), 1))))
                        sn = [cv2.UMat.get(cv2.circle(cv2.UMat(s),(int(self.gt.target.x()), int(self.gt.target.y())),3,(255,0,0),-1)) for s in gts[ds::ds]]

                        gts = np.concatenate([s1]+sn,axis=1)
                        gts = cv2.cvtColor(gts,cv2.COLOR_BGR2RGB)

                    else:
                        gts = np.concatenate(gts[::ds],axis=1)

                    cv2.imwrite(path.join(basepath, "enrollment_gt.png"), gts)



                    self.logger.info("Finished saving ground truth enrollment!")



            self.same_img_count += 1




    def _generate_poke(self):
        source, target = self.gt.source, self.gt.target
        x_diff, y_diff = float(target.x() - source.x())/self.gt.max_amplitude, float(target.y() - source.y())/self.gt.max_amplitude
        # scale = np.sqrt(x_diff ** 2 + y_diff ** 2) / self.gt.max_amplitude * self.dataset.flow_cutoff
        if self.dataset.var_sequence_length:
            x_poke, y_poke = x_diff * self.dataset.flow_cutoff, y_diff * self.dataset.flow_cutoff
        else:
            if self.dataset.normalize_flows:
                maxval = self.dataset.flow_cutoff * 0.9
                x_poke, y_poke = x_diff * maxval, y_diff * maxval
            else:
                self.logger.info("Generating poke in ablolute pixels")
                x_poke = float(target.x() - source.x()) / ( self.target_img_size / self.display_image_w )
                y_poke = float(target.y() - source.y()) / ( self.target_img_size / self.display_image_h )

        if self.config["ui"]["fixed_length"]:
            length = self.config["ui"]["seq_length_to_generate"]
        else:
            poke_norm = np.linalg.norm([x_poke, y_poke])
            gs = [key for key in self.dataset.seq_len_T_chunk if self.dataset.seq_len_T_chunk[key] > poke_norm]
            length = self.dataset.min_frames + min(gs) if len(gs) > 0 else max(self.dataset.seq_len_T_chunk)
        poke = torch.zeros((2, self.input_h, self.input_w))
        half_poke_size = int(self.dataset.poke_size / 2)
        poke[0, int(source.y() * self.scale_h) - half_poke_size:int(source.y() * self.scale_h) + half_poke_size + 1,
                int(source.x() * self.scale_w) - half_poke_size:int(source.x() * self.scale_w) + half_poke_size + 1] = x_poke
        poke[1, int(source.y() * self.scale_h) - half_poke_size:int(source.y() * self.scale_h) + half_poke_size + 1,
                int(source.x() * self.scale_w) - half_poke_size:int(source.x() * self.scale_w) + half_poke_size + 1] = y_poke

        return poke.unsqueeze(0).to(self.config["gpu"]), length

    def generate_sequence(self, path=""):
        print("Begin sequence generation")
        # get poke
        poke, self.actual_length = self._generate_poke()
        # x_diff = positive if source left and target right
        # y_diff = positive if source top and target bottom
        input_img = self.actual_torch_image.to(self.config["gpu"])

        seq = self.forward(input_img,poke,self.actual_length)

        self.actual_torch_image = seq[:,-1]

        seq = ((seq + 1.) * 127.5).squeeze(0).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

        #seq = ((seq + 1.) * 127.5).squeeze(0).cpu().numpy().astype(np.uint8)

        #seq_debug = np.concatenate([np.stack([np.full_like(seq[0],255),np.zeros_like(seq[0])],0)] * 15,0)
        self.current_video = seq

        # if self.show_id:
        #     seq_shown = put_text_to_video_row(deepcopy(seq), f"id {self.actual_id}")
        # else:
        #     seq_shown = seq

        for i,img in enumerate(seq):
            self.gt.set_image(img, id=self.actual_id if self.show_id else None, sleep=True,draw=i<seq.shape[0]-1)
            # self.update_pd(img,id=self.actual_id if self.show_id else None)

        self.gt.set_image(seq[-1],id=self.actual_id if self.show_id else None, sleep=True)
        # if self.config["ui"]["save_gif"]:
        #image_sequence = list(seq)
        # make gif
        # write_gif(image_sequence, f"{path}video.gif", fps=self.fps)
        # # #
        # movie = QtGui.QMovie(f"{path}video.gif")
        # self.pd.setMovie(movie)
        # movie.start()
        # if not self.config["ui"]["save_gif"]:
        #     os.remove(f"{path}video.gif")
    def update_pd(self,img, id=None):
        if img.shape[0] != self.display_image_h or img.shape[1] != self.display_image_w:
            img = cv2.resize(img,(self.display_image_h,self.display_image_w),interpolation=cv2.INTER_LINEAR)
        if self.show_id:
            if id is not None:
                img = cv2.UMat.get(cv2.putText(cv2.UMat(img), f"id {id}", (int(img.shape[1] // 3), img.shape[0] - int(img.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                               float(img.shape[0] / 256), (255, 0, 0), int(img.shape[0] / 128)))
        self.pd.setPixmap(
            QtGui.QPixmap(self.numpy_to_qImage(img.copy())).scaled(self.display_image_w, self.display_image_h)
        )
        time.sleep(1. / self.fps)
        QApplication.processEvents()

    def init_images(self):
        self.setWindowTitle("Poking UI")

        # Ground truth frame
        #scale with reference image size 256 as all datasets' flow estimates are of size 256
        if self.dataset.normalize_flows:
            max_amplitude = None
        else:
            # perc = self.dataset.flow_norms["percentiles"][self.dataset.valid_lags[0]][self.config["ui"]["percentile"]]
            # self.logger.info(f'Chosen {self.config["ui"]["percentile"]}-percentile for current dataset is {perc} pxls')
            # max_amplitude = perc * (self.display_image_w / self.target_img_size)
            max_amplitude =  None

        self.gt = GTImage(self.display_image_w, self.display_image_h, self.actual_image, self,max_amplitude=max_amplitude,interactive_mode=self.interactive)
        self.gt.setGeometry(self.spacing, self.spacing, self.display_image_w, self.display_image_h)
        self.gt.set_image(self.actual_image, id=self.actual_id if self.show_id else None)
        self.gt_text = QtWidgets.QLabel(self)
        self.gt_text.setText("Generated Sequence")
        self.gt_text.setGeometry(self.spacing, 0, self.display_image_w, 20)

        # Predicted video
        self.pd = QtWidgets.QLabel(self)
        self.pd.setGeometry(self.spacing * 2 + self.display_image_w, self.spacing, self.display_image_w, self.display_image_h)
        self.pd.setPixmap(
            QtGui.QPixmap(self.numpy_to_qImage(self.actual_image)).scaled(self.display_image_w, self.display_image_h)
        )
        self.pd_text = QtWidgets.QLabel(self)
        self.pd_text.setText("Source Frame")
        self.pd_text.setGeometry(self.spacing * 2 + self.display_image_w, 0, self.display_image_w, 20)



        # finally gt and pd
        hbox = QtWidgets.QHBoxLayout()
        hbox2 = QtWidgets.QHBoxLayout()
        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)
        hbox.addWidget(self.gt_text)
        hbox.addWidget(self.pd_text)
        hbox2.addWidget(self.gt)
        hbox2.addWidget(self.pd)

        # Add a button to load next image in dataset
        btn2 = QtWidgets.QPushButton("Set to next Frame")
        btn2.clicked.connect(self.update_gt_img)
        generate_start = QtWidgets.QHBoxLayout()
        generate_start.addWidget(btn2)
        vbox.addLayout(generate_start)

        # Add a button to reset image in dataset
        btn3 = QtWidgets.QPushButton("Reset Frame")
        btn3.clicked.connect(self.reset_gt_img)
        generate_start = QtWidgets.QHBoxLayout()
        generate_start.addWidget(btn3)
        vbox.addLayout(generate_start)

        # add button to save generated video
        save_btn = QtWidgets.QPushButton("Save current Video")
        save_btn.clicked.connect(self.save_video)
        generate_start = QtWidgets.QHBoxLayout()
        generate_start.addWidget(save_btn)
        vbox.addLayout(generate_start)

        # show all
        self.setLayout(vbox)
        self.show()

    def init_dataset(self):
        if self.config["ui"]["debug"]:
            self.logger.info("Loading dataset in debug mode.")
            first_image = cv2.imread("UI/cat.jpg")
            first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
            first_image = cv2.resize(first_image, (self.display_image_w, self.display_image_h))
            first_image = first_image.astype(np.float)
            first_image /= 127.5
            first_image -= 1.0
            tensor_image = torch.Tensor(first_image)
            return [tensor_image, torch.ones_like(tensor_image), tensor_image], \
                   iter([tensor_image, torch.ones_like(tensor_image), tensor_image])
        else:
            dataset, transforms = get_dataset(config=self.config["data"])
            test_dataset = dataset(transforms, ["images","poke"], self.config["data"], train=False, google_imgs=self.config["ui"]["google_images"])
            if self.config["ui"]["target_id"] is not None and self.config["ui"]["write_path"]:
                if isinstance(self.config["ui"]["target_id"],int):
                    ids = [self.config["ui"]["target_id"]]
                else:
                    ids = self.config["ui"]["target_id"]

                self.logger.info("Write image paths....")
                savename = path.join(self.dirs["generated"],"image_files.txt")
                with open(savename,"w") as f:
                    for idx in ids:
                        img_path = test_dataset.datadict["img_path"][idx]
                        f.write(img_path + "\n")


            return test_dataset, transforms

class GTImage(QtWidgets.QLabel):
    def __init__(self, display_image_w, display_image_h, g_img, parent, max_amplitude=None, interactive_mode = False):
        super().__init__()
        self.draw = False
        self.display_image_w, self.display_image_h = display_image_w, display_image_h
        self.ground_image = g_img
        self.source, self.target = None, None
        self.parent = parent
        self.parent.logger.info(f"Max ampltude of GTImage is {max_amplitude}")
        if max_amplitude==None:
            self.max_amplitude = int(display_image_w/5)
        else:
            self.max_amplitude = max_amplitude

        self.interactive = interactive_mode
        if self.interactive:
            self.parent.logger.info("Start GUI in interactive mode")

    def numpy_to_qImage(self, np_image):
        np_image = np_image.astype(np.uint8)
        qimage = QtGui.QImage(np_image.copy(),
                              np_image.shape[1],
                              np_image.shape[0],
                              QtGui.QImage.Format_RGB888)
        return qimage

    def set_image(self, img, id=None, sleep=False,draw=False):
        if img.shape[0] != self.display_image_h or img.shape[1] != self.display_image_w != self.display_image_w:
            img = cv2.resize(img, (self.display_image_h, self.display_image_w), interpolation=cv2.INTER_LINEAR)
        if id is not None:
            img = cv2.UMat.get(cv2.putText(cv2.UMat(img), f"id {id}", (int(img.shape[1] // 3), img.shape[0] - int(img.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                           float(img.shape[0] / 256), (255, 0, 0), int(img.shape[0] / 128)))

        if self.interactive and self.source is not None and self.target is not None and draw:
            img = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(img),
                                           (int(self.source.x()), int(self.source.y())),
                                           (int(self.target.x()), int(self.target.y())),
                                           (0, 255, 0),
                                           thickness=min(int(np.log2(self.display_image_w))-5,1)))
        self.ground_image = img
        self.setPixmap(
            QtGui.QPixmap(self.numpy_to_qImage(self.ground_image)).scaled(self.display_image_w, self.display_image_h)
        )
        if sleep:
            time.sleep(1. / self.parent.fps)
            QApplication.processEvents()


    def mousePressEvent(self, event):
        self.draw = True
        self.source = event.localPos()

    def mouseReleaseEvent(self, event):
        self.draw = False
        self.parent.generate_sequence()

    def mouseMoveEvent(self, event):
        import copy
        pos = event.localPos()
        if pos.x() > 0.0 and pos.x() <= self.display_image_w and pos.y() > 0.0 and pos.y() <= self.display_image_h and self.draw:
            self.target = pos
            x_diff, y_diff = self.target.x() - self.source.x(), self.target.y() - self.source.y()
            amplitude = np.sqrt(x_diff**2 + y_diff**2)
            if amplitude > self.max_amplitude:
                scaler = self.max_amplitude/amplitude
                self.target.setX(int(self.source.x()+x_diff*scaler))
                self.target.setY(int(self.source.y()+y_diff*scaler))
            new_img = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(copy.deepcopy(self.ground_image)),
                                           (int(self.source.x()), int(self.source.y())),
                                           (int(self.target.x()), int(self.target.y())),
                                           (0, 255, 0),
                                           thickness=min(int(np.log2(self.display_image_w))-5,1)))
            self.setPixmap(
                QtGui.QPixmap(self.numpy_to_qImage(new_img)).scaled(self.display_image_w, self.display_image_h)
            )

def create_dir_structure(config):
    subdirs = ["ckpt", "config", "generated", "log"]
    structure = {subdir: path.join(config["general"]["base_dir"], config["general"]["experiment"], subdir, config["ui"]["project_name"]) for subdir in subdirs}
    return structure


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/sequence_poke_model.yaml",
                        help="Define config file")
    parser.add_argument("--gpu", default=[0], type=int,
                        nargs="+", help="GPU to use.")#parser.add_argument("--project_name","-n",type=str,required=True,help="The name of the project to be load.")
    parser.add_argument("-gi","--google_imgs", default=False,action="store_true", help="Whether to use images loaded from google as the data")
    parser.add_argument("-si", "--show_id", default=False, action="store_true", help="Whether to display the actual id of the image or not.")
    parser.add_argument("-me","--make_enrollment",default=False, action="store_true", help="Make enrollment plot or not")
    parser.add_argument("-da", "--draw_arrow", default=False, action="store_true", help="Whether to draw an arrow or not")
    parser.add_argument("-ds","--disc_step", default=1, type=int, help="discretization step for enrollments.")
    parser.add_argument("-gp", "--gt_poke", default=False, action="store_true", help="whether to output ground truth poke or not.")
    parser.add_argument("-pp", "--prepare_parts", default=False, action="store_true", help="whether to prepare parts or not.")
    parser.add_argument("-id", "--target_id", default=None, type=int,nargs="+", help="target od.")
    parser.add_argument("-wp", "--write_path", default=False, action="store_true", help="write image oaths or not.")

    #parser.add_argument("-np","--norm_percentile",type=int, default=50, choices=list(range(0,100,10)),help="The percentile of maxnorms of flow which shall be used for the input poke weighting for the model.")
    args = parser.parse_args()


    app = QtWidgets.QApplication(sys.argv)
    import yaml
    config_name = args.config
    with open(config_name,"r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)


    # this is for debug purposes on local machine
    if "BASEPATH_PREFIX" in os.environ:
        config["general"]["base_dir"] = os.environ["BASEPATH_PREFIX"] + config["general"]["base_dir"]
    elif "DATAPATH" in os.environ:
        config["general"]["base_dir"] = os.environ["DATAPATH"]+ config["general"]["base_dir"]#

    print(f'base dir is {config["general"]["base_dir"]}')


    #load actual model config for all fields but the "ui"-field
    dir_structure = create_dir_structure(config)
    saved_config = path.join(dir_structure["config"], "config.yaml")

    print(f'saved config is {saved_config}')

    if path.isfile(saved_config):
        with open(saved_config, "r") as f:
            complete_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

    complete_config.update({"ui": config["ui"]})
    complete_config.update({"gpu": torch.device(
        f"cuda:{int(args.gpu[0])}"
        if torch.cuda.is_available() and int(args.gpu[0]) >= 0
        else "cpu"
    )})

    complete_config["ui"].update({"google_images": args.google_imgs, "show_id": args.show_id})
    complete_config["ui"].update({"make_enrollment": args.make_enrollment})
    complete_config["ui"].update({"draw_arrow": args.draw_arrow})
    complete_config["ui"].update({"disc_step": args.disc_step})
    complete_config["ui"].update({"gt_poke": args.gt_poke})
    complete_config["ui"].update({"prepare_parts": args.prepare_parts})
    complete_config["ui"].update({"target_id":args.target_id})
    complete_config["ui"].update({"write_path": args.write_path})

    if complete_config["general"]["experiment"] == "poke_scale_model" and "match_target" in complete_config and complete_config["match_target"]["use"]:
        complete_config["data"]["normalize_flows"] = False
        complete_config["data"]["subsample_step"] = complete_config["match_target"]["subsample_step"] if "subsample_step" in complete_config["match_target"] else complete_config["data"]["subsample_step"]
        complete_config["data"]["max_frames"] = complete_config["match_target"]["max_frames"] if "max_frames" in complete_config["match_target"] else complete_config["data"]["max_frames"]
        complete_config["architecture"].update({"poke_scale":True})

    torch.cuda.set_device(complete_config["gpu"])
    if complete_config["ui"]["fixed_seed"]:
        ########## seed setting ##########
        torch.manual_seed(complete_config["general"]["seed"])
        torch.cuda.manual_seed(complete_config["general"]["seed"])
        np.random.seed(complete_config["general"]["seed"])
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(complete_config["general"]["seed"])
        rng = np.random.RandomState(complete_config["general"]["seed"])

    app_gui = Form(complete_config, dir_structure)
    app_gui.show()
    sys.exit(app.exec_())