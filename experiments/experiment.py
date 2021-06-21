from abc import abstractmethod
import torch
import wandb
import os
from os import path
from glob import glob
import numpy as np

from utils.general import get_logger

WANDB_DISABLE_CODE = True

class Experiment:
    def __init__(self, config:dict, dirs: dict, device):
        self.parallel = isinstance(device, list)
        self.config = config
        self.logger = get_logger(self.config["general"]["project_name"])
        self.is_debug = self.config["general"]["debug"]
        if self.is_debug:
            self.logger.info("Running in debug mode")

        if self.parallel:
            self.device = torch.device(
                f"cuda:{device[0]}" if torch.cuda.is_available() else "cpu"
            )
            self.all_devices = device
            self.logger.info("Running experiment on multiple gpus!")
        else:
            self.device = device
            self.all_devices = [device]
        self.dirs = dirs
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(dev.index if self.parallel else dev) for dev in self.all_devices])

        if self.config["general"]["restart"]:
            self.logger.info(f'Resume training run with name "{self.config["general"]["project_name"]}" on device(s) {self.all_devices}')
        else:
            self.logger.info(f'Start new training run with name "{self.config["general"]["project_name"]}" on device(s) {self.all_devices}')

        ########## seed setting ##########
        torch.manual_seed(self.config["general"]["seed"])
        torch.cuda.manual_seed(self.config["general"]["seed"])
        np.random.seed(self.config["general"]["seed"])
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config["general"]["seed"])
        rng = np.random.RandomState(self.config["general"]["seed"])

        if self.config["general"]["mode"] == "train":

            project = "visual_poking_unsupervised"
            wandb.init(
                dir=self.dirs["log"],
                project=project,
                name=self.config["general"]["project_name"],
                group=self.config["general"]["experiment"],
            )

            # log paramaters
            self.logger.info("Training parameters:")
            for key in self.config:
                if key != "testing":
                    self.logger.info(f"{key}: {self.config[key]}")  # print to console
                wandb.config.update({key: self.config[key]})  # update wandb config

    def _load_ckpt(self, key, dir=None,name=None, single_opt = True, use_best=False, load_name = "model"):
        if dir is None:
            dir = self.dirs["ckpt"]

        if name is None:
            if len(os.listdir(dir)) > 0:
                ckpts = glob(path.join(dir,"*.pt"))

                # load latest stored checkpoint
                ckpts = [ckpt for ckpt in ckpts if key in ckpt.split("/")[-1]]
                if len(ckpts) == 0:
                    self.logger.info(f"*************No ckpt found****************")
                    op_ckpt = mod_ckpt = None
                    return mod_ckpt, op_ckpt
                if use_best:
                    ckpts = [x for x in glob(path.join(dir,"*.pt")) if "=" in x.split("/")[-1]]
                    ckpts = {float(x.split("=")[-1].split(".")[0]): x for x in ckpts}

                    ckpt = torch.load(
                        ckpts[max(list(ckpts.keys()))], map_location="cpu"
                    )
                else:
                    ckpts = {float(x.split("_")[-1].split(".")[0]): x for x in ckpts}

                    ckpt = torch.load(
                        ckpts[max(list(ckpts.keys()))], map_location="cpu"
                    )

                mod_ckpt = ckpt[load_name] if load_name in ckpt else None
                if single_opt:
                    key = [key for key in ckpt if key.startswith("optimizer")]
                    assert len(key) == 1
                    key = key[0]
                    op_ckpt = ckpt[key]
                else:
                    op_ckpt = {key: ckpt[key] for key in ckpt if "optimizer" in key}

                msg = "best model" if use_best else "model"

                if mod_ckpt is not None:
                    self.logger.info(f"*************Restored {msg} with key {key} from checkpoint****************")
                else:
                    self.logger.info(f"*************No ckpt for {msg} with key {key} found, not restoring...****************")

                if op_ckpt is not None:
                    self.logger.info(f"*************Restored optimizer with key {key} from checkpoint****************")
                else:
                    self.logger.info(f"*************No ckpt for optimizer with key {key} found, not restoring...****************")
            else:
                mod_ckpt = op_ckpt = None

            return mod_ckpt, op_ckpt

        else:
            # fixme add checkpoint loading for best performing models
            ckpt_path = path.join(dir,name)
            if not path.isfile(ckpt_path):
                self.logger.info(f"*************No ckpt for model and optimizer found under {ckpt_path}, not restoring...****************")
                mod_ckpt = op_ckpt = None
            else:
                if "epoch_ckpts" in ckpt_path:
                    mod_ckpt = torch.load(
                        ckpt_path, map_location="cpu"
                    )
                    op_path = ckpt_path.replace("model@","opt@")
                    op_ckpt = torch.load(op_path,map_location="cpu")
                    return mod_ckpt,op_ckpt

                ckpt = torch.load(ckpt_path, map_location="cpu")
                mod_ckpt = ckpt[load_name] if load_name in ckpt else None
                op_ckpt = ckpt["optimizer"] if "optimizer" in ckpt else None

                if mod_ckpt is not None:
                    self.logger.info(f"*************Restored model under {ckpt_path} ****************")
                else:
                    self.logger.info(f"*************No ckpt for model found under {ckpt_path}, not restoring...****************")

                if op_ckpt is not None:
                    self.logger.info(f"*************Restored optimizer under {ckpt_path}****************")
                else:
                    self.logger.info(f"*************No ckpt for optimizer found under {ckpt_path}, not restoring...****************")

            return mod_ckpt,op_ckpt

    @abstractmethod
    def train(self):
        """
        Here, the experiment shall be run
        :return:
        """
        pass

    @abstractmethod
    def test(self):
        """
        Here the prediction shall be run
        :param ckpt_path: The path where the checkpoint file to load can be found
        :return:
        """
        pass
