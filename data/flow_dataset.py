from os import path
import numpy as np
import pickle
from copy import deepcopy
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms as tt
from tqdm import tqdm
import cv2
from natsort import natsorted
import os
from glob import glob


from utils.general import parallel_data_prefetch, LoggingParent
from data.helper_functions import preprocess_image
from data.base_dataset import BaseDataset


class PlantDataset(BaseDataset):
    def __init__(self, transforms, datakeys, config, train=True, google_imgs=False, n_ref_frames=None):

        self.excluded_objects = config["excluded_objects"] if "excluded_objects" in config else []
        super().__init__(transforms, datakeys, config,train=train)
        self.logger.info(f"Initializing {self.__class__.__name__}.")

        if self.config["spatial_size"][0] <= 256:
            self.flow_in_ram = self.config["flow_in_ram"] if "flow_in_ram" in self.config else False

        if self.config["spatial_size"][0] <= 256:
            self.imgs_in_ram = self.config["imgs_in_ram"] if "imgs_in_ram" in self.config else False
        # set instace specific fixed values which shall not be parameters from yaml
        self._set_instance_specific_values()


        self.subsample_step = config["subsample_step"] if "subsample_step" in config else self.subsample_step

        self.logger.info(f'Subsample step of {self.__class__.__name__} is {self.subsample_step}.')

        filt_msg = "enabled" if self.filter_flow else "disabled"
        self.logger.info(f"Flow filtering is {filt_msg} in {self.__class__.__name__}!")
        self.logger.info(f"Valid lag of {self.__class__.__name__} is {self.valid_lags[0]}")

        # load data
        metafile_path = path.join(self.datapath, f"{self.metafilename}.p")
        with open(metafile_path, "rb") as handle:
            self.data = pickle.load(handle)


        if path.isfile(path.join(self.datapath,"dataset_stats.p")) and self.normalize_flows:
            with open(path.join(self.datapath,"dataset_stats.p"),"rb") as norm_file:
                self.flow_norms = pickle.load(norm_file)


        # choose filter procedure
        available_frame_nrs = np.asarray([int(p.split("/")[-1].split(".")[0].split("_")[-1]) - int(p.split("/")[-1].split(".")[0].split("_")[-2]) for p in self.data["flow_paths"][0]])
        # filter invalid flow_paths
        self.data["flow_paths"] = [p for p in self.data["flow_paths"] if len(p) == len(available_frame_nrs)]

        self.filter_proc = self.config["filter"] if "filter" in self.config else "all"
        # remove invalid video
        valid_ids = np.logical_not(np.char.startswith(self.data["img_path"],"VID_0_3_1024x1024"))

        # set flow paths in right order after reading in the data
        if "max_fid" not in self.data:
            self.data["flow_paths"] = [natsorted(d) for d in self.data["flow_paths"]]

        # make absolute image and flow paths
        self.data["img_path"] = [
            path.join(self.datapath, p if not p.startswith("/") else p[1:]) for p in self.data["img_path"]
        ]
        self.data["flow_paths"] = [
            [path.join(self.datapath, f if not f.startswith("/") else f[1:]) for f in fs]
            for fs in self.data["flow_paths"]
        ]

        # convert to numpy array
        self.data = {key: np.asarray(self.data[key])[valid_ids] for key in self.data}

        # if max fid is not predefined, the videos, the dataset consists of are sufficiently long, such that it doesn't make much of a difference,
        # if some frames at the end are skipped, therefore, we set the last valid fid (which is indicated by "max_fid") to the maximum fid
        # in the respective sequence
        if "max_fid" not in self.data:
            available_frame_nrs = np.asarray([int(p.split("/")[-1].split(".")[0].split("_")[-1]) - int(p.split("/")[-1].split(".")[0].split("_")[-2]) for p in self.data["flow_paths"][0]])
            self.data.update({"max_fid": np.zeros((np.asarray(self.data["fid"]).shape[0],max(len(available_frame_nrs),self.valid_lags[0]+1)),dtype=np.int)})
            for vid in np.unique(self.data["vid"]):
                self.data["max_fid"][self.data["vid"] == vid] = np.amax(self.data["fid"][self.data["vid"] == vid])

        if not self.var_sequence_length and ("poke" in self.datakeys or "flow" in self.datakeys) and not self.normalize_flows:
            # reset valid_lags, such that always the right flow which corresponds to the respective sequence length, is chosen
            if not self.__class__.__name__ == "Human36mDataset":
                available_frame_nrs = np.asarray([int(p.split("/")[-1].split(".")[0].split("_")[-1]) - int(p.split("/")[-1].split(".")[0].split("_")[-2])  for p in self.data["flow_paths"][0]])
                if "n_ref_frames" not in self.config:
                    assert self.max_frames * self.subsample_step in available_frame_nrs
                    right_lag = int(np.argwhere(available_frame_nrs == self.max_frames * self.subsample_step))
                    self.logger.info(f'Last frames of sequence serves as target frame.')
                else:
                    self.logger.info(f'Number of frames in between target and start frames is {self.config["n_ref_frames"]}')
                    assert self.config["n_ref_frames"]*self.subsample_step in available_frame_nrs
                    right_lag = int(np.argwhere(available_frame_nrs==self.config["n_ref_frames"] * self.subsample_step))
                self.valid_lags = [right_lag]

            else:
                assert self.max_frames == 10
                assert self.subsample_step in [1,2]
                self.valid_lags = [0] if self.subsample_step == 1 else [1]

            self.logger.info(f"Dataset is run in fixed length mode, valid lags are {self.valid_lags}.")


        filt_msg = "enabled" if self.filter_flow else "disabled"
        self.logger.info(f"Flow filtering is {filt_msg} in {self.__class__.__name__}!")
        self.logger.info(f"Valid lag of {self.__class__.__name__} is {self.valid_lags[0]}")

        filt_msg = "enabled" if self.obj_weighting else "disabled"
        self.logger.info(f"Object weighting is {filt_msg} in {self.__class__.__name__}!")
        filt_msg = "enabled" if self.flow_weights else "disabled"
        self.logger.info(f"Patch weighting is {filt_msg} in {self.__class__.__name__}!")
        filt_msg = "enabled" if self.use_flow_for_weights else "disabled"
        self.logger.info(f"Flow patch extraction is {filt_msg} in {self.__class__.__name__}!")

        if self.filter_proc == "action":
            self.data = {key:self.data[key][self.data["action_id"]==2] for key in self.data}
        elif self.filter_proc == "pose":
            self.data = {key: self.data[key][self.data["action_id"] == 1] for key in self.data}


        # on this point, the raw data is parsed and can be processed further
        # exclude invalid object ids from data
        self.logger.info(f"Excluding the following, user-defined object ids: {self.excluded_objects} from dataloading.")
        kept_ids = np.nonzero(np.logical_not(np.isin(self.data["object_id"], self.excluded_objects)))[0]
        self.data = {key:self.data[key][kept_ids] for key in self.data}

        self.split = self.config["split"]
        split_data, train_indices, test_indices = self._make_split(self.data)

        self.datadict = (
            split_data["train"] if self.train else split_data["test"]
        )
        msg = "train" if self.train else "test"

        vids, start_ids = np.unique(self.datadict["vid"],return_index=True)

        # get start and end ids per sequence
        self.eids_per_seq = {vid: np.amax(np.flatnonzero(self.datadict["vid"] == vid)) for vid in vids}
        seids = np.asarray([self.eids_per_seq[self.datadict["vid"][i]] for i in range(self.datadict["img_path"].shape[0])],dtype=np.int)
        self.datadict.update({"seq_end_id": seids})

        self.sids_per_seq = {vid:i for vid,i in zip(vids,start_ids)}

        self.seq_len_T_chunk = {l: c for l,c in enumerate(np.linspace(0,self.flow_cutoff,self.max_frames,endpoint=False))}
        # add last chunk
        self.seq_len_T_chunk.update({self.max_frames: self.flow_cutoff})
        if self.var_sequence_length:
            if "flow_range" in self.datadict.keys():
                self.ids_per_seq_len = {length: np.flatnonzero(np.logical_and(np.logical_and(self.datadict["flow_range"][:,1,self.valid_lags[0]]>self.seq_len_T_chunk[length],
                                                                                             np.less_equal(np.arange(self.datadict["img_path"].shape[0]) +
                                                                                                           (self.min_frames + length)*self.subsample_step + 1,
                                                                                                            self.datadict["seq_end_id"])),
                                                                              np.less_equal(self.datadict["fid"],self.datadict["max_fid"][:,self.valid_lags[0]])))
                                    for length in np.arange(self.max_frames)}
            else:
                self.ids_per_seq_len = {length: np.flatnonzero(np.less_equal(self.datadict["fid"],self.datadict["max_fid"][:,self.valid_lags[0]])) for length in np.arange(self.max_frames)}


        for length in self.ids_per_seq_len:
            actual_ids = self.ids_per_seq_len[length]
            oids, counts_per_obj = np.unique(self.datadict["object_id"][actual_ids],return_counts=True)
            weights = np.zeros_like(actual_ids,dtype=np.float)
            for oid,c in zip(oids,counts_per_obj):
                weights[self.datadict["object_id"][actual_ids]==oid] = 1. / (c * oids.shape[0])

            self.object_weights_per_seq_len.update({length:weights})

        obj_ids, obj_counts = np.unique(self.datadict["object_id"], return_counts=True)
        weights = np.zeros_like(self.datadict["object_id"], dtype=np.float)
        for (oid, c) in zip(obj_ids, obj_counts):
            weights[self.datadict["object_id"] == oid] = 1. / c

        weights = weights / obj_ids.shape[0]

        self.datadict.update({"weights": weights})



        if self.flow_in_ram:
            self.logger.warn(f"Load flow maps in RAM... please make sure to have enough capacity there.")
            assert len(self.valid_lags) == 1
            self.loaded_flows = parallel_data_prefetch(self._read_flows, self.datadict["flow_paths"][:,self.valid_lags[0]],n_proc=72,cpu_intensive=True)
            assert self.loaded_flows.shape[0] == self.datadict["img_path"].shape[0]

        if self.imgs_in_ram:
            self.logger.warn(f"Load images in RAM... please make sure to have enough capacity there.")
            self.loaded_imgs = parallel_data_prefetch(self._read_imgs, self.datadict["img_path"],n_proc=72,cpu_intensive=True)
            assert self.loaded_imgs.shape[0] == self.datadict["img_path"].shape[0]

        if google_imgs:
            img_paths = [p for p in glob(path.join(self.datapath,"google_images", "*")) if path.isfile(p) and any(map(lambda x: p.endswith(x), ["jpg", "jpeg", "png"]))]
            self.datadict["img_path"] = np.asarray(img_paths)
            self.logger.info("Use images from Google.")

        msg = "Flow normalization enabled!" if self.normalize_flows else "Flow normalization disabled!"

        self.logger.info(
            f'Initialized {self.__class__.__name__} in "{msg}"-mode. Dataset consists of {self.__len__()} samples. ' + msg
        )

    def _set_instance_specific_values(self):
        # set flow cutoff to 0.2 as this seems to be a good heuristic for Plants

        self.valid_lags = [0]
        self.flow_cutoff = 0.4
        self.extended_annotations = False
        self.subsample_step = 2
        self.min_frames = 5
        self.obj_weighting = True

        if not 8 in self.excluded_objects:
            self.excluded_objects.append(8)
        self.metafilename = "meta"
        # self.metafilename = 'test_codeprep_metadata'

    def _read_flows(self,data):
        read_flows = []
        flow_paths = data
        def proc_flow(flow):
            org_shape = float(flow.shape[-1])
            dsize = None
            if "spatial_size" in self.config:
                dsize = self.config["spatial_size"]
            elif "resize_factor" in self.config:
                dsize = (
                    int(float(flow.shape[1]) / self.config["resize_factor"]),
                    int(float(flow.shape[2]) / self.config["resize_factor"]),
                )

            flow = F.interpolate(
                torch.from_numpy(flow).unsqueeze(0), size=dsize, mode="bilinear", align_corners=True
            ).numpy()

            flow = flow / (org_shape / dsize[0])

            return flow


        for i, flow_path in enumerate(tqdm(flow_paths)):
            try:
                f = np.load(flow_path)
                f = proc_flow(f)
            except ValueError:
                try:
                    f = np.load(flow_path, allow_pickle=True)
                    f = proc_flow(f)
                except Exception as ex:
                    self.logger.error(ex)
                    read_flows.append("None")
                    continue
            except:
                self.logger.error("Fallback error ocurred. Append None and continue")
                read_flows.append("None")
                continue

            read_flows.append(f)

        return np.concatenate(read_flows,axis=0)

    def _read_imgs(self,imgs):
        read_imgs = []

        for img_path in tqdm(imgs):
            img = cv2.imread(img_path)
            # image is read in BGR
            img = preprocess_image(img, swap_channels=True)
            img = cv2.resize(
                img, self.config["spatial_size"], cv2.INTER_LINEAR
            )
            read_imgs.append(img)

        return read_imgs

    def _make_split(self,data):

        vids = np.unique(self.data["vid"])
        split_data = {"train": {}, "test": {}}

        if self.split == "videos":
            # split such that some videos are held back for testing
            self.logger.info("Splitting data after videos")
            shuffled_vids = deepcopy(vids)
            np.random.shuffle(shuffled_vids)
            train_vids = shuffled_vids[: int(0.8 * shuffled_vids.shape[0])]
            train_indices = np.nonzero(np.isin(data["vid"], train_vids))[0]
            test_indices = np.nonzero(np.logical_not(train_indices))[0]
            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

        else:
            self.logger.info(f"splitting data across_videos")
            train_indices = np.asarray([],dtype=np.int)
            test_indices = np.asarray([], dtype=np.int)
            for vid in vids:
                indices = np.nonzero(data["vid"] == vid)[0]
                # indices = np.arange(len(tdata["img_path"]))
                # np.random.shuffle(indices)
                train_indices = np.append(train_indices,indices[: int(0.8 * indices.shape[0])])
                test_indices = np.append(test_indices,indices[int(0.8 * indices.shape[0]) :])


            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

        return split_data, train_indices, test_indices

class VegetationDataset(PlantDataset):

    def _set_instance_specific_values(self):
        self.filter_flow = False
        self.valid_lags = [0]
        self.flow_cutoff = .3
        self.min_frames = 5
        self.subsample_step = 2
        # self.datapath = "/export/data/ablattma/Datasets/vegetation_new/"
        self.metafilename = "meta"
        self.datadict.update({"train": []})
        self.obj_weighting = True
        # set flow_weights to false
        self.flow_weights = False

    def _make_split(self,data):
        split_data = {"train":{},"test":{}}
        train_ids = np.flatnonzero(data["train"])
        test_ids = np.flatnonzero(np.logical_not(data["train"]))
        assert np.intersect1d(train_ids,test_ids).size == 0
        split_data["train"] = {
                key: data[key][train_ids] for key in data
            }
        split_data["test"] = {
            key: data[key][test_ids] for key in data
        }

        return split_data, train_ids, test_ids



class TaichiDataset(VegetationDataset):

    def _set_instance_specific_values(self):
        self.filter_flow = True
        self.valid_lags = [1]
        self.flow_cutoff = .1
        self.min_frames = 5
        self.subsample_step = 2
        # self.datapath = "/export/scratch/compvis/datasets/taichi/taichi/"
        self.metafilename = 'meta'
        self.datadict.update({"train": []})
        self.obj_weighting = False
        # set flow_weights to false
        self.flow_weights = self.config["flow_weights"] if "flow_weights" in self.config else True
        self.flow_width_factor = 5
        self.target_lags = [10,20]


class LargeVegetationDataset(VegetationDataset):

    def _set_instance_specific_values(self):
        self.filter_flow = False
        self.valid_lags = [0]
        self.flow_cutoff = .1
        self.min_frames = 5
        self.subsample_step = 2
        # self.datapath = "/export/scratch/compvis/datasets/plants/processed_256_resized/"
        self.metafilename = "meta"
        self.datadict.update({"train": []})
        self.excluded_objects = [1,2,3]
        self.obj_weighting = True

class IperDataset(PlantDataset):

    def _set_instance_specific_values(self):
        self.filter_flow = True
        self.flow_width_factor = 5
        self.valid_lags = [0]
        # set flow cutoff to 0.45 as this seems to be a good heuristic for Iper
        self.flow_cutoff = 0.6

        self.min_frames = 5


        # self.datapath = "/export/scratch/compvis/datasets/iPER/processed_256_resized/"
        self.metafilename = 'meta' #"test_codeprep_metadata"

        self.datadict.update({"actor_id": [], "action_id": []})

        # set object weighting always to false
        self.obj_weighting = False
        self.flow_weights = self.config["flow_weights"] if "flow_weights" in self.config else True
        self.use_flow_for_weights = False


    def _make_split(self,data):
        split_data = {"train": {}, "test": {}}

        if self.split == "videos":
            key = "vid"
        elif self.split == "objects":
            key = "object_id"
        elif self.split == "actions":
            key = "action_id"
        elif self.split == "actors":
            key = "actor_id"
        elif self.split == "official":
            # this is the official train test split as in the original paper
            with open(path.join("/".join(self.datapath.split("/")[:-1]),"train.txt"),"r") as f:
                train_names = f.readlines()

            train_indices = np.asarray([],dtype=np.int)
            for n in train_names:
                n = n.replace("/","_").rstrip()
                train_indices = np.append(train_indices,np.flatnonzero(np.char.find(data["img_path"],n) != -1))

            train_indices = np.sort(train_indices)
            test_indices = np.flatnonzero(np.logical_not(np.isin(np.arange(data["img_path"].shape[0]),train_indices)))

            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

            return split_data, train_indices, test_indices

        else:

            vids = np.unique(self.data["vid"])

            self.logger.info(f"splitting data across_videos")
            train_indices = np.asarray([], dtype=np.int)
            test_indices = np.asarray([], dtype=np.int)
            for vid in vids:
                indices = np.nonzero(data["vid"] == vid)[0]
                # indices = np.arange(len(tdata["img_path"]))
                # np.random.shuffle(indices)
                train_indices = np.append(train_indices, indices[: int(0.8 * indices.shape[0])])
                test_indices = np.append(test_indices, indices[int(0.8 * indices.shape[0]):])

            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

            return split_data, train_indices, test_indices

        # split such that some objects are held back for testing
        self.logger.info(f"Splitting data after {key}")
        ids = np.unique(data[key])
        shuffled_ids = deepcopy(ids)
        np.random.shuffle(shuffled_ids)
        train_ids = shuffled_ids[: int(0.8 * shuffled_ids.shape[0])]
        train_indices = np.flatnonzero(np.isin(data[key], train_ids))
        test_indices = np.flatnonzero(np.logical_not(np.isin(np.arange(self.data["img_path"].shape[0]),train_indices)))

        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)

        split_data["train"] = {
            key: data[key][train_indices] for key in data
        }
        split_data["test"] = {
            key: data[key][test_indices] for key in data
        }


        return split_data, train_indices, test_indices

class Human36mDataset(PlantDataset):
    def _set_instance_specific_values(self):
        self.valid_lags = [1]
        self.flow_cutoff = 0.3

        self.min_frames = 5
        self.subsample_step = 2


        # self.datapath = "/export/scratch/compvis/datasets/human3.6M/video_prediction"

        self.metafilename = "meta"
        self.datadict.update({"actor_id": [], "action_id": [], "train": []})

        # set object weighting always to false
        self.obj_weighting = False
        self.filter_flow = False
        self.flow_width_factor = 5
        self.flow_weights = True
        self.use_flow_for_weights = True
        self.use_lanczos = True




    def _make_split(self,data):

        split_data = {"train": {}, "test": {}}

        if self.split == "official":
            train_ids = np.flatnonzero(data["train"])
            test_ids = np.flatnonzero(np.logical_not(data["train"]))
            assert np.intersect1d(train_ids, test_ids).size == 0
            split_data["train"] = {
                key: data[key][train_ids] for key in data
            }
            split_data["test"] = {
                key: data[key][test_ids] for key in data
            }

            return split_data, train_ids, test_ids
        elif self.split == "gui":
            vids = np.unique(self.data["vid"])

            self.logger.info(f"splitting data across_videos")
            train_indices = np.asarray([], dtype=np.int)
            test_indices = np.asarray([], dtype=np.int)
            for vid in vids:
                indices = np.nonzero(data["vid"] == vid)[0]
                # indices = np.arange(len(tdata["img_path"]))
                # np.random.shuffle(indices)
                train_indices = np.append(train_indices, indices[: int(0.8 * indices.shape[0])])
                test_indices = np.append(test_indices, indices[int(0.8 * indices.shape[0]):])

            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

            return split_data, train_indices, test_indices
        else:
            raise ValueError(f'Specified split type "{self.split}" is not valid for Human36mDataset.')





class GoogleImgDataset(Dataset, LoggingParent):

    def __init__(self, base_dir, config,):
        Dataset.__init__(self)
        LoggingParent.__init__(self)
        self.logger.info(f"Initialize GoogleImgDataset with basepath {base_dir}")
        self.config = config
        img_paths = [p for p in glob(path.join(base_dir,"*")) if path.isfile(p) and any(map(lambda x: p.endswith(x),["jpg","jpeg","png"]))]
        self.datadict = {"img_path": np.asarray(img_paths)}
        self.transforms = tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ])
        self.logger.info(f"Initialized Dataset with {self.__len__()} images")

    def __getitem__(self, idx):
        return self.datadict["img_path"][idx]

    def __len__(self):
        return self.datadict["img_path"].shape[0]

if __name__ == "__main__":
    import yaml
    import torch
    from torchvision import transforms as tt
    from torch.utils.data import DataLoader, RandomSampler
    import cv2
    from os import makedirs
    from tqdm import tqdm

    from data import get_dataset
    from data.samplers import SequenceSampler, SequenceLengthSampler
    from utils.testing import make_video, make_flow_grid
    from utils.general import get_patches

    seed = 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    # random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    rng = np.random.RandomState(42)

    # load config
    fpath = path.dirname(path.realpath(__file__))
    configpath = path.abspath(path.join(fpath, "../config/test_config.yaml"))
    with open(configpath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    transforms = tt.Compose(
        [tt.ToTensor(), tt.Lambda(lambda x: (x * 2.0) - 1.0)]
    )

    datakeys = ["images", "img_aT", "img_sT", "app_img_cmp", "app_img_random","flow", "poke"]

    make_overlay = config["general"]["overlay"]

    # generate dataset
    dset, transforms = get_dataset(config["data"],transforms)
    test_dataset = dset(transforms, datakeys, config["data"],train=True)

    save_dir = f"test_data/{test_dataset.__class__.__name__}"
    makedirs(save_dir, exist_ok=True)
    print(test_dataset.datapath)


    if test_dataset.yield_videos:
        def init_fn(worker_id):
            return np.random.seed(np.random.get_state()[1][0] + worker_id)

        if test_dataset.var_sequence_length:
            sampler = SequenceLengthSampler(test_dataset,shuffle=True,drop_last=False, batch_size=config["training"]["batch_size"],zero_poke=config["data"]["include_zeropoke"])
            loader = DataLoader(test_dataset, batch_sampler=sampler, num_workers=config["data"]["num_workers"], worker_init_fn=init_fn)
        else:
            sampler = RandomSampler(test_dataset)
            loader = DataLoader(test_dataset,batch_size=config["training"]["batch_size"], sampler=sampler,num_workers=config["data"]["num_workers"],
                                worker_init_fn=init_fn, drop_last= True)

        n_logged = config["testing"]["n_logged"]

        for i, batch in enumerate(tqdm(loader)):



            if i >200:
                break

            imgs = batch["images"][:n_logged]
            src_img = imgs[:,0]
            tgt_img = imgs[:,-1]
            flow = batch["flow"][:n_logged]
            poke = batch["poke"][:n_logged][0] if test_dataset.flow_weights else batch["poke"][:n_logged]
            weights = batch["poke"][:n_logged][1] if test_dataset.flow_weights else None

            postfix = "weighted" if config["data"]["object_weighting"] else "unweighted"
            if weights is not None:
                imgs = get_patches(imgs,weights,config["data"],test_dataset.weight_value_flow)
                postfix = postfix + "_patched"
            out_vid = make_video(imgs[:,0],poke,imgs,imgs,n_logged=min(n_logged,config["training"]["batch_size"]),flow=flow,logwandb=False, flow_weights=weights)

            warping_test = make_flow_grid(src_img,flow,tgt_img,tgt_img,n_logged=min(n_logged,config["training"]["batch_size"]))
            warping_test = cv2.cvtColor(warping_test,cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.join(save_dir,f'warping_test-{i}.png'),warping_test)

            savename = path.join(save_dir,f"vid-grid-{i}-{postfix}.mp4")

            writer = cv2.VideoWriter(
                savename,
                cv2.VideoWriter_fourcc(*"MP4V"),
                5,
                (out_vid.shape[2], out_vid.shape[1]),
            )

            # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

            for frame in out_vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()

    else:

        sampler = SequenceSampler(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, drop_last=False)
        loader = DataLoader(test_dataset, batch_sampler=sampler, num_workers=config["data"]["num_workers"])

        #assert sampler.batch_size == 1
        postfix = "filt" if test_dataset.filter_flow else "nofilt   "

        for i, batch in enumerate(tqdm(loader)):

            if i > 200:
                break


            batch = {key: batch[key].squeeze(0) if not isinstance(batch[key],list) else [e.squeeze(0) for e in batch[key]] for key in batch}
            src_img = batch["images"][0]
            tgt_img = batch["images"][-1]
            # vis augmented images
            img_aT = batch["img_aT"][0]
            img_sT = batch["img_sT"]
            img_dis = batch["app_img_random"]
            img_cmp = batch["app_img_cmp"]


            # # vis flow
            flow_map = batch["flow"].permute(1, 2, 0).cpu().numpy()
            flow_map -= flow_map.min()
            flow_map /= flow_map.max()
            flow_map = (flow_map * 255.0).astype(np.uint8)
            # vis poke
            poke = batch["poke"][0].permute(1, 2, 0).cpu().numpy() if test_dataset.flow_weights else batch["poke"].permute(1, 2, 0).cpu().numpy()
            if test_dataset.flow_weights:
                weight_map = batch["poke"][1].cpu().numpy()
                weight_map = ((weight_map - weight_map.min()) / weight_map.max() * 255.).astype(np.uint8)
                heatmap = cv2.applyColorMap(weight_map, cv2.COLORMAP_HOT)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

            # visualize poke patch in flow map as white region
            flow_map = np.where((poke**2).sum(-1,keepdims=True)>0, np.full_like(flow_map, 255), flow_map)
            poke -= poke.min()
            poke /= poke.max()
            poke = (poke * 255.0).astype(np.uint8)



            # vis inverted flow
            # flow_map_inv = batch["flow_inv"].permute(1, 2, 0).cpu().numpy()
            # flow_map_inv -= flow_map_inv.min()
            # flow_map_inv /= flow_map_inv.max()
            # flow_map_inv = (flow_map_inv * 255.0).astype(np.uint8)

            # vis images
            src_img = (
                ((src_img.permute(1, 2, 0).cpu() + 1) * 127.5)
                .numpy()
                .astype(np.uint8)
            )
            tgt_img = (
                ((tgt_img.permute(1, 2, 0).cpu() + 1) * 127.5)
                .numpy()
                .astype(np.uint8)
            )
            img_aT = ((img_aT.permute(1, 2, 0).cpu() + 1) * 127.5).numpy().astype(np.uint8)
            img_sT = ((img_sT.permute(1, 2, 0).cpu() + 1) * 127.5).numpy().astype(np.uint8)
            img_dis = ((img_dis.permute(1, 2, 0).cpu() + 1) * 127.5).numpy().astype(np.uint8)
            img_cmp = ((img_cmp.permute(1, 2, 0).cpu() + 1) * 127.5).numpy().astype(np.uint8)
            if make_overlay:
                overlay = cv2.addWeighted(src_img,0.5,tgt_img,0.5,0)
            else:
                tgt_img = [tgt_img,heatmap] if test_dataset.flow_weights else [tgt_img]
            zeros = np.expand_dims(np.zeros_like(flow_map).sum(2), axis=2)
            flow_map = np.concatenate([flow_map, zeros], axis=2)
            poke = np.concatenate([poke, zeros], axis=2)

            # flow_map_inv = np.concatenate([flow_map_inv,zeros],axis=2)
            if make_overlay:
                grid = np.concatenate([src_img, *tgt_img,overlay, img_sT, img_aT, img_dis, img_cmp, flow_map, poke], axis=1).astype(np.uint8)
            else:
                grid = np.concatenate([src_img, *tgt_img, img_sT, img_aT, img_dis, img_cmp, flow_map, poke], axis=1).astype(np.uint8)
            grid = cv2.cvtColor(grid,cv2.COLOR_BGR2RGB)
            cv2.imwrite(path.join(save_dir, f"test_grid_{i}-{postfix}.png"), grid)
