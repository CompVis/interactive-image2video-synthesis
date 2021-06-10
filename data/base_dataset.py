from functools import partial
from itertools import chain
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as FT
from PIL import Image
import numpy as np
from abc import abstractmethod
import cv2


from utils.general import convert_flow_2d_to_3d, get_flow_gradients
from data.helper_functions import preprocess_image
from utils.general import LoggingParent

class FlowError(Exception):
    """Raises an exception when no valid flow file could be found

    """
    def __init__(self, path, msg=None):
        if msg is None:
            message = f'Could not load flow file "{path}" neither with "allow_pickle=False" nor with "allow_pickle=True". Considering different sequence....'
        else:
            message = msg
        super().__init__(message)

class BaseDataset(Dataset, LoggingParent):
    def __init__(self, transforms, datakeys: list, config: dict, train=True):
        Dataset.__init__(self)
        LoggingParent.__init__(self)

        # list of keys for the data that shall be retained
        assert len(datakeys) > 0
        self.datakeys = datakeys
        # torchvision.transforms
        self.transforms = transforms
        # config: contains all relevant configuration parameters
        self.config = config
        self.train = train
        assert "spatial_size" in self.config

        self.datapath = self.config['datapath']

        # self.valid_lags = np.unique(self.config["valid_lags"]) if "valid_lags" in self.config else list(range(6))


        self.yield_videos = self.config["yield_videos"] if "yield_videos" in self.config else False

        # everything, which has to deal with variable sequence lengths
        self.var_sequence_length = self.config["var_sequence_length"] if "var_sequence_length" in self.config and self.yield_videos else False
        self.longest_seq_weight = self.config["longest_seq_weight"] if "longest_seq_weight" in self.config else None
        self.scale_poke_to_res = self.config["scale_poke_to_res"] if "scale_poke_to_res" in self.config else False
        if self.scale_poke_to_res:
            self.logger.info(f'Scaling flows and pokes to dataset resolution, which is {self.config["spatial_size"]}')

        self.logger.info(f'Dataset is yielding {"videos" if self.yield_videos else "images"}.')
        self.poke_size = self.config["poke_size"] if "poke_size" in self.config else self.config["spatial_size"][0] / 128 * 10
        if "poke" in self.datakeys:
            self.logger.info(f"Poke size is {self.poke_size}.")

        # for flow filtering: default values are such that nothing changes
        self.filter_flow = False
        self.flow_width_factor = None

        # whether fancy appearance augmentation shall be used or not
        self.fancy_aug = self.config["fancy_aug"] if "fancy_aug" in self.config else False

        # flow weighting, if intended to be enabled
        self.flow_weights = self.config["flow_weights"] if "flow_weights" in self.config else False
        self.weight_value_flow = self.config["foreground_value"] if "foreground_value" in self.config else 1.
        self.weight_value_poke = self.config["poke_value"] if "poke_value" in self.config else 1.
        self.weight_value_bg = self.config["background_weight"] if "background_weight" in self.config else 1.

        # whether to use only one value in for poke or the complete flow field within that patch
        self.equal_poke_val = self.config["equal_poke_val"] if "equal_poke_val" in self.config else True

        # Whether or not to normalize the flow values
        self.normalize_flows = self.config["normalize_flows"] if "normalize_flows" in self.config else False
        # Whether to weight different objects (i.e. samples with different object_ids) the way that the should be yield equally often (recommended for imbalanced datasets)
        self.obj_weighting = self.config["object_weighting"] if "object_weighting" in self.config else False

        self.p_col= self.config["p_col"] if "p_col" in self.config else 0
        self.p_geom = self.config["p_geom"] if "p_geom" in self.config else 0
        self.ab = self.config["augment_b"] if "augment_b" in self.config else 0
        self.ac = self.config["augment_c"] if "augment_c" in self.config else 0
        self.ah = self.config["augment_h"] if "augment_h" in self.config else 0
        self.a_s = self.config["augment_s"] if "augment_s" in self.config else 0
        self.ad = self.config["aug_deg"] if "aug_deg" in self.config else 0
        self.at = self.config["aug_trans"] if "aug_trans" in self.config else (0,0)
        self.use_lanczos = self.config["use_lanczos"] if "use_lanczos" in self.config else False

        self.pre_T = T.ToPILImage()
        self.z1_normalize = "01_normalize" in self.config and self.config["01_normalize"]
        if self.z1_normalize:
            self.post_T = T.Compose([T.ToTensor(),])
        else:
            self.post_T = T.Compose([T.ToTensor(),T.Lambda(lambda x: (x * 2.0) - 1.0)])
        self.post_edges = T.Compose([T.ToTensor()])


        # key:value mappings for every datakey in self.datakeys
        self._output_dict = {
            "images": [partial(self._get_imgs)],
            "poke": [self._get_poke],
            "flow": [self._get_flow],
            "img_aT": [partial(self._get_imgs,use_fb_aug = self.fancy_aug), ["color"]],
            "img_sT": [partial(self._get_imgs,sample=True),["geometry"]],
            "app_img_random": [self._get_transfer_img],
            "app_img_dis": [partial(self._get_imgs, sample=True), ["color", "geometry"]],
            "app_img_cmp": [self._get_transfer_img],
            "flow_3D": [self._get_3d_flow],
            "poke_3D": [self._get_3d_poke],
            "edge_image": [self._get_edge_image],
            "edge_flow": [self._get_edge_flow],
            "flow_3D_series": [self._get_flow_series],
            "image_series": [self._get_image_series]
        }

        if self.fancy_aug:
            assert "app_img_dis" not in self.datakeys


        # the data that's held by the dataset
        self.datadict = {
            "img_path": [],
            "flow_paths": [],
            "img_size": [],
            "flow_size": [],
            "vid": [],
            "fid": [],
            "object_id": [],
            # "original_id": [],
            "flow_range": []
        }


        self.max_frames = self.config["max_frames"] if "max_frames" in self.config else 1

        self.augment = self.config["augment_wo_dis"] if ("augment_wo_dis" in self.config and self.train) else False
        self.color_transfs = None
        self.geom_transfs = None


        self.subsample_step = 1
        self.min_frames = None


        # sequence start and end ids are related to the entire dataset and so is self.img_paths
        self.eids_per_seq = {}
        self.sids_per_seq = {}
        self.seq_len_T_chunk = {}
        self.max_trials_flow_load = 50
        #self.img_paths = {}
        self.mask=None
        self.flow_norms = None
        self.flow_in_ram = False
        self.imgs_in_ram = False
        self.outside_length = None
        self.loaded_flows = []
        self.loaded_imgs = []
        self.valid_lags = None
        self.ids_per_seq_len = {}
        self.object_weights_per_seq_len = {}
        if "weight_zeropoke" in self.config and "include_zeropoke" in self.config:
            self.zeropoke_weight = max(1.,float(self.max_frames) / 5) if self.config["weight_zeropoke"] and self.config["include_zeropoke"] else 1.
        else:
            self.zeropoke_weight = 1.
        # this is the value, which will be the upper bound for all normalized optical flows, when training on variable sequence lengths
        # per default, set to 1 here (max) can be adapted, if necessary, in the subclass of base dataset
        self.flow_cutoff = 1.

        self.valid_h = [self.poke_size, self.config["spatial_size"][0] - self.poke_size]
        self.valid_w = [self.poke_size, self.config["spatial_size"][1] - self.poke_size]

        self.use_flow_for_weights = False




    def __getitem__(self, idx):
        """

        :param idx: The idx is here a tuple, consisting of the actual id and the sampled lag for the flow in the respective iteration
        :return:
        """
        # collect outputs

        data = {}
        transforms = {"color": self._get_color_transforms(), "geometry" : self._get_geometric_transforms()}
        self.color_transfs = self._get_color_transforms() if self.augment else None
        self.geom_transfs = self._get_geometric_transforms() if self.augment else None

        # sample id (in case, sample is enabled)
        if self.var_sequence_length:
            idx = self._get_valid_ids(*idx)
        else:
            idx = self._get_valid_ids(length=None,index=idx)

        sidx = int(np.random.choice(np.flatnonzero(self.datadict["vid"] == self.datadict["vid"][idx[0]]), 1))
        tr_vid = int(np.random.choice(self.datadict["vid"][self.datadict["vid"] != self.datadict["vid"][idx[0]]], 1))
        for i in range(self.max_trials_flow_load):
            self.mask = {}
            try:
                self._get_mask(idx)
                data = {key: self._output_dict[key][0](idx, sample_idx = sidx,
                                                       transforms = chain.from_iterable([transforms[tkey] for tkey in self._output_dict[key][1]]) if len(self._output_dict[key])>1 else None,
                                                       transfer_vid= tr_vid) for key in self.datakeys}
                break
            except FlowError as fe:
                self.logger.error(fe)
                # sample new id and try again
                img_id = int(np.random.choice(np.arange(self.datadict["img_path"].shape[0]),1))
                # don't change lag
                idx = (img_id,idx[1])

        if len(data) == 0:
            raise IOError(f"Errors in flow files loading...tried it {self.max_trials_flow_load} times consecutively without success.")

        return data

    def _get_valid_ids(self,length,index = None):
        """

        :param length: The sequence length (or flow step, depending on whether var_sequence_length is True or False)
        :param index:  The id correspinding to the
        :return:
        """
        # we need to do the following things:
        # take care, that choose one start id from all samples, which have the appropriate flow_magnitude and result in sequences which are within the same video
        if self.var_sequence_length:
            #ids = np.flatnonzero(np.logical_and(self.datadict["flow_range"][:,1]>self.seq_len_T_chunk[length],np.less_equal(np.arange(self.datadict["img_path"].shape[0]) + self.min_seq_length[0] + length*self.subsample_step,self.datadict["seq_end_id"])))
            if length == -1:
                # use maximum sequence length for such cases
                # length = int(np.random.choice(np.arange(self.max_frames),1))
                # in case length == -1: index corresponds to actual sampled length for the regarded batch
                self.outside_length = index
                start_id = int(np.random.choice(self.ids_per_seq_len[self.outside_length], 1))
            else:
                ids = self.ids_per_seq_len[length]
                if self.obj_weighting:
                    start_id = int(np.random.choice(ids, 1, p=self.object_weights_per_seq_len[length]))
                else:
                    start_id = int(np.random.choice(ids, 1))
        else:
            if index == -1:
                length = -1
                if self.obj_weighting:
                    index = int(np.random.choice(np.arange(self.datadict["object_id"].shape[0]),p=self.datadict["weights"],size=1))
                else:
                    index = int(np.random.choice(np.arange(self.datadict["object_id"].shape[0]), p=self.datadict["weights"], size=1))

            max_id_fid = self.sids_per_seq[self.datadict["vid"][index]] + self.datadict["max_fid"][index,self.valid_lags[0]] - 1
            start_id = min(min(index,self.datadict["seq_end_id"][index]-(self.max_frames* self.subsample_step) - 1),max_id_fid)
        return (start_id,length)

    def _get_3d_flow(self, ids, **kwargs):
        flow = self._get_flow(ids)
        flow = convert_flow_2d_to_3d(flow)
        return flow

    def _get_3d_poke(self, ids, **kwargs):
        flow = self._get_poke(ids)
        flow = convert_flow_2d_to_3d(flow)
        return flow

    def _get_edge_image(self, ids, sample_idx, transforms=None, sample=False, use_fb_aug=False, **kwargs):
        imgs = []

        if sample:
            yield_ids = [sample_idx]
        else:
            yield_ids = self._get_yield_ids(ids)
        for i,idx in enumerate(yield_ids):
            img_path = self.datadict["img_path"][idx]
            img = cv2.imread(img_path)
            # image is read in BGR
            img = preprocess_image(img, swap_channels=True)
            img = cv2.resize(
                img, self.config["spatial_size"], cv2.INTER_LINEAR
            )

            # transformations
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gradient = cv2.Sobel(img/255, cv2.CV_64F, 1, 0, ksize=3)
            gradient = self.post_edges(gradient)[0]
            imgs.append(gradient)
            gradient = cv2.Sobel(img/255, cv2.CV_64F, 0, 1, ksize=3)
            gradient = self.post_edges(gradient)[0]
            imgs.append(gradient)
        return torch.stack(imgs, dim=0).squeeze(dim=0)

    def _get_edge_flow(self, ids, **kwargs):
        flow_path = self.datadict["flow_paths"][ids[0], self.valid_lags[0]]
        # debug, this path seems to be erroneous
        # flow_path = "/export/data/ablattma/Datasets/plants/processed_crops/VID_0_3_1024x1024/prediction_3_28.flow.npy"
        try:
            flow = np.load(flow_path)
        except ValueError:
            try:
                flow = np.load(flow_path,allow_pickle=True)
            except Exception as ex:
                print(ex)
                raise FlowError(flow_path)
        except:
            raise FlowError(flow_path)

        dsize = None
        if "spatial_size" in self.config:
            dsize = self.config["spatial_size"]
        elif "resize_factor" in self.config:
            dsize = (
                int(float(flow.shape[1]) / self.config["resize_factor"]),
                int(float(flow.shape[2]) / self.config["resize_factor"]),
            )

        flow = F.interpolate(
            torch.from_numpy(flow).unsqueeze(0), size=dsize, mode="nearest"
        ).squeeze(0)
        if self.config["predict_3D"]:
            flow = convert_flow_2d_to_3d(flow)
        gradient_d1_x, gradient_d1_y, gradient_d2_x, gradient_d2_y = get_flow_gradients(flow)
        all_gradients = [gradient_d1_x,
                         gradient_d1_y,
                         gradient_d2_x,
                         gradient_d2_y]
        return torch.stack(all_gradients, dim=0).squeeze(dim=0)

    def _get_transfer_img(self, ids, transfer_vid,**kwargs):
        imgs=[]
        yield_ids = [int(np.random.choice(np.flatnonzero(self.datadict["vid"] == transfer_vid), 1))]
        for idx in yield_ids:
            img_path = self.datadict["img_path"][idx]
            img = cv2.imread(img_path)
            # image is read in BGR
            img = preprocess_image(img, swap_channels=True)
            if "spatial_size" in self.config:
                img = cv2.resize(
                    img, self.config["spatial_size"], cv2.INTER_LINEAR
                )
            elif "resize_factor" in self.config:
                dsize = (
                    int(float(img.shape[1]) / self.config["resize_factor"]),
                    int(float(img.shape[0]) / self.config["resize_factor"]),
                )
                img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)

            # transformations
            img = self.pre_T(img)
            img = self.post_T(img)
            imgs.append(img)

        return torch.stack(imgs, dim=0).squeeze(dim=0)

    def _compute_mask(self,target_id):
        img = self._get_imgs([], sample_idx=target_id, sample=True)
        if self.z1_normalize:
            img = (img.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
        else:
            img = ((img.permute(1, 2, 0).numpy() + 1.) * 127.5).astype(np.uint8)
        mask = np.zeros(img.shape[:2], np.uint8)
        # rect defines starting background area
        rect = (int(img.shape[1] / self.flow_width_factor), int(self.valid_h[0]), int((self.flow_width_factor - 2) / self.flow_width_factor * img.shape[1]), int(self.valid_h[1] - self.valid_h[0]))
        # initialize background and foreground models
        fgm = np.zeros((1, 65), dtype=np.float64)
        bgm = np.zeros((1, 65), dtype=np.float64)
        # apply grab cut algorithm
        mask2, fgm, bgm = cv2.grabCut(img, mask, rect, fgm, bgm, 5, cv2.GC_INIT_WITH_RECT)
        return mask2

    def _compute_mask_with_flow(self,target_id):
        flow = self._get_flow([target_id])
        amplitude = torch.norm(flow, 2, dim=0)
        amplitude -= amplitude.min()
        amplitude /= amplitude.max()

        # use only such regions where the amplitude is larger than mean + 1 * std
        mask = torch.where(torch.gt(amplitude,amplitude.mean()+amplitude.std()),torch.ones_like(amplitude),torch.zeros_like(amplitude)).numpy().astype(np.bool)
        return mask

    def _get_mask(self,ids):

        if self.filter_flow or self.fancy_aug or (self.flow_weights and self.yield_videos):

            if self.use_flow_for_weights:
                mask_src = self._compute_mask_with_flow(ids[0])
                self.mask.update({"img_start": mask_src})
            else:
                mask_src = self._compute_mask(ids[0])
                self.mask.update({"img_start" : np.where((mask_src == 2) | (mask_src == 0), 0, 1).astype(np.bool)})

        if self.flow_weights:

            yield_ids = self._get_yield_ids(ids)
            tgt_id = yield_ids[-1]
            if self.use_flow_for_weights:
                mask_tgt = self._compute_mask_with_flow(tgt_id)
                self.mask.update({"img_tgt": mask_tgt})
            else:
                mask_tgt = self._compute_mask(tgt_id)
                self.mask.update({"img_tgt": np.where((mask_tgt == 2) | (mask_tgt == 0), 0, 1).astype(np.bool)})

            if self.yield_videos:

                mid_id = int((len(list(yield_ids))+yield_ids[0]) / 2)
                if self.use_flow_for_weights:
                    mask_mid = self._compute_mask_with_flow(mid_id)
                    self.mask.update({"img_mid": mask_mid})
                else:
                    mask_mid = self._compute_mask(mid_id)
                    self.mask.update({"img_mid": np.where((mask_mid == 2) | (mask_mid == 0), 0, 1).astype(np.bool)})



    def _get_yield_ids(self,ids):
        start_id = ids[0]

        if self.yield_videos:
            if ids[-1] == -1:
                if self.var_sequence_length:
                    n_frames = self.min_frames + self.outside_length
                    yield_ids = np.stack([start_id]* n_frames,axis=0).tolist()
                else:
                    yield_ids = np.stack([start_id]* (self.max_frames+1),axis=0).tolist()
            else:
                yield_ids = range(start_id, start_id + (self.min_frames + ids[-1]) * self.subsample_step + 1 ,self.subsample_step) \
                    if self.var_sequence_length else range(start_id, start_id + self.max_frames * self.subsample_step + 1, self.subsample_step)
        else:
            yield_ids = (start_id, start_id + (self.valid_lags[0] + 1) * 5)

        return yield_ids

    def _get_image_series(self, ids, step_width=10, **kwargs):
        all_imgs = []
        for i in range(1, step_width+1):
            new_ids = (ids[0] + i * (1 + self.valid_lags[0]) * 5, ids[1])
            flow = self._get_imgs(new_ids, None)
            all_imgs.append(flow)
        return torch.from_numpy(np.stack(all_imgs, axis=0))

    # grabs a series of images
    def _get_imgs(self, ids, sample_idx, transforms=None, sample=False, use_fb_aug=False, **kwargs):
        imgs = []

        if sample:
            yield_ids = [sample_idx]
        else:
            # avoid generating the entire sequence for the color transformed image
            if transforms is not None and self._get_color_transforms in transforms and not sample:
                yield_ids = [ids[0]]
            else:
                yield_ids = self._get_yield_ids(ids)

        for i,idx in enumerate(yield_ids):
            faug = use_fb_aug and (i == 0 or i == len(yield_ids) - 1)

            if self.imgs_in_ram:
                img = self.loaded_imgs[idx]
            else:
                img_path = self.datadict["img_path"][idx]
                img = cv2.imread(img_path)
                img = preprocess_image(img, swap_channels=True)
                # image is read in BGR
                if self.use_lanczos and self.config["spatial_size"] == 64:
                    img = np.array(Image.fromarray(img).resize(self.config["spatial_size"], resample=Image.LANCZOS))
                else:
                    img = cv2.resize(
                        img, self.config["spatial_size"], cv2.INTER_LINEAR
                    )

            # transformations
            img = self.pre_T(img)
            if transforms is not None:
                for t in transforms:
                    img = t(img)
                if faug:
                    bts = self._get_color_transforms()
                    img_back = img
                    for bt in bts:
                        img_back = bt(img_back)
                    img_back = self.post_T(img_back)
            else:
                if self.color_transfs is not None:
                    for t in self.color_transfs:
                        img = t(img)

                if self.geom_transfs is not None:
                    for t in self.geom_transfs:
                        img = t(img)

            img = self.post_T(img)
            if faug:
                img = torch.where(torch.from_numpy(self.mask["img_start"]).unsqueeze(0),img,img_back)
            imgs.append(img)

        return torch.stack(imgs, dim=0).squeeze(dim=0)

    # extracts pokes as flow patches
    def _get_poke(self, ids, **kwargs):
        seq_len_idx = ids[-1]
        if seq_len_idx == -1:
            # make fake ids to avoid returning zero flow for poke sampling
            fake_ids = (ids[0],10)
            flow = self._get_flow(fake_ids)
        else:
            flow = self._get_flow(ids)
        # compute amplitude
        amplitude = torch.norm(flow[:, self.valid_h[0]:self.valid_h[1], self.valid_w[0]:self.valid_w[1]], 2, dim=0)
        amplitude -= amplitude.min()
        amplitude /= amplitude.max()

        if seq_len_idx == -1:
            # use only very small poke values, this should indicate background values
            amplitude_filt = amplitude
            if self.filter_flow:
                # only consider the part of the mask which corresponds to the region considered in flow
                #amplitude_filt = torch.from_numpy(np.where(self.mask["img_start"][self.valid_h[0]:self.valid_h[1],self.valid_w[0]:self.valid_w[1]], amplitude, np.zeros_like(amplitude)))
                indices_pre = np.nonzero(np.logical_not(self.mask["img_start"][self.valid_h[0]:self.valid_h[1],self.valid_w[0]:self.valid_w[1]]))
                indices = torch.from_numpy(np.stack(indices_pre,axis=-1))
                if indices.shape[0] == 0:
                    indices = torch.lt(amplitude, np.percentile(amplitude.numpy(), 5)).nonzero(as_tuple=False)
            else:
                indices = torch.lt(amplitude, np.percentile(amplitude.numpy(), 5)).nonzero(as_tuple=False)
                #amplitude_filt = amplitude

            std = amplitude_filt.std()
            mean = torch.mean(amplitude_filt)
            indices_mgn = torch.gt(amplitude_filt, mean + (std)).nonzero(as_tuple=False)

            if indices_mgn.shape[0] == 0:
                # if flow is not entirely equally distributed, there should be at least 1 value which is above the mean
                # self.logger.warn("Fallback in Dataloading bacause no values remain after filtering.")
                indices_mgn = torch.gt(amplitude_filt, mean).nonzero(as_tuple=False)

            indices_mgn = indices_mgn + np.asarray([[self.valid_h[0], self.valid_w[0]]], dtype=np.int)
            indices_mgn = (indices_mgn[:, 0], indices_mgn[:, 1])


        else:
            if self.filter_flow:
                # only consider the part of the mask which corresponds to the region considered in flow
                amplitude_filt = torch.from_numpy(np.where(self.mask["img_start"][self.valid_h[0]:self.valid_h[1],self.valid_w[0]:self.valid_w[1]], amplitude, np.zeros_like(amplitude)))
            else:
                amplitude_filt = amplitude

            std = amplitude_filt.std()
            mean = torch.mean(amplitude_filt)
            if self.var_sequence_length:
                amplitude_filt = torch.where(torch.from_numpy(np.logical_and((amplitude_filt > self.seq_len_T_chunk[ids[-1]]).numpy(),(amplitude_filt<self.seq_len_T_chunk[ids[-1]+1]).numpy())),
                                             amplitude_filt, torch.zeros_like(amplitude_filt))

            # compute valid indices by thresholding
            indices = torch.gt(amplitude_filt, mean + (std * 2.0)).nonzero(as_tuple=False)
            if indices.shape[0] == 0:
                indices = torch.gt(amplitude, mean + std).nonzero(as_tuple=False)
                if indices.shape[0] == 0:
                    # if flow is not entirely equally distributed, there should be at least 1 value which is above the mean
                    #self.logger.warn("Fallback in Dataloading bacause no values remain after filtering.")
                    indices = torch.gt(amplitude, mean).nonzero(as_tuple=False)

        indices = indices + np.asarray([[self.valid_h[0], self.valid_w[0]]], dtype=np.int)
        # check if indices is not empty, if so, sample another frame (error is catched in __getitem__())
        if indices.shape[0] == 0:
            raise FlowError(path=[],msg=f"Empty indices array at index {ids[0]}....")

        # shift ids to match size of real flow patch
        indices = (indices[:, 0], indices[:, 1])

        # generate number of pokes
        n_pokes = int(
            np.random.randint(
                1, min(self.config["n_pokes"], int(indices[0].shape[0])) + 1
            )
        )

        if seq_len_idx == -1:
            ids_mgn = np.random.randint(indices_mgn[0].shape[0], size=n_pokes)
            row_ids_mgn = indices_mgn[0][ids_mgn]
            col_ids_mgn = indices_mgn[1][ids_mgn]
        # and generate the actual pokes
        ids = np.random.randint(indices[0].shape[0], size=n_pokes)

        row_ids = indices[0][ids]
        col_ids = indices[1][ids]

        pokes = []
        half_poke_size = int(self.poke_size / 2)
        zeros = torch.zeros_like(flow)
        poke_targets = []
        for n,ids in enumerate(zip(row_ids, col_ids)):
            poke = zeros
            if seq_len_idx == -1:
                poke_target =flow[:,row_ids_mgn[n],col_ids_mgn[n]].unsqueeze(-1).unsqueeze(-1) if self.equal_poke_val else \
                    flow[:,row_ids_mgn[n] - half_poke_size:row_ids_mgn[n] + half_poke_size +1,
                    col_ids_mgn[n] - half_poke_size:col_ids_mgn[n] + half_poke_size +1]
            else:

                poke_target = flow[:,ids[0],ids[1]].unsqueeze(-1).unsqueeze(-1) if self.equal_poke_val else flow[:,
                                                                                                            ids[0] - half_poke_size : ids[0] + half_poke_size + 1,
                                                                                                            ids[1] - half_poke_size : ids[1] + half_poke_size + 1,]

            poke[
            :,
            ids[0] - half_poke_size: ids[0] + half_poke_size + 1,
            ids[1] - half_poke_size: ids[1] + half_poke_size + 1,
            ] = poke_target

            pokes.append(poke)
            loc_and_poke = (ids,poke_target)
            poke_targets.append(loc_and_poke)
        # unsqueeze in case of num_pokes = 1


        if self.flow_weights:
            if self.yield_videos:
                if seq_len_idx == -1:
                    complete_mask = np.ones(self.config["spatial_size"], dtype=np.bool)
                else:
                    complete_mask = np.logical_or(np.logical_or(self.mask["img_tgt"],self.mask["img_start"]), self.mask["img_mid"])
                mask_ids = np.nonzero(complete_mask)
                try:
                    min_h = mask_ids[0].min()
                    max_h = mask_ids[0].max()
                    min_w = mask_ids[1].min()
                    max_w = mask_ids[1].max()
                    weights = np.full(self.mask["img_start"].shape,self.weight_value_bg)
                    weights[min_h:max_h,min_w:max_w] = self.weight_value_flow
                except Exception as e:
                    self.logger.warn(f'Catch exception in "dataset._get_poke()": {e.__class__.__name__}: "{e}". Using full image instead of patch....')
                    weights = np.full(self.mask["img_start"].shape,self.weight_value_bg)
                    weights[self.valid_h[0]:self.valid_h[1],self.valid_w[0]:self.valid_w[1]] = self.weight_value_flow
                #weights = np.where(complete_mask,np.full_like(complete_mask,self.weight_value_flow,dtype=np.float),np.full_like(complete_mask,self.weight_value_bg,dtype=np.float),)
            else:
                weights = np.where(self.mask["img_tgt"],np.full_like(self.mask["img_tgt"],self.weight_value_flow,dtype=np.float),np.full_like(self.mask["img_tgt"],self.weight_value_bg,dtype=np.float),)
            # poke regions get higher weights
            # for poke in pokes:
            #     weights = np.where(((poke**2).sum(0)>0),np.full_like(weights,self.weight_value_poke),weights)

            weights = torch.from_numpy(weights)
            pokes = torch.stack(pokes, dim=0).squeeze(0)
            if "yield_poke_target" in kwargs:
                return pokes, weights, poke_targets
            return pokes, weights
        else:
            pokes = torch.stack(pokes, dim=0).squeeze(0)
            if "yield_poke_target" in kwargs:
                return pokes, poke_targets
            return pokes

    def _get_flow_series(self, ids, step_width=10, **kwargs):
        all_flows = []
        for i in range(1, step_width+1):
            new_ids = (ids[0] + i * (1 + self.valid_lags[0]) * 5, self.valid_lags[0], ids[1])
            flow = self._get_3d_flow(new_ids)
            all_flows.append(flow)
        return torch.from_numpy(np.stack(all_flows, axis=0))


    # extracts entire flow
    def _get_flow(self, ids, **kwargs):
        if self.flow_in_ram:
            flow = torch.from_numpy(self.loaded_flows[ids[0]])
        else:
            flow_path = self.datadict["flow_paths"][ids[0], self.valid_lags[0]]
            # debug, this path seems to be erroneous
            # flow_path = "/export/data/ablattma/Datasets/plants/processed_crops/VID_0_3_1024x1024/prediction_3_28.flow.npy"
            try:
                flow = np.load(flow_path)
            except ValueError:
                try:
                    flow = np.load(flow_path,allow_pickle=True)
                except Exception as ex:
                    print(ex)
                    raise FlowError(flow_path)
            except:
                raise FlowError(flow_path)

            if self.normalize_flows:
                flow = flow / self.flow_norms["max_norm"][self.valid_lags[0]]
            elif not self.normalize_flows and self.scale_poke_to_res:
                # scaling of poke magnitudes to current resolution
                 flow = flow / (flow.shape[1]/self.config["spatial_size"][0])

            dsize = self.config["spatial_size"]
            flow = F.interpolate(
                torch.from_numpy(flow).unsqueeze(0), size=dsize, mode="bilinear",align_corners=True
            ).squeeze(0)

            if ids[-1] == -1:
                flow = torch.zeros_like(flow)

            if self.geom_transfs is not None:
                c1 = Image.fromarray(flow[0].numpy(),mode="F")
                c2 = Image.fromarray(flow[1].numpy(),mode="F")
                for tr in self.geom_transfs:
                    c1 = tr(c1)
                    c2 = tr(c2)

                flow = torch.from_numpy(np.stack([np.array(c1.getdata()).reshape(c1.size[0],c1.size[1]),
                                                  np.array(c2.getdata()).reshape(c2.size[0],c2.size[1])],axis=0)).to(torch.float)

        return flow

    def _get_color_transforms(self):
        # to make sure, the transformations are always coherent within the same sample

        make_trans = bool(np.random.choice(np.arange(2), size=1, p=[1 - self.p_col ,self.p_col]))
        brightness_val = float(np.random.uniform(-self.ab,self.ab,1)) if self.ab > 0. and make_trans else 0.
        contrast_val = float(np.random.uniform(-self.ac, self.ac, 1)) if self.ac > 0. and make_trans else 0.
        hue_val = float(np.random.uniform(-self.ah, 2 * self.ah, 1)) if self.ah > 0. and make_trans else 0.
        saturation_val = 1. + (float(np.random.uniform(-self.a_s,self.a_s)) if self.a_s > 0. and make_trans else 0)

        b_T = partial(FT.adjust_brightness,brightness_factor=1. + brightness_val)
        c_T = partial(FT.adjust_contrast,contrast_factor=1. + contrast_val)
        h_T = partial(FT.adjust_hue, hue_factor=hue_val)
        s_T = partial(FT.adjust_saturation,saturation_factor =saturation_val)

        return [b_T,c_T,h_T,s_T]


    def _get_geometric_transforms(self):
        # to make sure, the transformations are always coherent within the same sample
        make_trans = bool(np.random.choice(np.arange(2),size=1,p=[1-self.p_geom,self.p_geom]))
        rval = float(np.random.uniform(-self.ad,self.ad,1)) if self.ad > 0. and make_trans else 0.
        tval_vert = int(np.random.randint(int(-self.at[0]    * self.config["spatial_size"][1] / 2), int(self.at[0] * self.config["spatial_size"][1] / 2), 1)) if self.at[0] > 0 and make_trans else 0
        tval_hor = int(np.random.randint(int(-self.at[1] * self.config["spatial_size"][0] / 2), int(self.at[1] * self.config["spatial_size"][0] / 2), 1)) if self.at[1] > 0 and make_trans else 0
        a_T = partial(FT.affine,angle=rval,translate=(tval_hor,tval_vert),scale=1.0,shear=0)
        p = partial(FT.pad,padding=(int(self.config["spatial_size"][0] / 2), int(self.config["spatial_size"][1] / 2)),padding_mode="reflect")
        c = partial(FT.center_crop,output_size=self.config["spatial_size"])

        return [p,a_T,c]

    def _get_flip_transform(self):
        flip = bool(np.random.choice([True,False],size=1))
        if flip:
            return FT.vflip
        else:
            return None

    @abstractmethod
    def __len__(self):
        # as len at least once before dataloading, generic checks can be put here
        assert self.valid_lags is not None
        assert self.min_frames is not None
        if self.filter_flow:
            assert self.flow_width_factor is not None, f"If the dataset shall be filtered, the flow width factor has to be set in the constructor of the respective child class of BaseDataset"
            assert isinstance(self.flow_width_factor,int)

        if self.flow_weights:
            assert self.flow_width_factor is not None
        if self.normalize_flows:
            assert self.flow_norms is not None

        if self.flow_in_ram:
            assert len(self.loaded_flows) == self.datadict["flow_paths"].shape[0]

        if self.imgs_in_ram:
            assert len(self.loaded_imgs) == self.datadict["img_path"].shape[0]

        if self.var_sequence_length:
            assert self.normalize_flows
            assert self.yield_videos
            assert len(self.ids_per_seq_len) > 0
            assert len(self.object_weights_per_seq_len) == len(self.ids_per_seq_len)


    @abstractmethod
    def _set_instance_specific_values(self):
        pass

    @abstractmethod
    def get_test_app_images(self) -> dict:
        pass
