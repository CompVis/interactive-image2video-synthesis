import numpy as np
from torch.utils.data import BatchSampler,RandomSampler,SequentialSampler, WeightedRandomSampler

from data.base_dataset import BaseDataset
from data.flow_dataset import PlantDataset

class SequenceSampler(BatchSampler):
    def __init__(self, dataset:BaseDataset, batch_size, shuffle,  drop_last):
        assert isinstance(dataset, BaseDataset), "The used dataset in Sequence Sampler must inherit from BaseDataset"
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)


        self.dataset = dataset
        #self.max_lag = self.dataset.datadict["flow_paths"].shape[1]


    def __iter__(self):
        batch = []

        # sample sequence length
        lag = int(np.random.choice(self.dataset.valid_lags, 1))

        for idx in self.sampler:
            batch.append((idx, lag))
            if len(batch) == self.batch_size:
                yield batch
                batch = []

                # sample sequence length
                lag = int(np.random.choice(self.dataset.valid_lags, 1))

        if len(batch) > 0 and not self.drop_last:
            yield batch


class FixedLengthSampler(BatchSampler):

    def __init__(self, dataset:PlantDataset,batch_size,shuffle,drop_last, weighting, zero_poke,zero_poke_amount=None):
        if shuffle:
            if weighting:
                sampler = WeightedRandomSampler(weights=dataset.datadict["weights"],num_samples=len(dataset))
            else:
                sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)
        self.shuffle = shuffle
        self.dataset = dataset
        self.zero_poke = zero_poke
        self.zero_poke_amount = zero_poke_amount
        if self.zero_poke:
            assert self.zero_poke_amount is not None


    def __iter__(self):
        batch = []
        if self.zero_poke:
            # sample a certain proportion to be zero pokes
            zero_poke_ids = np.random.choice(np.arange(self.dataset.__len__()),size=int(self.dataset.__len__()/ self.zero_poke_amount),replace=False).tolist()
            self.dataset.logger.info(f"Sampling {len(zero_poke_ids)} zeropokes for next epoch")
        else:
            zero_poke_ids = []

        for idx in self.sampler:
            if idx in zero_poke_ids:
                batch.append(-1)
            else:
                batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []


        if len(batch) > 0 and not self.drop_last:
            yield batch



class SequenceLengthSampler(BatchSampler):
    def __init__(self, dataset:BaseDataset, batch_size, shuffle,  drop_last, n_frames=None, zero_poke = False,):
        assert isinstance(dataset, BaseDataset), "The used dataset in Sequence Sampler must inherit from BaseDataset"
        assert dataset.var_sequence_length and dataset.yield_videos, "The dataset has to be run in sequence mode and has to output variable sequence lengths"
        sampler = SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = dataset
        self.shuffle = shuffle
        if n_frames is not None:
            assert n_frames >= self.dataset.min_frames and n_frames <=(self.dataset.min_frames + self.dataset.max_frames)
            self.n_frames = (n_frames-self.dataset.min_frames)
        else:
            self.n_frames = n_frames
        self.start_n_frames = -1 if zero_poke else 0
        if zero_poke:
            if self.dataset.train:
                self.len_p = np.asarray([self.dataset.zeropoke_weight] + [1.] * self.dataset.max_frames)
            else:
                self.len_p = np.asarray([1.] * (self.dataset.max_frames + 1))
        else:
            self.len_p = np.asarray([1.] * self.dataset.max_frames)

        if self.dataset.longest_seq_weight != None and self.dataset.train:
            self.len_p[-1] = self.dataset.longest_seq_weight
            if zero_poke:
                # to keep sufficient outside pokes for the model to learn foreground and background
                self.len_p[0] = self.dataset.longest_seq_weight / 2
        self.len_p = self.len_p /self.len_p.sum()

    def __iter__(self):
        batch = []

        # sample sequence length
        if self.shuffle:
            # -1 corresponds to
            n_frames = int(np.random.choice(np.arange(self.start_n_frames,self.dataset.max_frames), 1, p=self.len_p))

        else:
            last_n = self.start_n_frames
            n_frames = last_n

        if n_frames == -1:
            n_frames_actual = int(np.random.choice(np.arange(self.dataset.max_frames), 1))
            appended = (n_frames, n_frames_actual)
        else:
            appended = (n_frames, None)
        for idx in self.sampler:
            appended = (appended[0] if self.n_frames is None else self.n_frames,appended[1])
            batch.append(appended)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

                # sample sequence length
                if self.shuffle:
                    n_frames = int(np.random.choice(np.arange(self.start_n_frames,self.dataset.max_frames), 1,p=self.len_p))
                else:
                    n_frames = last_n+1 if last_n<self.dataset.max_frames-1 else self.start_n_frames
                    last_n = n_frames

                if n_frames == -1:
                    n_frames_actual = int(np.random.choice(np.arange(self.dataset.max_frames), 1))
                    appended = (n_frames, n_frames_actual)
                else:
                    appended = (n_frames, None)

        if len(batch) > 0 and not self.drop_last:
            yield batch