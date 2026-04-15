import os.path
import random

import numpy as np
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from skimage import io


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert opt.resize_or_crop == 'resize_and_crop'
        self.transform = torch.from_numpy

    def _load_and_split(self, index):
        AB_path = self.AB_paths[index]
        img0 = io.imread(AB_path, plugin='simpleitk')

        AB = np.zeros((1, self.opt.fineSize * 2, self.opt.fineSize, self.opt.fineSize))
        AB[0, 0:self.opt.fineSize, :, :] = img0[0:self.opt.fineSize, :, :]
        AB[0, self.opt.fineSize:self.opt.fineSize * 2, :, :] = img0[self.opt.fineSize:(self.opt.fineSize * 2), :, :]

        AB = self.transform(AB)

        A = AB[:, :self.opt.fineSize, :, :]
        B = AB[:, self.opt.fineSize:self.opt.fineSize * 2, :, :]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(3) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(3, idx)
            B = B.index_select(3, idx)

        return A, B, AB_path

    def _augment_result(self, result, A, B, AB_path):
        return result

    def __getitem__(self, index):
        A, B, AB_path = self._load_and_split(index)
        result = {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        return self._augment_result(result, A, B, AB_path)

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


class AlignedDatasetDM(AlignedDataset):
    def _augment_result(self, result, A, B, AB_path):
        p = (AB_path.split('_')[-1])[:-5]
        result['time_ratio'] = float(p)
        return result

    def name(self):
        return 'AlignedDatasetDM'


class AlignedDatasetTPN(AlignedDataset):
    def _augment_result(self, result, A, B, AB_path):
        result['diff_map'] = torch.abs(A - B)
        result['time_period'] = int(AB_path.split('_')[-1].split('.')[0][:-1])
        return result

    def name(self):
        return 'AlignedDatasetTPN'


class AlignedDatasetTime(AlignedDataset):
    def _augment_result(self, result, A, B, AB_path):
        diff_map = torch.abs(A - B)
        time_period = int(AB_path.split('_')[-1].split('.')[0][:-1])
        return {'diff_map': diff_map, 'time_period': time_period, 'diff_map_paths': AB_path}

    def name(self):
        return 'AlignedDatasetTime'


class TestAlignedDataset(AlignedDataset):
    def _load_and_split(self, index):
        AB_path = self.AB_paths[index]
        img0 = io.imread(AB_path, plugin='simpleitk')
        img0 = img0.reshape((2 * self.opt.fineSize, self.opt.fineSize, self.opt.fineSize))

        AB = np.zeros((1, self.opt.fineSize * 2, self.opt.fineSize, self.opt.fineSize))
        AB[0, 0:self.opt.fineSize, :, :] = img0[0:self.opt.fineSize, :, :]
        AB[0, self.opt.fineSize:self.opt.fineSize * 2, :, :] = img0[self.opt.fineSize:(self.opt.fineSize * 2), :, :]

        AB = self.transform(AB)

        A = AB[:, :self.opt.fineSize, :, :]
        B = AB[:, self.opt.fineSize:self.opt.fineSize * 2, :, :]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(3) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(3, idx)
            B = B.index_select(3, idx)

        return A, B, AB_path


class TestAlignedDatasetDM(TestAlignedDataset, AlignedDatasetDM):
    def name(self):
        return 'AlignedDatasetDM'


class TestAlignedDatasetTPN(TestAlignedDataset, AlignedDatasetTPN):
    def name(self):
        return 'AlignedDatasetTPN'
