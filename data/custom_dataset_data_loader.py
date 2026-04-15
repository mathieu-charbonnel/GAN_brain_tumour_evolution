import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.aligned_dataset import (
    AlignedDataset,
    AlignedDatasetDM,
    AlignedDatasetTPN,
    AlignedDatasetTime,
)

_DATASET_MAP = {
    'aligned': AlignedDataset,
    'aligned_time': AlignedDatasetTime,
    'aligned_TPN': AlignedDatasetTPN,
    'aligned_DM': AlignedDatasetDM,
}


def CreateDataset(opt):
    dataset_cls = _DATASET_MAP.get(opt.dataset_mode)
    if dataset_cls is None:
        raise ValueError("Dataset mode [%s] not recognized." % opt.dataset_mode)
    dataset = dataset_cls(opt)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt):
        super().__init__(opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
