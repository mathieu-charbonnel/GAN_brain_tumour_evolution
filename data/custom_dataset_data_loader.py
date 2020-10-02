import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'aligned_time':
        from data.aligned_dataset_time import AlignedDatasetTime
        dataset = AlignedDatasetTime()
    elif opt.dataset_mode == 'aligned_TPN':
        from data.aligned_dataset_TPN import AlignedDatasetTPN
        dataset = AlignedDatasetTPN()
    elif opt.dataset_mode == 'aligned_DM':
        from data.aligned_dataset_DM import AlignedDatasetDM
        dataset = AlignedDatasetDM()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
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
