import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

    def name(self) -> str:
        return 'BaseDataset'
