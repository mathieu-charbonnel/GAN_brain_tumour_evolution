import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        return 'BaseDataset'

    def initialize(self, opt) -> None:
        pass

