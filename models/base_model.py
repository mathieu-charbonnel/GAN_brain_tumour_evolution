import os
from typing import Dict, List, Optional

import torch


class BaseModel:
    def name(self) -> str:
        return 'BaseModel'

    def initialize(self, opt) -> None:
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.optimizers: List[torch.optim.Optimizer] = []
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    def set_input(self, input) -> None:
        self.input = input

    def forward(self) -> None:
        pass

    def test(self) -> None:
        pass

    def get_image_paths(self) -> Optional[str]:
        pass

    def optimize_parameters(self) -> None:
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self) -> Dict:
        return {}

    def save(self, label) -> None:
        pass

    def save_network(self, network: torch.nn.Module, network_label: str, epoch_label, gpu_ids: List[int]) -> None:
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def load_network(self, network: torch.nn.Module, network_label: str, epoch_label) -> None:
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self) -> None:
        pass
