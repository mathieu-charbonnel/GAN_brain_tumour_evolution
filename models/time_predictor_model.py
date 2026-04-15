from collections import OrderedDict

import torch

from .base_model import BaseModel
from . import networks


class TimePredictorModel(BaseModel):
    def name(self):
        return 'TimePredictorModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.batchSize = opt.batchSize
        self.fineSize = opt.fineSize
        input_channel_size = opt.input_nc

        self.netDt = networks.define_Dt(input_channel_size, opt.ndf, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_D = torch.optim.Adam(self.netDt.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netDt)
            print('-----------------------------------------------')

    def set_input(self, input):
        self.diff_map = input['diff_map']
        self.true_time = input['time_period'][0]
        self.diff_map_paths = input['diff_map_paths']

    def forward(self):
        self.prediction = self.netDt(self.diff_map)

    def get_image_paths(self):
        return self.diff_map_paths

    def backward_D(self):
        true_time_tensor = torch.ones(self.prediction.shape) * self.true_time
        self.loss_D_real = self.criterionL2(true_time_tensor, self.prediction.cpu())
        self.loss_D = self.loss_D_real
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        return OrderedDict([('loss_D', self.loss_D.item()), ('test_loss0', 0)])

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, label):
        self.save_network(self.netDt, 'Dt', label, self.gpu_ids)
