
from . import networks_time as networks_new
from copy import deepcopy
from .models import create_model
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks as networks


class TimePredictorModel(BaseModel):
    def name(self):
        return 'TimePredictorModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.batchSize = opt.batchSize
        self.fineSize = opt.fineSize
        # define tensors
#        self.input_diff_map = self.Tensor(opt.batchSize, opt.input_nc,
#                                   opt.fineSize, opt.fineSize, opt.fineSize)
        # load/define networks
#        self.Dtype = opt.which_model_netD
        input_channel_size = opt.input_nc

        self.netDt = networks_new.define_Dt(input_channel_size, opt.ndf, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_D = torch.optim.Adam(self.netDt.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))



            print('---------- Networks initialized -------------')
            networks_new.print_network(self.netDt)
            print('-----------------------------------------------')

    def set_input(self, input):
        self.diff_map = input['diff_map']
        self.true_time = input['time_period'][0]
        self.diff_map_paths = (input['diff_map_paths'])

    def forward(self):
        self.prediction = self.netDt(self.diff_map)

    # get image paths
    def get_image_paths(self):
        return self.diff_map_paths

    def backward_D(self):

        # Calculate Loss for D
        # Store scalar in a 1x1 matrix so that we can use it for the loss
        true_time_tensor = torch.ones(self.prediction.shape) * self.true_time
        self.loss_D_real = self.criterionL2(true_time_tensor, self.prediction.cpu())
        self.loss_D = self.loss_D_real
        self.loss_D.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()

    def get_current_errors(self):
        #visualizer plot function does not handle less than 2 losses right now
        return OrderedDict([('loss_D', self.loss_D.item()),('test_loss0', 0)])

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, label):
        self.save_network_time(self.netDt, 'Dt', label, self.gpu_ids)
