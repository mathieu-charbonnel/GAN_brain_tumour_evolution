from collections import OrderedDict
from copy import deepcopy

import torch

from . import networks
from .base_gan_model import BaseGanModel
from .time_predictor_model import TimePredictorModel


class GeaGanModelTPN(BaseGanModel):
    """GEA-GAN with Time Prediction Network."""

    def name(self):
        return 'GeaGanModelTPN'

    def initialize(self, opt):
        self.TPN_enabled = bool(opt.TPN)
        if self.TPN_enabled:
            opt.which_model_netG = 'unet_128_TPN'

        super().initialize(opt)

        if self.isTrain and self.TPN_enabled:
            # Store final gamma value and then set it to 0
            self.final_gamma = deepcopy(opt.gamma)
            opt.gamma = 0

            # Initialize m and c to None
            self.update_m = None
            self.update_c = None

            # Setup TPN
            print("\nSetting up TPN\n")
            opt_TPN = deepcopy(opt)
            opt_TPN.model = 'time_predictor'
            opt_TPN.name = opt.TPN
            opt_TPN.ndf = 16
            opt_TPN.display_id = -1
            opt_TPN.isTrain = False
            print("Options TPN: {}\n\n".format(opt_TPN))
            self.TPN = TimePredictorModel()
            self.TPN.initialize(opt_TPN)
            self.TPN.load_network(self.TPN.netDt, 'Dt', epoch_label=200)

    def _compute_d_channel(self, opt):
        d_channel = opt.input_nc + opt.output_nc
        if self.TPN_enabled:
            d_channel += 1  # time layer channel
        return d_channel

    def set_input(self, input):
        super().set_input(input)
        self.diff_map = input['diff_map']
        self.true_time = input['time_period'][0]

    def _make_time_tensor(self):
        return torch.ones(self.real_A.shape, device=self.real_A.device) * self.true_time

    def _generate(self):
        if self.TPN_enabled:
            time = torch.ones((1, 1), device=self.real_A.device) * self.true_time
            return self.netG(self.real_A, time)
        return self.netG(self.real_A)

    def forward(self):
        self.real_A = self.input_A
        self.real_B = self.input_B
        self.fake_B = self._generate()
        if self.TPN_enabled and self.isTrain:
            self.TPN.diff_map = self.diff_map
            self.TPN.forward()
            self.fake_time = self.TPN.prediction

    def test(self):
        self.real_A = self.input_A
        self.real_B = self.input_B
        self.fake_B = self._generate()
        if self.TPN_enabled and self.isTrain:
            self.TPN.diff_map = self.diff_map
            self.TPN.forward()
            self.fake_time = self.TPN.prediction

    def backward_D(self):
        self.fake_sobel = networks.sobelLayer(self.fake_B)

        if self.TPN_enabled:
            self.true_time_layer = self._make_time_tensor()
            fake_AB = self.fake_AB_pool.query(torch.cat((self.true_time_layer, self.real_A, self.fake_B), 1))
        else:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))

        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        self.real_sobel = networks.sobelLayer(self.real_B).detach()

        if self.TPN_enabled:
            real_AB = torch.cat((self.true_time_layer, self.real_A, self.real_B), 1)
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        if self.TPN_enabled:
            fake_AB = torch.cat((self.true_time_layer, self.real_A, self.fake_B), 1)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        if self.TPN_enabled:
            true_time_tensor = torch.ones(self.fake_time.shape, device=self.fake_time.device) * self.true_time
            self.loss_G_TPN = self.criterionL1(true_time_tensor, self.fake_time) * self.opt.gamma
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_TPN
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_sobelL1 = self.criterionL1(self.fake_sobel, self.real_sobel) * self.sobelLambda
        self.loss_G += self.loss_sobelL1

        self.loss_G.backward()

    def get_current_errors(self):
        errors = super().get_current_errors()
        if self.TPN_enabled:
            errors['G_TPN'] = self.loss_G_TPN.item()
        return errors

    def update_current_gamma(self, epoch):
        start_epoch = 1
        end_epoch = 100

        if self.update_m is None and self.update_c is None:
            self.update_m = self.final_gamma / (end_epoch - start_epoch)
            self.update_c = -self.update_m * start_epoch

        if epoch < start_epoch:
            self.opt.gamma = 0
        elif start_epoch < epoch < end_epoch:
            self.opt.gamma = self.update_m * epoch + self.update_c
        else:
            self.opt.gamma = self.final_gamma

        print('gamma = %.7f' % self.opt.gamma)
