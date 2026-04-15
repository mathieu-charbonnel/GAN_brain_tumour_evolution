import torch

from . import networks
from .base_gan_model import BaseGanModel


class GeaGanModelDM(BaseGanModel):
    """GEA-GAN with deformation model: conditions on a time ratio."""

    def name(self):
        return 'GeaGanModelDM'

    def initialize(self, opt):
        opt.which_model_netG = 'unet_128_TPN'
        super().initialize(opt)

    def _compute_d_channel(self, opt):
        # +1 for the time layer channel
        return opt.input_nc + opt.output_nc + 1

    def set_input(self, input):
        super().set_input(input)
        self.true_time = input['time_ratio'][0]

    def _make_time_tensor(self):
        return torch.ones(self.real_A.shape, device=self.real_A.device) * self.true_time

    def _generate(self):
        time = torch.ones((1, 1), device=self.real_A.device) * self.true_time
        return self.netG(self.real_A, time)

    def _build_fake_ab(self):
        self.true_time_layer = self._make_time_tensor()
        return torch.cat((self.true_time_layer, self.real_A, self.fake_B), 1)

    def _build_real_ab(self):
        return torch.cat((self.true_time_layer, self.real_A, self.real_B), 1)

    def backward_G(self):
        fake_AB = torch.cat((self._make_time_tensor(), self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_sobelL1 = self.criterionL1(self.fake_sobel, self.real_sobel) * self.sobelLambda
        self.loss_G += self.loss_sobelL1

        self.loss_G.backward()
