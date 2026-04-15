import torch

from . import networks
from .base_gan_model import BaseGanModel


class DeaGanModel(BaseGanModel):
    """DEA-GAN: concatenates sobel edges into the discriminator input."""

    def name(self):
        return 'DeaGanModel'

    def _compute_d_channel(self, opt):
        # +1 for the sobel edge channel
        return opt.input_nc + opt.output_nc + 1

    def _build_fake_ab(self):
        return torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1)

    def _build_real_ab(self):
        return torch.cat((self.real_A, self.real_B, self.real_sobel), 1)

    def backward_D(self):
        self.fake_sobel = networks.sobelLayer(self.fake_B)
        fake_AB = self.fake_AB_pool.query(self._build_fake_ab())

        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        self.real_sobel = networks.sobelLayer(self.real_B).detach()
        real_AB = self.fake_AB_pool.query(self._build_real_ab())

        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # DEA-GAN queries the pool again in backward_G
        fake_AB = self.fake_AB_pool.query(self._build_fake_ab())
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_sobelL1 = self.criterionL1(self.fake_sobel, self.real_sobel) * self.sobelLambda
        self.loss_G += self.loss_sobelL1

        self.loss_G.backward()
