from collections import OrderedDict

import torch

import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class BaseGanModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.batchSize = opt.batchSize
        self.fineSize = opt.fineSize

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize, opt.fineSize)

        if self.opt.rise_sobelLoss:
            self.sobelLambda = 0
        else:
            self.sobelLambda = self.opt.lambda_sobel

        which_netG = opt.which_model_netG
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      which_netG, opt.norm, opt.use_dropout, self.gpu_ids)

        if self.isTrain:
            self.D_channel = self._compute_d_channel(opt)
            use_sigmoid = opt.no_lsgan

            self.netD = networks.define_D(self.D_channel, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
        if not self.isTrain:
            self.netG.eval()

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            if self.opt.labelSmooth:
                self.criterionGAN = networks.GANLoss_smooth(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def _compute_d_channel(self, opt):
        return opt.input_nc + opt.output_nc

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def _generate(self):
        return self.netG.forward(self.real_A)

    def forward(self):
        self.real_A = self.input_A
        self.fake_B = self._generate()
        self.real_B = self.input_B

    def test(self):
        self.real_A = self.input_A
        self.fake_B = self._generate()
        self.real_B = self.input_B

    def get_image_paths(self):
        return self.image_paths

    def _build_fake_ab(self):
        return torch.cat((self.real_A, self.fake_B), 1)

    def _build_real_ab(self):
        return torch.cat((self.real_A, self.real_B), 1)

    def backward_D(self):
        self.fake_sobel = networks.sobelLayer(self.fake_B)
        fake_AB = self.fake_AB_pool.query(self._build_fake_ab())

        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        self.real_sobel = networks.sobelLayer(self.real_B).detach()
        real_AB = self._build_real_ab()

        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = self._build_fake_ab()
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_sobelL1 = self.criterionL1(self.fake_sobel, self.real_sobel) * self.sobelLambda
        self.loss_G += self.loss_sobelL1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('G_sobelL1', self.loss_sobelL1.data.item()),
                            ('D_GAN', self.loss_D.data.item())])

    def get_current_visuals(self):
        real_A = util.tensor2array(self.real_A.data)
        fake_B = util.tensor2array(self.fake_B.data)
        real_B = util.tensor2array(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def get_current_img(self):
        real_A = util.tensor2im(self.real_A.data)
        real_A_co = util.tensor2im_co(self.real_A.data)
        real_A_sa = util.tensor2im_sa(self.real_A.data)

        fake_B = util.tensor2im(self.fake_B.data)
        fake_B_co = util.tensor2im_co(self.fake_B.data)
        fake_B_sa = util.tensor2im_sa(self.fake_B.data)

        real_B = util.tensor2im(self.real_B.data)
        real_B_co = util.tensor2im_co(self.real_B.data)
        real_B_sa = util.tensor2im_sa(self.real_B.data)

        return OrderedDict([
            ('real_A', real_A), ('real_A_co', real_A_co), ('real_A_sa', real_A_sa),
            ('fake_B', fake_B), ('fake_B_co', fake_B_co), ('fake_B_sa', fake_B_sa),
            ('real_B', real_B), ('real_B_co', real_B_co), ('real_B_sa', real_B_sa)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_sobel_lambda(self, epochNum):
        self.sobelLambda = self.opt.lambda_sobel / 20 * epochNum
        print('update sobel lambda: %f' % (self.sobelLambda))
