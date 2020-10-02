import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_time as networks
from .time_predictor_model import TimePredictorModel
from copy import deepcopy


class gea_ganModelTPN(BaseModel):
    def name(self):
        return 'gea_ganModelTPN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.batchSize = opt.batchSize
        self.fineSize = opt.fineSize


        #enable TPN
        if opt.TPN:
            self.TPN_enabled = True
        else:
            self.TPN_enabled = False

        if self.TPN_enabled:
            opt.which_model_netG = 'unet_128_TPN'




        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize, opt.fineSize)



        if self.opt.rise_sobelLoss:
            self.sobelLambda = 0
        else:
            self.sobelLambda = self.opt.lambda_sobel


        # load/define networks


        which_netG = opt.which_model_netG
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      which_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:

            self.D_channel = opt.input_nc + opt.output_nc
            use_sigmoid = opt.no_lsgan
            if self.TPN_enabled:
                self.D_channel +=1 # Additional Channel for Time Input



            self.netD = networks.define_D(self.D_channel, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

            if self.TPN_enabled:
                self.loss_names = ['G_GAN', 'G_L1', 'G_TPN', 'D_real', 'D_fake']

                # Store final gamma value and then set it to 0
                self.final_gamma = deepcopy(opt.gamma)
                opt.gamma = 0

                # Initiliaze m and c to None
                self.update_m = None
                self.update_c = None

                # Setup TPN if set to True
                print("\nSetting up TPN\n")
                opt_TPN = deepcopy(opt) # copy train options and change later
                opt_TPN.model = 'time_predictor'
                opt_TPN.name = opt.TPN
                #opt_TPN.netD = 'time_input'
                opt_TPN.ndf = 16 # Change depending on the ndf size used with the TPN model specified
                # hard-code some parameters for TPN test phase
                opt_TPN.display_id = -1   # no visdom display;
                opt_TPN.isTrain = False
                print("Options TPN: {}\n\n".format(opt_TPN))
                self.TPN = TimePredictorModel()
                self.TPN.initialize(opt_TPN)
                self.TPN.load_network(self.TPN.netDt, 'Dt', epoch_label=200)



        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
        if not self.isTrain:
            self.netG.eval()

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            if self.opt.labelSmooth:
                self.criterionGAN = networks.GANLoss_smooth(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()


            # initialize optimizers

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.diff_map = input['diff_map']
        self.true_time = input['time_period'][0]

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if self.TPN_enabled:
            self.fake_B = self.netG(self.real_A, torch.ones((1,1)) * self.true_time) # Pass the image and time

            if self.isTrain:
                # Predict the time between real image A and generated image B
                self.TPN.diff_map = self.diff_map
                self.TPN.forward()
                self.fake_time = self.TPN.prediction
        else:
            self.fake_B = self.netG(self.real_A)  # G(A)



    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if self.TPN_enabled:
            self.fake_B = self.netG(self.real_A, torch.ones((1,1)) * self.true_time) # Pass the image and time

            if self.isTrain:
                # Predict the time between real image A and generated image B
                self.TPN.diff_map = self.diff_map
                self.TPN.forward()
                self.fake_time = self.TPN.prediction
        else:
            self.fake_B = self.netG.forward(self.real_A)  # G(A)



    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):

        # Fake; stop backprop to the generator by detaching fake_B

        self.fake_sobel = networks.sobelLayer(self.fake_B)

        if self.TPN_enabled:
            self.true_time_layer = (torch.ones(self.real_A.shape) * self.true_time)
            #fake_AB = self.fake_AB_pool.query(torch.cat((self.true_time_layer,self.real_A, self.fake_B), 1))
            fake_AB = self.fake_AB_pool.query(torch.cat((self.true_time_layer.cuda(),self.real_A, self.fake_B), 1))
            # we use conditional GANs with TPN; we need to feed both time, input and output to the discriminator
        else:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))

        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)


        # Real
        self.real_sobel = networks.sobelLayer(self.real_B).detach()

        if self.TPN_enabled:
            #real_AB = torch.cat((self.true_time_layer, self.real_A, self.real_B), 1)
            real_AB = torch.cat((self.true_time_layer.cuda(), self.real_A, self.real_B), 1)
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)



        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()



    def backward_G(self):

        # First, G(A) should fake the discriminator
        if self.TPN_enabled:
            fake_AB = torch.cat((self.true_time_layer.cuda(), self.real_A, self.fake_B), 1)
            #real_AB = torch.cat((self.true_time_layer, self.real_A, self.real_B), 1)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)



        # Second, G(A) = B

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        # TPN Loss
        if self.TPN_enabled:
            true_time_tensor = torch.ones(self.fake_time.shape) * self.true_time
            self.loss_G_TPN = self.criterionL1(true_time_tensor.cuda(), self.fake_time.cuda()) * self.opt.gamma
            self.loss_G_TPN.cuda()
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_TPN
            self.loss_G.cuda()
        else:
            # combine loss and calculate gradients
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

        '''return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                        ('G_L1', self.loss_G_L1.data[0]),
                        ('G_sobelL1', self.loss_sobelL1.data[0]),
                        ('D_GAN', self.loss_D.data[0])
                        ])'''
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                        ('G_L1', self.loss_G_L1.data.item()),
                        ('G_sobelL1', self.loss_sobelL1.data.item()),
                        ('D_GAN', self.loss_D.data.item()),
                        ('G_TPN', self.loss_G_TPN.item())
                        ])


    def get_current_visuals(self):
        real_A = util.tensor2array(self.real_A.data)
        fake_B = util.tensor2array(self.fake_B.data)
        real_B = util.tensor2array(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def get_current_img(self):
        real_A = util.tensor2im(self.real_A.data)
        real_A_co=util.tensor2im_co(self.real_A.data)
        real_A_sa=util.tensor2im_sa(self.real_A.data)

        fake_B = util.tensor2im(self.fake_B.data)
        fake_B_co = util.tensor2im_co(self.fake_B.data)
        fake_B_sa = util.tensor2im_sa(self.fake_B.data)

        real_B = util.tensor2im(self.real_B.data)
        real_B_co = util.tensor2im_co(self.real_B.data)
        real_B_sa = util.tensor2im_sa(self.real_B.data)

        return OrderedDict([('real_A', real_A), ('real_A_co', real_A_co), ('real_A_sa', real_A_sa), ('fake_B', fake_B),('fake_B_co', fake_B_co),('fake_B_sa', fake_B_sa), ('real_B', real_B), ('real_B_co', real_B_co), ('real_B_sa', real_B_sa)])



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
        self.sobelLambda = self.opt.lambda_sobel/20*epochNum
        print('update sobel lambda: %f' % (self.sobelLambda))

    def update_current_gamma(self, epoch):
        ''' Update gamma value for TPN from opt, depending on the epoch '''
        #start_epoch = 50
        start_epoch=1
        end_epoch = 100

        # Values should be None only at the first call
        if self.update_m == None and self.update_c == None:
            self.update_m = self.final_gamma / (end_epoch - start_epoch)
            self.update_c = -self.update_m * start_epoch

        if epoch < start_epoch:
            self.opt.gamma = 0
        elif start_epoch < epoch < end_epoch:
            # Linearly update gamma
            self.opt.gamma = self.update_m * epoch + self.update_c
        else: # epoch > end_epoch
            self.opt.gamma = self.final_gamma

        print('gamma = %.7f' % self.opt.gamma)
