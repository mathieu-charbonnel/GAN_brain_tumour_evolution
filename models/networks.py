import functools
import random
from typing import List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def gaussian(ins: torch.Tensor, mean: float, stddev: float) -> torch.Tensor:
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1 or classname.find('InstanceNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type: str) -> Type[nn.Module]:
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm3d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm3d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net: nn.Module, init_type: str = 'normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=1)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    return init_func


def init_net(net: nn.Module, init_type: str = 'normal', gpu_ids: Optional[List[int]] = None) -> nn.Module:
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.cuda(gpu_ids[0])
    init_weights(net, init_type=init_type)
    net.apply(init_weights(init_type))
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert torch.cuda.is_available()

    if which_model_netG == 'unet_32':
        netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128_TPN':
        netG = UnetGeneratorTPN(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)

    init_type = 'normal'

    return init_net(netG, init_type, gpu_ids)




def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert torch.cuda.is_available()
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_Dt(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []
    norm_layer = get_norm_layer(norm_type=norm)
    net = TimeDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    return init_net(net, init_type, gpu_ids)



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor.requires_grad_(False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor.requires_grad_(False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class GANLoss_smooth(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real, smooth):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label + smooth * 0.5 - 0.3)
                self.real_label_var = real_tensor.requires_grad_(False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label + smooth * 0.3)
                self.fake_label_var = fake_tensor.requires_grad_(False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        a = random.uniform(0, 1)
        target_tensor = self.get_target_tensor(input, target_is_real, a)
        return self.loss(input, target_tensor)




def create3DsobelFilter():
    num_1, num_2, num_3 = np.zeros((3, 3))
    num_1 = [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
    num_2 = [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    num_3 = [[-1., -2., -1.],
             [-2., -4., -2.],
             [-1., -2., -1.]]
    sobel_filter = np.zeros((3, 1, 3, 3, 3))

    sobel_filter[0, 0, 0, :, :] = num_1
    sobel_filter[0, 0, 1, :, :] = num_2
    sobel_filter[0, 0, 2, :, :] = num_3
    sobel_filter[1, 0, :, 0, :] = num_1
    sobel_filter[1, 0, :, 1, :] = num_2
    sobel_filter[1, 0, :, 2, :] = num_3
    sobel_filter[2, 0, :, :, 0] = num_1
    sobel_filter[2, 0, :, :, 1] = num_2
    sobel_filter[2, 0, :, :, 2] = num_3

    return torch.from_numpy(sobel_filter).type(torch.cuda.FloatTensor)



def sobelLayer(input):
    pad = nn.ConstantPad3d((1, 1, 1, 1, 1, 1), -1)
    kernel = create3DsobelFilter()
    act = nn.Tanh()
    padded = pad(input)
    fake_sobel = F.conv3d(padded, kernel, padding=0, groups=1) / 4
    n, c, h, w, l = fake_sobel.size()
    fake = torch.norm(fake_sobel, 2, 1, True) / c * 3
    fake_out = act(fake) * 2 - 1

    return fake_out




class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=None):
        super().__init__()
        self.gpu_ids = gpu_ids or []


        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):

        return self.model(input)





class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super().__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True, track_running_stats=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True, track_running_stats=True)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up


        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


class UnetGeneratorTPN(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=None):
        super().__init__()
        self.gpu_ids = gpu_ids or []


        unet_block = UnetSkipConnectionBlockTPN(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlockTPN(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlockTPN(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockTPN(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockTPN(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockTPN(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input, time):

        return self.model(input, time)


class UnetSkipConnectionBlockTPN(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True, track_running_stats=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True, track_running_stats=True)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:#add TPN
            upconv = nn.ConvTranspose3d(inner_nc+1, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:#add TPN
            upconv = nn.ConvTranspose3d(inner_nc * 2+1, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up


        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)


    def forward(self, x, time):
        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1, time)
            return self.up(x2)
        elif self.innermost:
            x1 = self.down(x)
            x1_and_time = torch.cat([time.expand(1, 1, x1.shape[2], x1.shape[3], x1.shape[4]), x1], 1)
            x2 = self.up(x1_and_time)
            return torch.cat([x2, x], 1)
        else:
            x1 = self.down(x)
            x2 = self.submodule(x1, time)
            x2_and_time = torch.cat([time.expand(1, 1, x2.shape[2], x2.shape[3], x2.shape[4]), x2], 1)
            return torch.cat([self.up(x2_and_time), x], 1)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=None):
        super().__init__()
        self.gpu_ids = gpu_ids or []

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        input_conv = nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        sequence = [
            input_conv,
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            intermediate_conv = nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=kw, stride=2, padding=padw)
            sequence += [
                intermediate_conv,

                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        intermediate_conv2 = nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)
        sequence += [
            intermediate_conv2,

            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        last_conv = nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

        sequence += [last_conv]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Flatten(nn.Module):
    def flatten(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

    def forward(self, x):
        return self.flatten(x)


class TimeDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm3d
        else:
            use_bias = norm_layer != nn.InstanceNorm3d

        self.net = [


            nn.Conv3d(input_nc, ndf, kernel_size=4, stride=1, bias=use_bias),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3),
            norm_layer(ndf),


            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=1, bias=use_bias),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3),
            norm_layer(ndf * 2),


            nn.Conv3d(ndf * 2, ndf * 3, kernel_size=4, stride=1, bias=use_bias),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            norm_layer(ndf * 3),


            Flatten(),


            nn.Linear(ndf * 192, 50),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(50, 10),
            nn.ReLU(),

            nn.Linear(10, 1),
            nn.ReLU(),

            ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input.float())
