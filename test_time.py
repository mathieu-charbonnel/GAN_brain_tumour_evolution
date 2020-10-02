"""Test script for Time Predictor Network (TPN).

Once you have trained your model with train_time.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and print out the results.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and prints out the results.

Example (You need to train models first:
    Test a TimePredictoNetwork model:
        python test_time.py --dataroot #DATASET_LOCATION# --name #EXP_NAME# --model time_predictor --netD time_input --direction AtoB

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
#from util.visualizer import save_images
from util import html
import torch

def predict_time(opt=None, dataset=None, model=None):

    if dataset == None:
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    if model == None:
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Create matrix to hold predictions:
    predictions = torch.zeros(min(opt.num_test, len(dataset)))
    true_times = torch.zeros(len(predictions))
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        predictions[i] = torch.mean(model.prediction).item()
        true_times[i] = model.true_time

    L1 = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()
    loss_l1 = L1(predictions, true_times)
    loss_mse = MSE(predictions, true_times)

    print("Loss for {} set: L1: {}, MSE: {}".format(opt.phase, loss_l1, loss_mse))

    return predictions, true_times

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    predictions, true_times = predict_time(opt)

    for i, (pred, true_t) in enumerate(zip(predictions, true_times)):
        print("Image {}: Predicted {}, True time {}".format(i, pred, true_t))
