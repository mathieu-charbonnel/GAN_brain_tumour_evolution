"""Test script for Time Predictor Network (TPN).

Usage:
    python test_time.py --dataroot <path> --name <exp_name> --model time_predictor \
        --dataset_mode aligned_time
"""
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model


def predict_time(opt):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    predictions = []
    true_times = []
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.forward()
        predictions.append(torch.mean(model.prediction).item())
        true_times.append(model.true_time)

    predictions = torch.tensor(predictions)
    true_times = torch.tensor(true_times, dtype=torch.float)

    loss_l1 = torch.nn.L1Loss()(predictions, true_times)
    loss_mse = torch.nn.MSELoss()(predictions, true_times)
    print("Loss for %s set: L1: %.4f, MSE: %.4f" % (opt.phase, loss_l1, loss_mse))

    return predictions, true_times


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 0
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    predictions, true_times = predict_time(opt)

    for i, (pred, true_t) in enumerate(zip(predictions, true_times)):
        print("Image %d: Predicted %.4f, True time %.4f" % (i, pred, true_t))
