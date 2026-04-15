import os
from typing import List, Union

import numpy as np
import SimpleITK as sitk
import scipy.io as sio
import torch


def tensor2im(image_tensor: torch.Tensor, imtype: type = np.uint8) -> np.ndarray:
    image_numpy = image_tensor.cpu().float().numpy()[0, 0, 60:66, :, :]
    image_numpy = (image_numpy + 1) / 2.00 * 255.0
    return image_numpy.astype(imtype)


def tensor2im_co(image_tensor: torch.Tensor, imtype: type = np.uint8) -> np.ndarray:
    image_numpy = image_tensor.cpu().float().numpy()[0, 0, :, 60:66, :]
    image_numpy = np.transpose(image_numpy, (1, 0, 2))
    image_numpy = (image_numpy + 1) / 2.00 * 255.0
    image_numpy = np.flip(image_numpy, axis=1)
    return image_numpy.astype(imtype)


def tensor2im_sa(image_tensor: torch.Tensor, imtype: type = np.uint8) -> np.ndarray:
    image_numpy = image_tensor.cpu().float().numpy()[0, 0, :, :, 60:66]
    image_numpy = np.transpose(image_numpy, (2, 1, 0))
    image_numpy = (image_numpy + 1) / 2.00 * 255.0
    image_numpy = np.rot90(image_numpy, axes=(1, 2))
    return image_numpy.astype(imtype)


def tensor2array(image_tensor: torch.Tensor) -> np.ndarray:
    image_numpy = image_tensor.cpu().numpy()
    return image_numpy[0, 0, :, :, :]


def tensor2array_labels(image_tensor: torch.Tensor) -> np.ndarray:
    image_numpy = image_tensor.cpu().numpy()
    return image_numpy[0, :, :, :, :]


def diagnose_network(net: torch.nn.Module, name: str = 'network') -> None:
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy: np.ndarray, image_path: str) -> None:
    sav_img = sitk.GetImageFromArray(image_numpy[:, :, :])
    sitk.WriteImage(sav_img, image_path)


def save_labels(image_numpy: np.ndarray, image_path: str) -> None:
    sio.savemat(image_path, {'label': image_numpy})


def print_numpy(x: np.ndarray, val: bool = True, shp: bool = False) -> None:
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths: Union[List[str], str]) -> None:
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
