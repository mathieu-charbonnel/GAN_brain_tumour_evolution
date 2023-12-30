# Ea-GANs
This repository is written by myself Mathieu Charbonnel for testing Ea GANS on a private dataset, as well as variants of gea-GAN and dea-GAN.
The code is based on the implementation of [Ea-GANs: Edge-Aware Generative Adversarial Networks for Cross-Modality MR Image Synthesis](https://ieeexplore.ieee.org/document/8629301) (IEEE TMI) by Biting Yu, Luping Zhou, Lei Wang, Yinghuan Shi, Jurgen Fripp, and Pierrick Bourgeat.
Their code is based on [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Here are the commands that you could find useful to run if you find a dataset to use (preprocessing steps are detailed in my report):  

./pipeline_mask_FLAIR_patients_v2.sh <data_directory>/MscProject/brain_data/t2_masked_data <data_directory>/MscProject/brain_data/t2_masked_data/BATCH3_20.02.19_nifti/DIGr_P044/04.10.17 20171004143249_t2_tirm_tra_dark-fluid_fs_3 nii

applying a mask:
fslmaths 'image_1.nii.gz' -mas 'mask.nii.gz' 'saving_location.nii.gz'


TRAIN and TEST

python train.py --dataroot <data_directory>/MscProject/brain_data/standard_preprocessed/pairing  --name example_gEaGAN --model gea_gan --which_model_netG unet_128 --which_direction AtoB --lambda_A 1.0 --dataset_mode aligned --use_dropout --batchSize 4 --niter 100 --niter_decay 100 --lambda_sobel 1.0  --fineSize 128 --lr 0.00002 --beta1 0.65
python test.py --dataroot <data_directory>/MscProject/brain_data/standard_preprocessed/pairing --name example_gEaGAN --model gea_gan  --which_direction AtoB --dataset_mode aligned --use_dropout

TIME predictor
python train_time.py --dataroot <data_directory>/MscProject/brain_data/standard_preprocessed/pairing  --name time_pred --model time_predictor  --which_direction AtoB --lambda_A 1.0 --dataset_mode aligned_time --use_dropout --batchSize 1 --niter 100 --niter_decay 100 --lambda_sobel 1.0  --fineSize 128 --lr 0.00002 --beta1 0.65 (--gpu_ids -1)

GEA TPN
python train.py --dataroot <data_directory>/MscProject/brain_data/standard_preprocessed/pairing  --name gea_TPN_test --TPN time_pred --model gea_TPN --which_model_netG unet_128 --which_direction AtoB --lambda_A 1.0 --dataset_mode aligned_TPN --use_dropout --batchSize 1  --lambda_sobel 1.0  --fineSize 128 --lr 0.00002 --beta1 0.65 --save_epoch_freq 10 --gamma 0.1 --niter 600 --niter_decay 200
python test.py --dataroot <data_directory>/MscProject/brain_data/standard_preprocessed/pairing  --name gea_TPN --model gea_TPN  --which_direction AtoB  --dataset_mode aligned_TPN  --fineSize 128 --which_model_netG unet_128 --TPN time_pred
EVAL
python mc_eval.py '<data_directory>/MscProject/Ea-GANs-time/results/gea_TPN2/test_latest/images'

DM GEA
python train.py --dataroot <data_directory>/MscProject/brain_data/DM_preprocessed/pairing  --name gea_DM_example --model gea_DM --which_direction AtoB --lambda_A 1.0 --dataset_mode aligned_DM --use_dropout --batchSize 1  --lambda_sobel 1.0  --fineSize 128 --lr 0.00002 --beta1 0.65 --save_epoch_freq 10 --gamma 0.1 --niter 100 --niter_decay 100
