# Edge-Aware GANs for Brain Tumour MRI Synthesis

Testing and extending Edge-Aware GANs (Ea-GANs) for cross-modality MR image synthesis on a private brain tumour dataset. This includes variants of gEa-GAN, dEa-GAN, a time prediction network (TPN), and a deformation model (DM) for modelling tumour evolution over time.

Based on [Ea-GANs: Edge-Aware Generative Adversarial Networks for Cross-Modality MR Image Synthesis](https://ieeexplore.ieee.org/document/8629301) (IEEE TMI) by Yu et al., which itself builds on [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

The full experimental setup and results are described in the MSc report (`Msc_project_glioma_recognition__Final_.pdf`).

## Model Variants

| Model | `--model` | Description |
|---|---|---|
| gEa-GAN | `gea_gan` | Generator edge-aware GAN |
| dEa-GAN | `dea_gan` | Discriminator edge-aware GAN (sobel edges fed to D) |
| Time Predictor | `time_predictor` | Predicts time period from difference maps |
| gEa-GAN + TPN | `gea_TPN` | gEa-GAN conditioned on time via a pretrained TPN |
| gEa-GAN + DM | `gea_DM` | gEa-GAN with deformation model (time ratio conditioning) |

## Getting Started

### Install

```bash
uv sync
```

### Preprocessing

Preprocessing steps are detailed in the report. Example commands:

```bash
# Masking pipeline
./pipeline_mask_FLAIR_patients_v2.sh <dataroot>/brain_data/t2_masked_data \
  <dataroot>/brain_data/t2_masked_data/BATCH3_20.02.19_nifti/DIGr_P044/04.10.17 \
  20171004143249_t2_tirm_tra_dark-fluid_fs_3 nii

# Applying a mask
fslmaths image_1.nii.gz -mas mask.nii.gz output.nii.gz
```

## Usage

### gEa-GAN

```bash
# Train
uv run python train.py \
  --dataroot <dataroot>/brain_data/standard_preprocessed/pairing \
  --name example_gEaGAN --model gea_gan \
  --which_model_netG unet_128 --which_direction AtoB \
  --lambda_A 1.0 --dataset_mode aligned --use_dropout \
  --batchSize 4 --niter 100 --niter_decay 100 \
  --lambda_sobel 1.0 --fineSize 128 --lr 0.00002 --beta1 0.65

# Test
uv run python test.py \
  --dataroot <dataroot>/brain_data/standard_preprocessed/pairing \
  --name example_gEaGAN --model gea_gan \
  --which_direction AtoB --dataset_mode aligned --use_dropout
```

### Time Predictor

```bash
uv run python train_time.py \
  --dataroot <dataroot>/brain_data/standard_preprocessed/pairing \
  --name time_pred --model time_predictor \
  --which_direction AtoB --lambda_A 1.0 --dataset_mode aligned_time \
  --use_dropout --batchSize 1 --niter 100 --niter_decay 100 \
  --lambda_sobel 1.0 --fineSize 128 --lr 0.00002 --beta1 0.65
```

### gEa-GAN + TPN

```bash
# Train (requires a pretrained time predictor)
uv run python train.py \
  --dataroot <dataroot>/brain_data/standard_preprocessed/pairing \
  --name gea_TPN_test --TPN time_pred --model gea_TPN \
  --which_model_netG unet_128 --which_direction AtoB \
  --lambda_A 1.0 --dataset_mode aligned_TPN --use_dropout \
  --batchSize 1 --lambda_sobel 1.0 --fineSize 128 \
  --lr 0.00002 --beta1 0.65 --save_epoch_freq 10 \
  --gamma 0.1 --niter 600 --niter_decay 200

# Test
uv run python test.py \
  --dataroot <dataroot>/brain_data/standard_preprocessed/pairing \
  --name gea_TPN --model gea_TPN \
  --which_direction AtoB --dataset_mode aligned_TPN \
  --fineSize 128 --which_model_netG unet_128 --TPN time_pred
```

### gEa-GAN + DM

```bash
uv run python train.py \
  --dataroot <dataroot>/brain_data/DM_preprocessed/pairing \
  --name gea_DM_example --model gea_DM \
  --which_direction AtoB --lambda_A 1.0 --dataset_mode aligned_DM \
  --use_dropout --batchSize 1 --lambda_sobel 1.0 --fineSize 128 \
  --lr 0.00002 --beta1 0.65 --save_epoch_freq 10 \
  --gamma 0.1 --niter 100 --niter_decay 100
```

### Tests

```bash
uv run pytest
```

## Authors

Mathieu Charbonnel

## Acknowledgments

Dr. Matthew Williams (supervisor) and Dr. Elsa Angelini (co-supervisor), Imperial College London. This 3D work builds on Andreas Zinonos' 2D Pix2Pix experiments on brain tumour evolution.

Yu, B., Zhou, L., Wang, L., Shi, Y., Fripp, J., & Bourgeat, P. (2019). Ea-GANs: Edge-aware generative adversarial networks for cross-modality MR image synthesis. IEEE TMI.
