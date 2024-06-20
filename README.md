# Latent Color Diffusion 

This repository contains [Latent Color Diffusion](https://github.com/Marie-ClairePRO/LatentColorDiffusion), a model diffusing the colors in the latent space, inspired by the paper Diffusing Colors: Image Colorization with Text Guided Diffusion by Zabari et al.
It trains the UNet model of [stablediffusion2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) at 512x512 resolution, initialized on [Stable Diffusion weights](https://huggingface.co/stabilityai/stable-diffusion-2-1), and freezes the AutoEncoder + CLIP. 

________________
*The original Stable Diffusion model was created in a collaboration with [CompVis](https://arxiv.org/abs/2202.00512) and [RunwayML](https://runwayml.com/) and builds upon the work:*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
_[CVPR '22 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
[GitHub](https://github.com/CompVis/latent-diffusion) | [arXiv](https://arxiv.org/abs/2112.10752) | [Project page](https://ommer-lab.com/research/latent-diffusion-models/)_

Stable Diffusion is a latent text-to-image diffusion model.

This code was largely inspired by the concept of [Cold-Diffusion](https://arxiv.org/abs/2208.09392) which was implemented here : [Cold Diffusion Models](https://github.com/arpitbansal297/Cold-Diffusion-Models). It was adapted to colorizing images, as described in the paper [Diffusing Colors](https://arxiv.org/pdf/2312.04145), released in December 2023. We added control with color hints, and the objective is to add a temporal version.

The prompts are unfortunately of no concrete use.


________________________________
  
## Requirements

You can create a new python environment on conda, and install requirements by running

```
conda env create -f environment.yml
pip install -r requirements.txt
```

We suggest to refer to the [Stable Diffusion GitHub page](https://github.com/Stability-AI/stablediffusion) for installation issues because we started from here.

## The weights

The weights are not available yet.


### Inference

By default, this uses the [DDIM sampler](https://arxiv.org/abs/2010.02502), and renders images of size 512x512 (which it was trained on) in 50 steps. The images can be resized to original with parameters.
The code for inference is in *colorize/colorize.py*, and you can change parameters in the code.
Put your images in a sample_data folder. Prompts can be given for inference but have shown unuseful, even deteriorating the results. We chose the cfg_scale to be 1. and give no prompt.<br/>
Inference parameters :<br/>
*ddim_steps : set to 50, 30 can sometimes show good results. <br/>
sample : True by default, start the basic inference.<br/>
inpaint / outpaint : start inference with mask to inpaint or outpaint.<br/>
mask : mask is in format 64x64 (latent space), 1 for unchanged zone and 0 for in/outpaint.<br/>
progressive_sampling : save intermediate steps of prediction.<br/>
reshape_to_initial : inference is on 512x512 square images, you want to reshape them afterwards back to original.<br/>
apply_color_to_sample : mix original input luminance to sampled output colors (VAE modify image details).<br/>
write_inputs : to save the input (color and grayscale if given color etc).<br/>
color_scale : only natural desaturation post sampling to give better looking results.<br/>
starting_desat : in ]0,1] for initial level of desaturation (1 for grayscale).*<br/>

Dataset parameters:
*path_to_im : might be image or directory to video frames (png or jpg).<br/>
outdir : by default is outputs.<br/>
isSource : whether image is source or you want to convert it grayscale.*


#### Training

Download the weights for [_SD2.1-v_](https://huggingface.co/stabilityai/stable-diffusion-2-1) or you can intialize them randomly. We trained on [COCO](#) dataset, removing all black and white and very desaturated images, resulting in around 114,000 images of resolution 512x512. We trained for 6 epochs with one RTX4090 GPU. Training took 3 days.
COCO dataset has to be put in data/colorization/training/train.

Training code is in train/train_dc.py. 
As prompts appear to not help the model, we will likely remove them from Dataset models and directly list images from folder without using any json file.

