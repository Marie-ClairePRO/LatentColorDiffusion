# Latent Color Diffusion 

This repository contains [Latent Color Diffusion](https://github.com/Marie-ClairePRO/LatentColorDiffusion) inspired by the paper Diffusing Colors: Image Colorization with Text Guided Diffusion by Zabari et al, training the model of [stablediffusion2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) at 512x512 resolution, initialized on [Stable Diffusion weights](). 

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


________________________________
  
## Requirements

You can create a new python environment on conda, and install requirements by running

```
conda env create -f environment.yml
pip install -r requirements.txt
```

We suggest to refer to the Stable Diffusion GitHub page for installation issues because we started from here.

## The weights

The weights are available via [the StabilityAI organization at Hugging Face](https://huggingface.co/StabilityAI) under the [CreativeML Open RAIL++-M License](LICENSE-MODEL). 


### Inference

By default, this uses the [DDIM sampler](https://arxiv.org/abs/2010.02502), and renders images of size 512x512 (which it was trained on) in 50 steps. 

#### Training

Download the weights for [_SD2.1-v_](https://huggingface.co/stabilityai/stable-diffusion-2-1) or you can intialize them to random. We trained on [COCO](#) dataset, removing all black and white and very desaturated images, resulting in around 114,000 images of resolution 512x512. We trained for 6 epochs with one RTX4090 GPU. Training took 3 days.


## Shout-Outs



