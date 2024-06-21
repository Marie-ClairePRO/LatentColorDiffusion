import cv2
import numpy as np
import os
import torch
import torchvision
from PIL import Image

from torch.utils.data import DataLoader
from scripts.dataset import InferenceDataset
from data import apply_color
from ldm.models.diffusion.ddim import DDIMSampler
from scripts.model_load import create_model, load_state_dict
from scripts.dataset_create_source import create_source

model = create_model('./configs/v2-inference.yaml').cpu()

#path to weights
weights = "/home/mclsaintdizier/Documents/stablediffusion_utils/lightning_logs/version_20_aug_controlled/epoch=06-loss_epoch=0.00.ckpt"
model.load_state_dict(load_state_dict(weights, location='cuda'))
model = model.cuda().eval()
ddim_sampler = DDIMSampler(model)


#set up
prompt = ""
#negative prompt for unconditional guidance
n_prompt = ""                           
#number of steps (<=50)
ddim_steps = 50                        
#unconditional guidance scale. If !=1.0, use unconditional guidance 
#cfg_scale = 1.6 in Diffusing Colors
cfg_scale = 1.                            
#start normal ddim sample without unconditional guidance
sample = True                           
#if inpaint
inpaint = False                         
#if outpaint
outpaint = False                        
#mask for inpaint or outpaint, default : square in the middle
mask = None
#show grid of 4 intermediate steps
progressive_sampling = False            
#ddpm sampling (100 steps)
use_ddpm = False                        
#reshape 512x512 image to original size
reshape_to_initial = True               
#apply original input luminance on colorized result
apply_color_to_sample = True            
#write input image + decoded encoded input image + grayscale input image + conditionning
write_input = False   
#scale : output = s * model_output + (1 - s) * grayscale 
#color_scale = 0.8 in Diffusing Colors
color_scale = 0.8
#desaturation starting point, between 0 and 1 / 1 is grayscale
#number of steps if ddim_steps * starting_desat
starting_desat = 1.     


#### dataset parameters
#resize to 512x512 for inference
resize = True
#keep image as source or change it
isSource = True

#image/video and output
#path_to_im is either image file or folder of image files
path_to_im = "/home/mclsaintdizier/Documents/stablediffusion_utils/sample_data/video_1"
outdir = "/home/mclsaintdizier/Documents/stablediffusion_utils/outputs/version_20"

#image/video and output
#path_to_im is either image file or folder of image files
path_to_im = "sample_data/woman_color_3.png"
outdir = "outputs/version_20"

#create mask by hand
'''h, w = 64, 64
mask = torch.ones(1, h, w).to('cuda')
mask[:, h // 12:3* h // 5, 5*w // 8:3*w//4] = 0.
mask = mask[:, None, ...]'''

#path and names of outputs
name_out = path_to_im.split('/')[-1].split('.')[0]
os.makedirs(os.path.join(outdir,name_out), exist_ok=True)


#dataloader
input_dataset = InferenceDataset(data_root=path_to_im, 
                                prompt=prompt, 
                                resize=resize,
                                isSource=isSource)

input_dataset = DataLoader(input_dataset)


def init_grid(grid):
    grid = grid.detach().cpu()
    grid = torch.clamp(grid, -1., 1.)
    grid = torchvision.utils.make_grid(grid, nrow=4)
    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    return grid

#sampling
ratio = 1
count = 0
for batch in input_dataset:
    with torch.no_grad():
        images = model.infer_images(batch, 
                                    #unconditional_guidance_scale = cfg_scale,
                                    ddim_steps=ddim_steps,
                                    #unconditional_guidance_label = [n_prompt],
                                    inpaint = inpaint,
                                    mask = mask,
                                    plot_desat_rows = progressive_sampling,
                                    plot_progressive_rows = use_ddpm,
                                    original = write_input,
                                    starting_desat = starting_desat
                                    )
        
    #write results of 1 inference
    for k in images:
        if isinstance(images[k], torch.Tensor):
            grid = init_grid(images[k])
            filename = f"{name_out}_{k}_{count}.png"
            save_path = os.path.join(outdir,name_out, filename)

            if reshape_to_initial and resize:
                input_image = cv2.cvtColor(cv2.imread(batch["filename"][0]),cv2.COLOR_BGR2RGB)
                H0, W0, _ = input_image.shape
                H, W, _ = grid.shape
                ratio = round(float(W) / float(H)) #for grids
                grid = cv2.resize(grid, (ratio*W0,H0),interpolation=cv2.INTER_AREA)
            else:
                input_image = batch["target"]

            #apply color only for samples (no grid or input)
            if ratio==1 and apply_color_to_sample and k not in ["mask", "conditioning","reconstruction", "inputs", "gray_input"]:
                #model output in RGB, apply returns in RGB order
                grid = apply_color(input_image, grid)
                gray = create_source(cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
                grid = (color_scale * grid + (1.- color_scale) * gray).astype(np.uint8)

            Image.fromarray(grid).save(save_path)
            print("saved : ", save_path)
    count += 1