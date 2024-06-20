import cv2
import numpy as np
import os
import torch
import torchvision
from PIL import Image

from torch.utils.data import DataLoader
from train.dataset import InferenceDataset
from data import apply_color
from ldm.models.diffusion.ddim import DDIMSampler
from train.model_load import create_model, load_state_dict
from train.dataset_create_source import create_source

model = create_model('./configs/v2-inference.yaml').cpu()

#weights = "lightning_logs/version_1_trainingCoco/checkpoints/epoch=6-step=266685.ckpt"
#weights = "lightning_logs/version_13_v1_fintuned/checkpoints/epoch=7-step=3535.ckpt"
weights = "lightning_logs/version_20_aug_controlled/epoch=06-loss_epoch=0.00.ckpt"
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
#show desaturation from z (if target input is colorful)
progressive_desaturation = False        
#reshape 512x512 image to original size
reshape_to_initial = True               
#apply original input luminance on colorized result
apply_color_to_sample = True            
#write input image + decoded encoded input image + grayscale input image + conditionning
write_inputs = True    
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
path_to_im = "sample_data/woman_color_3.png"
outdir = "outputs/version_20"

#create mask by hand
'''h, w = 64, 64
mask = torch.ones(1, h, w).to('cuda')
mask[:, h // 12:3* h // 5, 5*w // 8:3*w//4] = 0.
mask = mask[:, None, ...]'''

#path and names of outputs
if not os.path.isdir(path_to_im):
    H0, W0, _ = cv2.imread(path_to_im).shape
else:
    assert len(os.listdir(path_to_im))>0
    H0, W0, _ = cv2.imread(os.path.join(path_to_im,os.listdir(path_to_im)[0])).shape
name_out = path_to_im.split('/')[-1].split('.')[0]
os.makedirs(os.path.join(outdir,name_out), exist_ok=True)


#dataloader
input_dataset = InferenceDataset(data_root=path_to_im, 
                                prompt_path=prompt, 
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
        images = model.log_images(batch, unconditional_guidance_scale = cfg_scale,
                                  ddim_steps=ddim_steps,
                                  unconditional_guidance_label = [n_prompt],
                                  sample = sample,
                                  inpaint = inpaint,
                                  outpaint = outpaint,
                                  mask = mask,
                                  plot_desat_rows = progressive_sampling,
                                  plot_progressive_rows = use_ddpm,
                                  plot_diffusion_rows = progressive_desaturation,
                                  originals = write_inputs,
                                  starting_desat = starting_desat
                                  )
        
    #write results of 1 inference
    for k in images:
        if isinstance(images[k], torch.Tensor):
            grid = init_grid(images[k])
            filename = f"{name_out}_{k}_{count}.png"
            save_path = os.path.join(outdir,name_out, filename)

            if reshape_to_initial and resize:
                H, W, _ = grid.shape
                ratio = round(float(W) / float(H))
                grid = cv2.resize(grid, (ratio*W0,H0),interpolation=cv2.INTER_AREA)
                input_image = cv2.cvtColor(cv2.imread(batch["filename"][0]),cv2.COLOR_BGR2RGB)
            else:
                input_image = batch["target"]

            #apply color only for samples (no grid or input)
            if ratio==1 and apply_color_to_sample and k not in ["mask","conditioning","reconstruction", "inputs", "gray_input"]:
                #model output in RGB, apply returns in RGB order
                grid = apply_color(input_image, grid)
                gray = create_source(cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
                grid = color_scale * grid + (1.- color_scale) * gray

            Image.fromarray(grid.astype(np.uint8)).save(save_path)
            print("saved : ", save_path)
    count += 1



'''

resize_square = False
if resize_square:
    k = float(512) / min(float(H0), float(W0))
    gray = cv2.resize(input_image, (512, 512), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
else:
    gray = resize_image(input_image, resolution=512)


num_samples = 3


gray = (torch.from_numpy(gray.copy()).float().cuda() / 127.5) - 1.0
gray = torch.stack([gray for _ in range(num_samples)], dim=0)
gray = einops.rearrange(gray, 'b h w c -> b c h w').clone()
encoder_posterior_gray = model.encode_first_stage(gray)
gray = model.get_first_stage_encoding(encoder_posterior_gray).detach()


cond = {"c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

shape = (4, H // 8, W // 8)

with torch.cuda.amp.autocast():
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, gray, cond, verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)

pred_x0_samples = intermediates['pred_x0']

def write_intermediate(intermediate_samples, name):
    grid = model._get_desat_row_from_list(intermediate_samples).detach().cpu()
    grid = torchvision.utils.make_grid(grid)
    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).clip(0, 255).astype(np.uint8)    
    print("intermediates saved in",f"{outdir}/{name_out}/inter_{name}_{prompt}_samples.jpg")
    cv2.imwrite(f"{outdir}/{name_out}/inter_{name}_{prompt}_samples.jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

write_intermediate(intermediate_samples=pred_x0_samples, name = "pred_x0")

decode_gray = model.decode_first_stage(gray)
decode_gray = (einops.rearrange(decode_gray, 'b c h w -> b h w c')
         * 127.5 + 127.5)[0].cpu().numpy().clip(0, 255).astype(np.uint8)   
cv2.imwrite(f"{outdir}/{name_out}/gray.jpg", cv2.cvtColor(decode_gray, cv2.COLOR_RGB2BGR))

x_samples = model.decode_first_stage(samples)       #decoding stage
x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
         * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
resized = [cv2.resize(result, (W0,H0),interpolation=cv2.INTER_AREA) for result in x_samples]
colored_results = [apply_color(input_image, result) for result in resized]
print(f"{outdir}/{name_out}/{prompt}.jpg saved")
[cv2.imwrite(f"{outdir}/{name_out}/{prompt}_{i}.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR)) for i, result in enumerate(colored_results)]'''