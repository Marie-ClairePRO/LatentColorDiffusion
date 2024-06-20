import json
from pycocotools.coco import COCO
from pprint import pprint
from random import randint
import os
import tqdm

def read_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return data

def init_coco():
	dirType='train'
	#dirType = "val"
	dir = f"/home/mclsaintdizier/Documents/coco/annotations/captions_{dirType}2017.json"
	images_dir = f"/home/mclsaintdizier/Documents/coco/{dirType}2017"

	promptsDir = "/home/mclsaintdizier/Documents/ColorizeNet-main/data/colorization/training/"
	newPrompts = promptsDir + "promptsTrain.json"
	valid_imgs = os.listdir(images_dir)
	
	coco = COCO(dir)
	imgs_keys = coco.imgs.keys()
	ann = coco.anns
	
	return newPrompts, valid_imgs, imgs_keys, ann

def init_finetune():
    dir = "/home/mclsaintdizier/Documents/ColorizeNet-main/data/colorization/finetune_dataset/"
    images_dir = os.path.join(dir,"train")
    newPrompts = os.path.join(dir,"promptsTrain.json")
    valid_imgs = os.listdir(images_dir)

    imgs_keys = range(len(valid_imgs))
    return newPrompts, valid_imgs, imgs_keys

def get_captions(capdir):
	captions = []
	old_prompts = read_json_lines(capdir)
	for prompt in old_prompts:
		if prompt['prompt'] not in captions:
			captions.append(prompt['prompt'])
	return captions

#random_captions = get_captions(getPrompts)

def find_color(caption):
    colors = ["white ", "black ", " red ", " blue ", " green", "cyan", 
              "pink", "orange ", "purple", "grey ", "yellow","magenta", 
              "turquoise", "maroon", "olive", "beige"]
    for color in colors:
        if color in caption:
            return True
    return False

def create_json(imgs_keys, valid_imgs=None, ann=None, zfill=12, type='jpg'):
    prompts = []
    for img_key in tqdm.tqdm(imgs_keys):   
        if ann is not None:    
            all_captions = [ann[key]["caption"] for key in ann.keys() if ann[key]["image_id"] == img_key]
        else:   
            all_captions = []
        image_name = str(img_key).zfill(zfill)+f'.{type}'
        if image_name in valid_imgs:
            source = 'train/'+ image_name
            if not all_captions:        #check if empty
                prompt = ""
            else:
                prompt = all_captions[0]
                for caption in all_captions:
                    if find_color(caption):
                        prompt = caption
                        break
            prompts.append({ 'source' : source, 'target' : source, 'prompt' : prompt})
    return prompts

def create_coco_json_file():
    newPrompts, valid_imgs, imgs_keys, ann = init_coco()
    json_lines = [json.dumps(prompt) for prompt in create_json(imgs_keys,valid_imgs,ann,zfill=12, type='jpg')]

    with open(newPrompts, "w") as json_file:
        json_file.write('\n'.join(json_lines))

def create_finetune_json_file():
    newPrompts, valid_imgs, imgs_keys = init_finetune()
    json_lines = [json.dumps(prompt) for prompt in create_json(imgs_keys,valid_imgs,zfill=0,type='png')]

    with open(newPrompts, "w") as json_file:
        json_file.write('\n'.join(json_lines))

create_finetune_json_file()


