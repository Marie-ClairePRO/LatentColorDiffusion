import json
import cv2
import numpy as np
import random
import os

from torch.utils.data import Dataset
from scripts.dataset_create_source import *
from scripts.data import apply_color


class TrainingDataset(Dataset):
    def __init__(self, 
                 prompt_path, 
                 data_root = "data/colorization/training/", 
                 resize=True, 
                 p=0.1, 
                 control_points=0.8,
                 augmentation = 0.5):
        
        self.resize = resize
        self.data = []
        self.prompt_path = prompt_path
        self.p = p
        self.data_root = data_root
        self.control_points =control_points
        self.augmentation = augmentation
        assert os.path.isdir(data_root)
        json_path = os.path.join(self.data_root,self.prompt_path)
        assert os.path.isfile(json_path), json_path + " does not exist"
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filename = item['target']

        #no prompt for __% of dataset
        if random.random() > self.p:
            prompt = item['prompt']
        else:
            prompt = ""

        target = cv2.imread(os.path.join(self.data_root,target_filename)) 

        #data augmentation : for better sampling of old images
        if random.random() < self.augmentation:
            blur=random.randint(0,2) #max = 2
            contrast = min(1.,random.random()*0.5 + 0.7) #min = 0.7, proba 2/5 to unchange
            noise=random.randint(0,15) #max = 15
            motion_blur=random.randint(0,7) #max = 7
            compression_factor = random.randint(1,2) #max = 2
            target = deteriorate_image(target, blur, contrast, noise, motion_blur, compression_factor)

        # create_source return RGB image + mask if asked
        # give color hints for __% of dataset
        nb_pts = 50 if random.random() < self.control_points else 0
        source = create_source(target, NB_PTS=nb_pts, desat=0.25)
        
        # convert to RGB -- OpenCV reads images in BGR order
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        if self.resize:
            source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_CUBIC)
            target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            H, W, _ = source.shape
            k = float(512) / min(float(H), float(W))
            H = int(np.round(float(H)*k / 64.0)) * 64            # image resized to H and W being multiple of 64
            W = int(np.round(float(W)*k / 64.0)) * 64
            source = cv2.resize(source, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
            target = cv2.resize(target, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
            assert np.shape(source) == np.shape(target)

        # Normalize images to [-1, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(image=target, txt=prompt, source=source, filename=target_filename)
    


class InferenceDataset(Dataset):
    def __init__(self, prompt, 
                 data, 
                 resize=True, 
                 isSource = False,
                 withControl=False):
        
        self.resize = resize
        self.data_path = data
        self.with_control = withControl 
        self.isSource = isSource
        self.isImgDir = os.path.isdir(self.data_path)
        if self.isImgDir:
            imgs = os.listdir(self.data_path)
            imgs.sort() #in case is video, to keep frames in good order
            imgs = np.flip(imgs)
            self.data = [{'data': os.path.join(self.data_path,img) , 'prompt':prompt} for img in imgs
                          if img.endswith(("png","jpg","jpeg","tif","bmp","webp","dib"))]
        else:
            assert os.path.isfile(self.data_path)
            self.data = [{'data': self.data_path, 'prompt':prompt}]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        data_filename = item['data']
        prompt = item['prompt']
        image_data = cv2.imread(data_filename)

        #if image given is source, only convert to rgb
        if self.isSource:
            source = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        else:
            #for testing on color images
            nb_pts = 50 if self.with_control else 0
            source = create_source(image_data, NB_PTS=nb_pts, desat=0.25)
        
        # convert to RGB -- OpenCV reads images in BGR order
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        #resizing
        if self.resize:
            source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_CUBIC)
            image_data = cv2.resize(image_data, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            H, W, _ = source.shape
            k = float(512) / min(float(H), float(W))
            H = int(np.round(float(H)*k / 64.0)) * 64            # image resized to H and W being multiple of 64
            W = int(np.round(float(W)*k / 64.0)) * 64
            source = cv2.resize(source, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
            image_data = cv2.resize(image_data, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
            assert np.shape(source) == np.shape(image_data)

        #Normalize source images to [-1, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0
        image_data = (image_data.astype(np.float32) / 127.5) - 1.0

        #image data only for inpaint or outpaint
        return dict(image=image_data, txt=prompt, source=source, filename=data_filename)

