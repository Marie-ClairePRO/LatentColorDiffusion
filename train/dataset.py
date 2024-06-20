import json
import cv2
import numpy as np
import random
import os

from torch.utils.data import Dataset
from train.dataset_create_source import *


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
        assert os.path.isfile(os.path.join(self.data_root,self.prompt_path))
        with open(os.path.join(self.data_root,self.prompt_path), 'rt') as f:
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
            desat = min(1.,random.random() + 0.6) #min = 0.6
            noise=random.randint(0,15) #max = 15
            motion_blur=random.randint(0,7) #max = 7
            compression_factor = random.randint(1,2) #max = 2
            target = deteriorate_image(target, blur, desat, noise, motion_blur, compression_factor)

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

        return dict(jpg=target, txt=prompt, source=source, filename=target_filename)
    


class InferenceDataset(Dataset):
    def __init__(self, prompt_path, 
                 data_root = "data/colorization/training/", 
                 resize=True, 
                 augmentation = 0.3,
                 isSource = False,
                 withControl=False):
        
        self.resize = resize
        self.data = []
        self.prompt_path = prompt_path
        self.data_root = data_root
        self.with_control = withControl 
        self.augmentation = augmentation
        self.isSource = isSource
        isVideo = os.path.isdir(data_root)
        if isVideo:
            imgs = os.listdir(data_root)
            imgs.sort()
            self.data = [{'target': os.path.join(data_root,img) , 'prompt':prompt_path} for img in imgs
                          if img.endswith(("png","jpg"))]
        else:
            assert os.path.isfile(data_root)
            self.data.append({'target': data_root, 'prompt':prompt_path})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filename = item['target']
        prompt = item['prompt']

        target = cv2.imread(target_filename)

        #if image given is source, only convert to rgb
        if self.isSource:
            source = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        else:
            nb_pts = 50 if self.with_control else 0
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

        #Normalize source images to [-1, 1].
        #source = source.astype(np.float32) / 255.0
        source = (source.astype(np.float32) / 127.5) - 1.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, source=source, filename=target_filename)

