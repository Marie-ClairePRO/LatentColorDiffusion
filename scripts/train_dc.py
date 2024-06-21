from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from scripts.dataset import TrainingDataset
from logger import ImageLogger, Logger
from scripts.model_load import create_model, load_state_dict


# Configs
#resume_path = 'lightning_logs/version_13/checkpoints/epoch=7-step=3535.ckpt'
resume_path = "/home/mclsaintdizier/Documents/ColorizeNet-main/weights/base/v2-1_512-ema-pruned.ckpt"
prompts_train_coco_path = 'promptsTrain.json'
prompts_test_path = 'promptsTest.json'
prompts_val_path = 'promptsVal.json'

batch_size = 3
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

def dataloader(prompts, p=0.1, b=batch_size, shuffle=False, data_root = "data"):
    dataset = TrainingDataset(prompts, data_root= data_root, p=p)
    return DataLoader(dataset, num_workers=b, batch_size=b, shuffle=shuffle)

model = create_model('./configs/ldm_2.yaml').cpu()
def print_parameters():
    for name,param in model.named_parameters():
        if param.requires_grad == False:
            print("not trainable :",name)
missing, unexpected = model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict = False)
if len(missing) > 0 :
    print("missing keys:")
    print(missing)
if len(unexpected) > 0:
    print("unexpected keys:")
    print(unexpected)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

train_coco_dataloader = dataloader(prompts_train_coco_path, shuffle=True)
val_dataloader = dataloader(prompts_val_path)
test_dataloader = dataloader(prompts_test_path, p=0)
old_photos_dataloader = dataloader(prompts_train_coco_path, data_root="data/old_photos_dataset", p=1)

checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs/version_5/',
    filename='{epoch:02d}-{loss_epoch:.2f}',
    save_top_k=-1,
    every_n_epochs=1
)

#loggers
callbacks = ImageLogger(batch_frequency=logger_freq)
logger = Logger(batch_frequency=30)
trainer = pl.Trainer(max_epochs = 7,
                     gpus = 1,
                     precision=32,                          #32=default=full precision
                     callbacks=[checkpoint_callback, 
                                callbacks, logger],         #image logger + loss and other measures
                     #logger= logger,                       #when no logger is given, default Tensorboard logger is used
                     limit_val_batches = 1.0)               #only takes 20% of validation set, changes each period -> not constant

# Train
trainer.fit(model, train_coco_dataloader, val_dataloader)
