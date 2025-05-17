import torch
from torch.utils.data import DataLoader
from utils.dataset import SegData
import utils.config as config
from torch.optim import lr_scheduler
from net.creratemodel import CreateModel
import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import os
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/train.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

if __name__ == '__main__':
    args = get_parser()
    print("cuda is used:",torch.cuda.is_available())
    ds_train = SegData(dataname="cov19", #cov19
                    csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='train')

    ds_valid = SegData(dataname="cov19",#cov19
                    csv_path=args.val_csv_path,
                    root_path=args.val_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)

    model = CreateModel(args)
    model_ckpt = ModelCheckpoint(
    dirpath=args.model_save_path,
    filename=args.model_save_filename,
    monitor='val_MIoU',  
    save_top_k=1,  
    mode='max',  
    verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor='val_MIoU',
        patience=args.patience, 
        mode='max',  
    )

    trainer = pl.Trainer(logger=True,
                        min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                        accelerator='gpu', 
                        devices=args.device,
                        callbacks=[model_ckpt,early_stopping],
                        enable_progress_bar=False,
                        ) 
    print('====start====')
    trainer.fit(model,dl_train,dl_valid)
    print('====finish====')

