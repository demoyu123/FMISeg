import json
import os
import torch
import pandas as pd
from monai.transforms import (Compose, NormalizeIntensityd,RandRotated,RandZoomd,Resized, ToTensord, LoadImaged, EnsureChannelFirstd,RandGaussianNoised)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class SegData(Dataset):

    def __init__(self, dataname,csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(SegData, self).__init__()

        self.dataname=dataname
        self.mode = mode
        self.root_path = root_path
        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.output_path = os.path.join(self.root_path, 'GTs')
        self.image_list = os.listdir(self.output_path)
        if dataname=="cov19":
            self.caption_list = {image: caption for image, caption in zip(self.data['Image'], self.data['Description'])} #qata
        else:
            self.caption_list = {image: caption for image, caption in zip(self.data['Image'], self.data['text'])} #mod
        
        self.image_size = image_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)
        if self.dataname=="cov19": #QaTa-COV19
            image = os.path.join(self.root_path,'Images_H',self.image_list[idx].replace('mask_',''))#H
            image2 = os.path.join(self.root_path,'Images_L',self.image_list[idx].replace('mask_',''))#L
            gt = os.path.join(self.root_path,'GTs', self.image_list[idx])
        else:  #MosMedData
            image = os.path.join(self.root_path,'Images_H',self.image_list[idx])#H
            image2 = os.path.join(self.root_path,'Images_L',self.image_list[idx])#L
            gt = os.path.join(self.root_path,'GTs', self.image_list[idx])
        

        caption = self.caption_list[self.image_list[idx]]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image,'image2':image2, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,image2,gt,token,mask = data['image'],data['image2'],data['gt'],data['token'],data['mask']
        gt = torch.where(gt==255,1,0)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 

        return ([image,image2, text], gt)

    def transform(self,image_size=[224,224]):

        keys = ["image","image2", "gt"]
        if self.mode == 'train':  
            trans = Compose([
                LoadImaged(keys=keys, reader='PILReader'),  
                EnsureChannelFirstd(keys=keys),  
                RandZoomd(keys=keys, min_zoom=0.95, max_zoom=1.15, mode=["bicubic","bicubic", "nearest"], prob=0.3),  
                RandRotated(keys=keys, range_x=[-0.3, 0.3], keep_size=True, mode=['bicubic','bicubic', 'nearest'],  prob=0.3),  
                RandGaussianNoised(keys=["image2"], prob=0.3, mean=0.0, std=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["image2"],spatial_size=image_size,mode='bicubic'),
                Resized(keys=["gt"], spatial_size=image_size, mode='nearest'), 
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                NormalizeIntensityd(['image2'], channel_wise=True),
                ToTensord(keys=["image","image2", "gt", "token", "mask"])  
            ])            
        else:  
            trans = Compose([
                LoadImaged(["image","image2","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","image2","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["image2"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                NormalizeIntensityd(['image2'], channel_wise=True),
                ToTensord(["image","image2","gt","token","mask"]),
            ])

        return trans