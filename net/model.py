import torch
import torch.nn as nn
from einops import rearrange, repeat
from net.decoder import Decoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel



class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) 
        embed = self.project_head(embed)
        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)
        return {"feature":output['hidden_states'], "project":project}

class FFBI(nn.Module):
    def __init__(self, dim, num,batchf):
        super(FFBI, self).__init__()
        self.cross_attnh = nn.MultiheadAttention(embed_dim=dim,num_heads=num,batch_first=batchf)
        self.cross_attnl = nn.MultiheadAttention(embed_dim=dim,num_heads=num,batch_first=batchf)

    def forward(self, x,y):
        x1, _=self.cross_attnl(query=x,key=y,value=y)
        x2 = x1 + x
        y1, _ = self.cross_attnh(query=y,key=x,value=x)
        y2 = y1+ y
        return x2,y2


class SegModel(nn.Module):
    def __init__(self, bert_type, vision_type, project_dim=512):
        super(SegModel, self).__init__()
        self.encoder = VisionModel(vision_type, project_dim)
        self.encoder2 = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        self.spatial_dim = [7,14,28,56]   
        feature_dim = [768,384,192,96]

        self.decoder16 = Decoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = Decoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = Decoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

        self.decoder16_2 = Decoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8_2 = Decoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4_2 = Decoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1_2 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out_2 = UnetOutBlock(2, in_channels=24, out_channels=1)
        self.ffbi = FFBI(feature_dim[0],4,True)
    def forward(self, data):

        image2,image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)
            image2 = repeat(image2,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_output2 = self.encoder2(image2)
        image_features, _ = image_output['feature'], image_output['project']
        image_features2, _ = image_output2['feature'], image_output2['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, _ = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 
            image_features2 = image_features2[1:]  
            image_features2 = [rearrange(item,'b c h w -> b (h w) c') for item in image_features2]
        os32 = image_features[3]
        os32_2 = image_features2[3]

        fu32,fu32_2=self.ffbi(os32,os32_2)

        os16 = self.decoder16(fu32,image_features[2], text_embeds[-1])
        os16_2 = self.decoder16_2(fu32_2,image_features2[2], text_embeds[-1])
        
        os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
        os8_2 = self.decoder8_2(os16_2,image_features2[1], text_embeds[-1])

        os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
        os4_2 = self.decoder4_2(os8_2,image_features2[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os4_2 = rearrange(os4_2, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        os1_2 = self.decoder1_2(os4_2)

        out = self.out(os1).sigmoid()
        out_2 = self.out_2(os1_2).sigmoid()
        return out,out_2
    
