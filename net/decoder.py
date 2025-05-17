import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock

class SelfAugment(nn.Module):
    def __init__(self, in_channels):
        super(SelfAugment, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.vis_pos = PositionalEncoding(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.self_attn_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        vis = self.norm(x)
        q = k = self.vis_pos(vis)
        vis = self.self_attn(q, k, value=vis)[0]
        vis = self.self_attn_norm(vis)
        vis = x + vis
        return vis

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]

class FeedLinear(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedLinear, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class LFFI(nn.Module):
    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):
        super(LFFI, self).__init__()
        self.in_channels = in_channels
        self.augment = SelfAugment(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.norm4 = nn.LayerNorm(in_channels)
        self.norm5 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.fl1=FeedLinear(in_channels,in_channels*2)
        self.fl2=FeedLinear(in_channels,in_channels*2)
        self.line=nn.Linear(output_text_len, in_channels)
    def forward(self,x,txt):
        '''
        x:[B N C1]
        tExt:[B,L,C]
        '''
        txt = self.text_project(txt)
        vis = self.augment(x)
        vis2 = self.norm1(vis)
        vis2_v,_ = self.cross_attn1(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt)  
        vis2_l,_ = self.cross_attn2(query=self.txt_pos(txt),
                                   key=self.vis_pos(vis2),
                                   value=vis2)
        vis2_v=self.norm2(vis2_v+vis2)
        vis2_l=self.norm3(vis2_l+txt)
        vis2_v=self.norm4(self.fl1(vis2_v)+vis2_v)
        vis2_l=self.norm5(self.fl2(vis2_l)+vis2_l)
        vis2=vis2_v+self.line(torch.matmul(vis2_v,vis2_l.transpose(1, 2)))
        vis2 = self.cross_attn_norm(vis2)
        vis = vis*self.scale*vis2
        return vis

class Decoder(nn.Module):
    def __init__(self,in_channels, out_channels, spatial_size, text_len) -> None:
        super().__init__()
        self.lffi_layer = LFFI(in_channels,text_len)  
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')

    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis =  self.lffi_layer(vis, txt)
        vis = rearrange(vis,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.spatial_size*2,W=self.spatial_size*2)
        output = self.decoder(vis,skip_vis)
        output = rearrange(output,'B C H W -> B (H W) C')
        return output


