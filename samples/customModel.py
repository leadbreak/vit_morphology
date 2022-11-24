import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
import cv2
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image


# Dataset Class
class customDataset(torch.utils.data.Dataset) :
    def __init__(self, X, Y) :
        self.X = X
        self.Y = Y

        
    def __len__(self) :
        return len(self.X)
    
    def __getitem__(self, index) :                  
          
        return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])


class VitInputLayer(nn.Module) :
    def __init__(self,
                 in_channels:int=3,    # 입력 채널 수
                 emb_dim:int=512,      # embedding vector의 길이
                 num_patch_row:int=64,  # 분할할 Patch 단위(Height를 기준으로 - 보통 정사각형으로 처리)
                 image_size:int=512     # 입력 이미지 한 변의 길이
                ) :
        
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size
        
        # STEP 01. patch 수대로 입력 이미지 분할
        ## 기본 입력값(num_patch_row)대로 분할한다면, 2x2=4개의 이미지로 분할됨
        self.num_patch = self.num_patch_row**2
        
        ## patch에 따른 size를 계산하고, 만약 사이즈가 떨어지지 않으면 error
        self.patch_size = int(self.image_size / self.num_patch_row)
        assert self.image_size % self.num_patch_row == 0, "patch size doesn't match with image size"
        
        ## 입력 이미지를 Patch로 분할하고, Patch 단위로 Embedding하기 위한 레이어 구축
        self.patch_emb_layer = nn.Conv2d(
                                        in_channels=self.in_channels,
                                        out_channels=self.emb_dim,
                                        kernel_size=self.patch_size,
                                        stride=self.patch_size
                                        )
        
        # STEP 02. cls token & position embedding
        ## class token(cls token)
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_dim)) # (1, 1, emb_dim) 차원의 Parameter(변경가능한 값)을 정의
        
        ## pos embedding for sequential info(This is general function in NLP, but optional in CV)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patch+1, emb_dim))
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor : 
        """
        x : (B:batch_size, C:channel_nums, H:height, W:width) 차원의 input image
        
        z_0 : (B:batch_size, N:token_nums, D:dim_of_embedding_vector) 차원의 ViT 입력
        
        """
        
        # STEP 03. Patch Embedding & Flatten
        ## Patch Embedding : (B, C, H, W) -> (B, D, H/P, W/P)
        z_0 = self.patch_emb_layer(x)
        
        ## Flatten : (B, D, H/P, W/P) -> (B, D, Np)
        ### Np : patch_nums = ((H*W)/(P^2))
        z_0 = z_0.flatten(2)
        
        ## Transpose : (B, D, Np) -> (B, Np, D)
        z_0 = z_0.transpose(1,2)
        
        # STEP 04. Patch Embedding + cls token
        ## (B, Np, D) -> (B, N, D)
        ### N = (Np+1)
        z_0 = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)
        
        ## + pos embedding
        try :
            z_0 = z_0 + self.pos_emb
        except :
            raise (z_0.shape, self.pos_emb.shape)
        
        return z_0 # (B, N, D) - B:batch_size, N:token_nums, D:dim_of_embedding_vector
    
class MultiHeadSelfAttention(nn.Module) :
    def __init__(self,
                 emb_dim:int=512,  # embedding vector 길이
                 head:int=4,       # head 개수
                 dropout:float=0.3  # dropout rate
                ) :
        
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head = head
        assert emb_dim % head == 0, f"emb_dim can't be divided by head. emb_dim is {emb_dim} and head is {head}"
        self.head_dim = emb_dim // head        
        self.sqrt_dh = self.head_dim**0.5 # scaling factor로 나눔으로써 feature dimension 구현
        
        # STEP 01. Define Layers
        ## Linear Layer for Query, Key, Value weights
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)
        
        ## Dropout Layer
        self.attn_drop = nn.Dropout(dropout)
        
        # MSHA's output layer
        self.w_o = nn.Sequential(
                                nn.Linear(emb_dim, emb_dim),
                                nn.Dropout(dropout)
        )
        
    def forward(self, z:torch.Tensor) -> torch.tensor :
        """
        z : (B:batch_size, N:token_nums, D:vector_dims) 차원 MHSA 입력
        
        out : (B:batch_size, N:token_nums, D:embedding_vector_dims) 차원 MHSA 출력
        """
        
        batch_size, num_patch, _ = z.size()
        
        # STEP 02. calculate self attention score 
        ## q, k, v embedding
        ## (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)
        
        ## Attention Score 계산을 위한 사전작업
        ## (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)
        
        ## (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        ## 내적 : matmul
        ## k_T : (B, h, N, D//h) -> (B, h, D/h, N)
        k_T = k.transpose(2,3)
        
        ## QKt : (B,h,N,D//h)@(B,h,D//h,N)=(B,h,N,N)
        dots = (q@k_T) / self.sqrt_dh
        
        ## 열방향 softmax 값 & dropout
        attn = F.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)
        
        # 가중화
        ## (B,h,N,N)@(B,h,N,D//h)=(B,h,N,D//h)
        out = attn @ v
        
        ## (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1,2)
        
        ## (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)
        
        ## 출력층 : (B, N, D) -> (B, N, D)
        out = self.w_o(out)
        
        return out

class VitEncoderBlock(nn.Module) :
    def __init__(
                self,
                emb_dim:int=512,
                head:int=8,
                hidden_dim:int=512*4,
                dropout:float=0.3
                ) :
        
        super(VitEncoderBlock, self).__init__()
        
        # STEP 01. Define Encoder Block Layers
        ## 1st Layer Norm
        self.ln1 = nn.LayerNorm(emb_dim)
        self.emb_dim1 = emb_dim // 2
        self.emb_dim2 = emb_dim // 4
        self.emb_dim3 = emb_dim // 8
        self.emb_dim4 = emb_dim // 16
        
        ## MHSA
        self.mhsa = MultiHeadSelfAttention(
                                           emb_dim = emb_dim,
                                           head=head,
                                           dropout=dropout,
                                           )
        
        ## 2nd Layer Norm
        self.ln2 = nn.LayerNorm(emb_dim)
        
        ## MLP2
        self.mlp2 = nn.Sequential(*[nn.Sequential(
                                nn.Linear(emb_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, self.emb_dim1),
                                nn.Dropout(dropout),
                                nn.Linear(self.emb_dim1, self.emb_dim1),
                                nn.Upsample(scale_factor=2),
                                nn.LayerNorm(emb_dim))
                                for _ in range(2)]
                                )
 
        ## MLP3
        self.mlp3 = nn.Sequential(*[nn.Sequential(
                                nn.Linear(emb_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, self.emb_dim2),
                                nn.Dropout(dropout),
                                nn.Linear(self.emb_dim2, self.emb_dim2),
                                nn.Upsample(scale_factor=4),
                                nn.LayerNorm(emb_dim))
                                for _ in range(2)]
                                )        
        
        ## MLP4
        self.mlp4 = nn.Sequential(*[nn.Sequential(
                                nn.Linear(emb_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, self.emb_dim3),
                                nn.Dropout(dropout),
                                nn.Linear(self.emb_dim3, self.emb_dim3),
                                nn.Upsample(scale_factor=8),
                                nn.LayerNorm(emb_dim))
                                for _ in range(4)]
                                )

        
        ## MLP5
        self.mlp5 = nn.Sequential(*[nn.Sequential(
                                nn.Linear(emb_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, self.emb_dim4),
                                nn.Dropout(dropout),
                                nn.Linear(self.emb_dim4, self.emb_dim4),
                                nn.Upsample(scale_factor=16),
                                nn.LayerNorm(emb_dim))       
                                for _ in range(2)]
                                )     
        
        
        
    def forward(self, z:torch.Tensor) -> torch.Tensor :
        """
        z : (B:batch_size, N:token_nums, D:vector_dims) 차원 Encoder Block 입력
        
        out : (B:batch_size, N:token_nums, D:embedding_vector_dims) 차원 Encoder Block 출력
        """
        
        # STEP 02. Construct Encoder Block
        ## 하나의 Encoder Block은 Layer Norm을 기준으로 크게 둘로 나뉘어져 있으며,
        ## 이 과정에서 Residual connection이 고려된다
        
        ### part 1 : MHSA(layerNorm)+ResidualConnection1
        out = self.mhsa(self.ln1(z)) + z
        
        ### part 2 : MLP(layerNorm)+ResidualConnection2
        out1 = self.mlp2(out)+out
        out2 = self.mlp3(out)+out 
        out3 = self.mlp4(out)+out
        out3 = self.mlp4(out3)+out
        out4 = self.mlp5(out)+out
        
        out = out1+out2+out3+out4+out
        
        return out
    
      
class VitDecoderBlock(nn.Module) :
    def __init__(
                 self,
                 emb_dim:int=512,
                 head:int=8,
                 hidden_dim:int=512*4,
                 dropout:float=0.3
                 ) :
        
        super(VitDecoderBlock, self).__init__()
        
        ## 1st Layer Norm
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
                                 nn.Linear(emb_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, emb_dim)
        )
        
        ## 2nd Layer Norm
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mhsa = MultiHeadSelfAttention(
                                           emb_dim = emb_dim,
                                           head=head,
                                           dropout=dropout
        )
        
    def forward(self, z:torch.Tensor) -> torch.Tensor :
        
        
        out = self.mlp(self.ln2(z)) + z
        
        out = self.mlp(self.ln2(z)) + out
        
        return out

class VitOutputLayer(nn.Module) :
    def __init__(self, num_classes:int) :
        
        super(VitOutputLayer, self).__init__()
        
        self.dePatch_emb_layer1 = nn.Conv2d(
                                in_channels=1,
                                out_channels=4,
                                kernel_size=(1,3),
                                stride=(1, 4),
                                padding=(0,0),
                                dilation=(1, 1)
                                )
        self.ln1 = nn.LayerNorm((4, 512, 1024))
        
        self.dePatch_emb_layer2 = nn.Conv2d(
                                in_channels=4,
                                out_channels=8,
                                kernel_size=(1,2),
                                stride=(1, 2),
                                padding=(0,0),
                                dilation=(1, 1)
                                )
        
        self.ln2 = nn.LayerNorm((8, 512, 512))
        
        self.dePatch_emb_layer3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, padding=1, dilation=1),
                                   nn.GELU(),
                                   nn.Conv2d(in_channels=6, out_channels=num_classes, kernel_size=3, padding=1, dilation=1),
                                   nn.GELU(),
                                   nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, padding=1, dilation=1),
                                   nn.GELU(),
                                   nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, padding=1, dilation=1),
                                )
        
        self.ln3 = nn.LayerNorm((num_classes, 512, 512))
        
        
    def forward(self, z:torch.Tensor) -> torch.Tensor :
        # z : (B, N, D)
        z = torch.unsqueeze(z, dim=-1) # (B, N, D, 1)
        z = z.transpose(1,3) # (B, 1, D, N)
                
        z = self.dePatch_emb_layer1(z)  # (B, 1, D, N) -> (B, 3, H, W)
        z = self.ln1(z)
        z = self.dePatch_emb_layer2(z)
        z = self.ln2(z)
        z = self.dePatch_emb_layer3(z)
        z = self.ln3(z)
        
        return z # (B, C, H, W) - 70, 3, 32, 32
    
    
class ViT(nn.Module) :
    def __init__(self,
                 in_channels:int=3,
                 num_classes:int=2,
                 emb_dim:int=512,
                 num_patch_row:int=64,
                 image_size:int=512,
                 num_blocks:int=4,     # Encoder Block의 수
                 head:int=16,
                 hidden_dim:int=512*8,
                 dropout:float=0.3,
                ) :
        
        super(ViT, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.image_size = image_size
        
        # STEP 01. Input Layer
        self.input_layer = VitInputLayer(in_channels,
                                         emb_dim,
                                         num_patch_row,
                                         image_size)
        
        # STEP 02. Encoder = Encoder Block x num_blocks
        self.encoder = nn.Sequential(*[VitEncoderBlock(emb_dim,
                                                       head,
                                                       hidden_dim,
                                                       dropout
                                                      )
                                       for _ in range(num_blocks)])
        
        
        # STEP 03. Decoder = Decoder Block x num blocks
        self.decoder = nn.Sequential(*[VitDecoderBlock(emb_dim,
                                                       head,
                                                       hidden_dim,
                                                       dropout
                                                      )
                                       for _ in range(num_blocks)])
        
        # STEP 04. Output Layer
        self.output_layer = VitOutputLayer(num_classes)
        
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        """
        x : (B:batch_size, C:channel_nums, H:height, W:width) 차원의 ViT 입력 이미지 
        
        out : (B:batch_size, M:class_nums) 차원의 ViT 출력값
        
        """
        
        # STEP 04. Construct ViT
        
        ## Input Layer : (B,C,H,W)->(B,N,D)
        ## N : num_tokens, D : dim_vector
        out = self.input_layer(x)
        self.batch_size = out.shape[0]
        
        ## Encoder : (B,N,D)->(B,N,D)
        out = self.encoder(out)
        
        ## Decoder : (B,N,D)->(B,N,D)
        out = self.decoder(out)
        
        ## Output Head : (B,N,D)->(B,C,H,W)
        pred = self.output_layer(out)
        
        return pred