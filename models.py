import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torchsummary import summary


# class PositionalEncoding(nn.Module):
#     def __init__(self):
#         super(PositionalEncoding, self).__init__()

#     def forward(self, x):
#         ##### 덧셈이 addition인지 뒤에 concat되는건지 #####
#         return x + self.encoding

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        batch_size: int=100,
        in_channels: int=3,
        img_size: int=32,
        patch_size: int=8,
        embedding_dim: int=192
    ):
        super(PatchEmbedding, self).__init__()
        self.batch_size = batch_size          # =B
        self.in_channels = in_channels        # =C
        self.image_size = img_size            # =H =W
        self.patch_size = patch_size          # =P
        self.embedding_dim = embedding_dim    # =D

        self.linear = nn.Linear(patch_size * patch_size * in_channels, embedding_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, patch_size * patch_size * in_channels))
        # self.positional_encoding = PositionalEncoding()
        self.positional_encoding = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, patch_size * patch_size * in_channels))
        
    def forward(self, x):
        print(x.shape, self.batch_size)
        assert x.shape[0] == self.batch_size

        b, c, h, w = x.shape

        # 1. slice into patches
        # input : [b, c, h, w]
        # patches : [b, N=HW/p^2, P^2C]
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> b (h2 w2) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        

        # 2. prepend class token
        class_tokens = repeat(self.class_token,'() n e -> b n e', b=self.batch_size) 
        ####### broadcasting안되나 굳이 repeat 써야하나 #######
        x = torch.cat([class_tokens, x], dim=1)
        # print(class_tokens.shape)
        
        # 3. add positional encoding
        # x = self.positional_encoding(x)
        x += self.positional_encoding

        # [10, 3, 32, 32] -> [10, 16, 192] -> [10, 17, 192]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int=192, num_heads: int=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embedding_dim, embedding_dim*3, bias=False)  # 편의를 위해 qkv를 한꺼번에 계산하며, multihead처리를 위해 rearrange해줌
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x : [b, N+1, embedding_dim]

        x = self.qkv(x) # [b, N+1, embedding_dim] -> [b, N+1, embedding_dim*3]
        q, k, v = rearrange(x, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads) # [3, b, num_heads, N+1, embedding_dim/num_heads]
        # q, k, v 각각 [b, num_heads, N+1, embedding_dim/num_heads]

        # attention(q,k,v) = sofrmax(QK^T/sqrt{d_k})V
        attention_dist = F.softmax(torch.einsum('bhqd, bhkd -> bhqk', q, k) / (self.embedding_dim**0.5), dim=-1)
        out = torch.einsum('bhal, bhlv -> bhav', attention_dist, v)
        out = rearrange(out, "b h n d -> b n (h d)")  # multi-head를 하나로 합쳐주기

        out = self.projection(out)
        # 마지막에 layer 하나더 왜 통과지?

        # Dropout?

        return out


class MLPLayer(nn.Module):
    def __init__(self, dim, mlp_dim, drop_rate):
        super(MLPLayer, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, function):
        super(ResidualBlock, self).__init__()
        self.function = function

    def forward(self, x):
        residual = x
        return self.function(x) + residual


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int=192, num_heads: int=8, mlp_dim: int=768, drop_rate: float=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.drop_rate = drop_rate
        
        self.attention_block_res = ResidualBlock(
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, **kwargs)
            )
        )
        self.MLP_block_res = ResidualBlock(
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                MLPLayer(dim=embedding_dim, mlp_dim=mlp_dim, drop_rate=drop_rate),
            )
        )

    def forward(self, x):
        x = self.attention_block_res(x)
        x = self.MLP_block_res(x)

        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, encoder_depth: int=12, **kwargs):
        super(TransformerEncoder, self).__init__(
            *[
                TransformerEncoderBlock(**kwargs) for _ in range(encoder_depth)
            ]
        )


class ClassificationHead(nn.Module):
    # pre-training : MLP w/ one hidden layer
    # fine-tuning : single linear layer
    def __init__(self, embedding_dim: int=192, num_classes: int=100):
        super(ClassificationHead, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # x = Reduce('b n e -> b e', reduction='mean')
        x = x[:, 0, :]
        x = self.layer_norm(x)
        x = self.linear(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, 
        batch_size: int = 100,
        in_channels: int = 3,
        img_size: int = 32,
        patch_size: int = 8,
        embedding_dim: int = 192,
        encoder_depth: int = 12,
        num_heads: int = 8,
        mlp_dim: int = 768,
        drop_rate: float = 0.1,
        num_classes: int = 100,
    ):
        super(VisionTransformer, self).__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        
        self.patch_embedding = PatchEmbedding(
            batch_size = batch_size,
            in_channels = in_channels,
            img_size = img_size,
            patch_size = patch_size,
            embedding_dim = embedding_dim
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_depth = encoder_depth,
            embedding_dim = embedding_dim,
            num_heads = num_heads,
            mlp_dim = mlp_dim,
            drop_rate = drop_rate
        )
        self.classification_head = ClassificationHead(
            embedding_dim = embedding_dim,
            num_classes = num_classes,
        )
    
    def forward(self, x):
        print(x.shape)
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x)

        return x


if __name__ == '__main__':
    summary(VisionTransformer(batch_size=2), (3, 32, 32), device='cpu')