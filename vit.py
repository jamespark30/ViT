import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
import wandb

import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torchsummary import summary

from tqdm import tqdm


# class PositionalEncoding(nn.Module):
#     def __init__(self):
#         super(PositionalEncoding, self).__init__()

#     def forward(self, x):
#         ##### 덧셈이 addition인지 뒤에 concat되는건지 #####
#         return x + self.encoding

class PatchEmbedding(nn.Module):
    def __init__(self, batch_size: int=100, in_channels: int=3, img_size: int=32, patch_size: int=8, embedding_dim: int=192, pre_train: bool=False):
        super(PatchEmbedding, self).__init__()
        self.batch_size = batch_size          # =B
        self.in_channels = in_channels        # =C
        self.image_size = img_size            # =H =W
        self.patch_size = patch_size          # =P
        self.embedding_dim = embedding_dim    # =D
        self.pre_train = pre_train

        self.linear = nn.Linear(patch_size * patch_size * in_channels, embedding_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, patch_size * patch_size * in_channels))
        # self.positional_encoding = PositionalEncoding()
        self.positional_encoding = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, patch_size * patch_size * in_channels))
        
    def forward(self, x):
        # print(x.shape, self.batch_size)
        assert x.shape[0] == self.batch_size

        b, c, h, w = x.shape

        # 1. slice into patches
        # input : [b, c, h, w]
        # patches : [b, N=HW/p^2, P^2C]
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> b (h2 w2) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)

        if self.pre_train == True:
            pass
        else:   
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
        pre_train: bool = False,
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
            embedding_dim = embedding_dim,
            pre_train = pre_train,
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
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x)

        return x


def test():
    model.eval()
    num_correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader):
            model.eval()
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            loss = criterion(output, target)
            
            output = torch.softmax(output, dim=1)
            pred, idx_ = output.max(-1)
            num_correct += torch.eq(target, idx_).sum().item()
            total += target.size(0)
            avg_loss += loss.item()

    accuracy = num_correct / total
    avg_loss = avg_loss / len(test_loader)            
    print('accuracy : {:.4f}%, avg_loss : {:.4f}'.format(accuracy * 100., avg_loss))


def main(args):
    # download dataset
    os.makedirs('/home/elicer/MM/CIFAR100', exist_ok=True)

    train_set = CIFAR100(root='/home/elicer/MM/CIFAR100',
                        train=True,
                        download=True,
                        transform=T.Compose([
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                        ])
    )
    test_set = CIFAR100(root='/home/elicer/MM/CIFAR100',
                        train=False,
                        download=True,
                        transform=T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                        ])
    )

    # dataloader
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=5)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionTransformer(
        batch_size = 100,
        in_channels = 3,
        img_size = 32,
        patch_size = 8,
        embedding_dim = 192,
        encoder_depth = 12,
        num_heads = 8,
        mlp_dim = 768,
        drop_rate = 0.1,
        num_classes = 100,
        pre_train = True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    step_size=100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=1e-5)

    # Pre-train


    # Train
    print("Start Training")
    for epoch in range(epochs):
        model.train()
        for idx, (img, target) in enumerate(train_loader):
            model.train()
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        num_correct = 0
        val_avg_loss = 0
        total = 0
        with torch.no_grad():
            for idx, (img, target) in enumerate(test_loader):
                model.eval()
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss = criterion(output, target)
                
                output = torch.softmax(output, dim=1)
                pred, idx_ = output.max(-1)
                num_correct += torch.eq(target, idx_).sum().item()
                total += target.size(0)
                val_avg_loss += loss.item()

        accuracy = num_correct / total
        val_avg_loss = val_avg_loss / len(test_loader)            
        print('Epoch {} test : accuracy : {:.4f}%, avg_loss : {:.4f}'.format(epoch, accuracy * 100., val_avg_loss))

        
        # model.eval()
        # num_correct = 0
    
        # with torch.no_grad():
        #     for idx, (img, target) in enumerate(test_loader):
        #         img = img.to(device)
        #         target = target.to(device)

        #         output = model(img)

        #         loss = criterion(output, target)

        #         pred = torch.max(output)

        
        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ViT")

    parser.add_argument('--epoch', default=32, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--embedding_dim', default=192, type=int)
    parser.add_argument('--encoder_depth', default=12, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--mlp_dim', default=768, type=int)
    parser.add_argument('--drop_rate', default=0.1, type=float)
    parser.add_argument('--num_classes', default=100, type=int)

    args = parser.parse_args()

    main(args)