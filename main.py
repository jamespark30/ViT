import torch
import torch.nn as nn

import os
import argparse

import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

from torchsummary import summary
from tqdm import tqdm

from models import VisionTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VisionTransformer")
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


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
    

    model = VisionTransformer(
        batch_size = args.batch_size,
        in_channels = args.in_channels,
        img_size = args.img_size,
        patch_size = args.patch_size,
        embedding_dim = args.embedding_dim,
        encoder_depth = args.encoder_depth,
        num_heads = args.num_heads,
        mlp_dim = args.mlp_dim,
        drop_rate = args.drop_rate,
        num_classes = args.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    step_size=100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=1e-5)

    # train()

    # test()