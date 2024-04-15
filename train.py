import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse

import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

from vit import ViT
from tqdm import tqdm

# train()
def train():
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

        # if epoch % 10 == 0:

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
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    step_size=100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=1e-5)

    # train()

    # test()