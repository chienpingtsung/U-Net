import logging
from itertools import count

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.APD import ImageFolder
from losses.FocalLoss import FocalLoss
from models.UNet import UNet
from test import test

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'{torch.cuda.device_count()} cuda devices available.')
logger.info(f'Using {device} device.')

batch_size = 16

trainset = ImageFolder('data/04v2crack512/train/')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

net = UNet(3, 1)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

criterion = FocalLoss()
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

best_F1 = 0
best_F1_epoch = 0
writer = SummaryWriter()

for epoch in count():
    net.train()
    train_loss = 0
    train_time = 0
    tq = tqdm(trainloader)
    for data in tq:
        images, labels = data
        images.to(device, dtype=torch.float)
        labels.to(device, dtype=torch.float)

        output = net(images)
        loss = criterion(output, labels)
        train_loss += loss.item()
        train_time += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch)
        tq.set_description(f'Training epoch {epoch:3}, loss {loss.item()}')

    scheduler.step(train_loss / train_time)

    prec, reca, F1 = test(net, device,
                          'data/04v2crack/val/images/',
                          'data/04v2crack/val/labels/',
                          f'data/test/{epoch}/')

    writer.add_scalar('Precision/test', prec, epoch)
    writer.add_scalar('Recall/test', reca, epoch)
    writer.add_scalar('F1/test', F1, epoch)
    logger.info(f'Epoch {epoch}, precision {prec}, recall {reca}, F1 {F1}.')

    torch.save(net.module.state_dict(), f'UNet{epoch}.pth')

    if F1 > best_F1:
        best_F1 = F1
        best_F1_epoch = epoch

    if epoch - best_F1_epoch > 10:
        break

logger.info(f'Best at epoch {best_F1_epoch}, F1 is {best_F1}.')
