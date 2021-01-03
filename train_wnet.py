import logging
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.APD import ImageFolder
from losses.FocalLoss import FocalLoss
from models.WNet import WNet
from test import test
from transforms.transforms import PILToTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'{torch.cuda.device_count()} cuda devices available.')
logger.info(f'Using {device} device.')

batch_size = 16

trainset = ImageFolder('data/BSDS500512/train/', transform=PILToTensor())
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

net = WNet(3, 1)
net.use_seg_net = False
net.use_edge_net = True
net.requires_grad_for_layers(False, 'seg')
net.requires_grad_for_layers(True, 'edge')
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
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float) // 255

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
                          'data/BSDS500512/test/images/',
                          'data/BSDS500512/test/labels/',
                          f'data/BSDS500test/{epoch}/')

    writer.add_scalar('Precision/test', prec, epoch)
    writer.add_scalar('Recall/test', reca, epoch)
    writer.add_scalar('F1/test', F1, epoch)
    logger.info(f'Epoch {epoch}, precision {prec}, recall {reca}, F1 {F1}.')

    torch.save(net.module.state_dict(), Path(writer.log_dir).joinpath(f'WNet{epoch}.pth'))

    if F1 > best_F1:
        best_F1 = F1
        best_F1_epoch = epoch

    if epoch - best_F1_epoch > 10:
        break

logger.info(f'Best at epoch {best_F1_epoch}, F1 is {best_F1}.')
