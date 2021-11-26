import logging
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.APD import WNetDataset
from losses.FocalLoss import FocalLoss
from models.WNet import WNet, WNetSIM
from test import test
from transforms.transforms import PILToTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'{torch.cuda.device_count()} cuda devices available.')
logger.info(f'Using {device} device.')

batch_size = 3 * torch.cuda.device_count()

trainset = WNetDataset('../JupyterLab/datasets/07v2crack_512/train/images/',
                       '../JupyterLab/datasets/07v2crack_512/train/labels/',
                       '../JupyterLab/datasets/07v2crack_edge_512/train/labels/',
                       transform=PILToTensor())
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

net = WNetSIM(3, 1)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

criterion = FocalLoss()
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)

best_F1 = 0
best_F1_epoch = 0
writer = SummaryWriter()

for epoch in count():
    net.train()
    tq = tqdm(trainloader)
    for data in tq:
        images, seg_labels, edge_labels = data
        images = images.to(device, dtype=torch.float)
        seg_labels = seg_labels.to(device, dtype=torch.float) // 255
        edge_labels = edge_labels.to(device, dtype=torch.float) // 255

        seg_output, edge_output = net(images)
        seg_loss = criterion(seg_output, seg_labels)
        edge_loss = criterion(edge_output, edge_labels)

        optimizer.zero_grad()
        (seg_loss + edge_loss).backward()
        optimizer.step()

        writer.add_scalar('Loss/seg', seg_loss.item(), epoch)
        writer.add_scalar('Loss/edge', edge_loss.item(), epoch)
        tq.set_description(f'Training epoch {epoch:3}, seg_loss {seg_loss.item()}, edge_loss {edge_loss.item()}')

    prec, reca, F1 = test(net, device,
                          '../JupyterLab/datasets/07v2crack/val/images/',
                          '../JupyterLab/datasets/07v2crack/val/labels/',
                          Path(writer.log_dir).joinpath(f'test/{epoch}/'))

    writer.add_scalar('Precision/test', prec, epoch)
    writer.add_scalar('Recall/test', reca, epoch)
    writer.add_scalar('F1/test', F1, epoch)
    logger.info(f'Epoch {epoch}, precision {prec}, recall {reca}, F1 {F1}.')

    torch.save(net.module.state_dict(), Path(writer.log_dir).joinpath(f'WNetSIM{epoch}.pth'))

    if F1 > best_F1:
        best_F1 = F1
        best_F1_epoch = epoch

    if epoch - best_F1_epoch > 10:
        break

    scheduler.step(F1)

logger.info(f'Best at epoch {best_F1_epoch}, F1 is {best_F1}.')
