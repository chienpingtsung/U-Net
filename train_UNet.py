import logging
from itertools import count

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.APD import APD202004v2crack
from losses.FocalLoss import FocalLoss
from models.UNet import UNetM
from test import test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Now using {torch.cuda.device_count()} {device} divces.")

batch_size = 16

trainset = APD202004v2crack("ds/04v2crack_edge_tile_572_484/train/")

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

net = UNetM(3, 1)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

criterion = FocalLoss()
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

bestF1 = 1000
bestF1_epoch = 0
write = SummaryWriter()

for epoch in count():
    net.train()
    train_loss = 0.0
    train_times = 0
    tq = tqdm(trainloader)
    for data in tq:
        inputs = data['image'].to(device, dtype=torch.float)
        labels = data['mask'].to(device, dtype=torch.float) // 255
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(dim=1))
        train_loss += loss.item()
        train_times += 1

        loss.backward()
        optimizer.step()

        write.add_scalar("Loss/train", loss.item(), epoch)
        tq.set_description("Training epoch {:3} loss is {}".format(epoch, loss.item()))

    scheduler.step(train_loss / train_times)

    prec, reca, F1 = test(net, device,
                          "ds/04v2crack/val/images/",
                          "ds/04v2crack/val/labels/")

    logger.info(f"Precision {prec}, recall {reca}, F1 {F1}.")

    torch.save(net.module.state_dict(), f'UNetM{epoch}.pth')

    if F1 > bestF1:
        bestF1 = F1
        bestF1_epoch = epoch

    if epoch - bestF1_epoch > 10:
        break

logger.info("Best at epoch {}, F1 is {}.".format(bestF1_epoch, bestF1))
