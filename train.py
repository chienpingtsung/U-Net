import argparse
import shutil
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import tqdm

from lib.datasets import ImageMaskFolder
from lib.losses import FocalLoss
from lib.models import UNet
from lib.transforms import ComposeTogether, RandomRotationTogether, RandomCropTogether, RandomHorizontalFlipTogether, \
    RandomVerticalFlipTogether, Dilation, ToTensorTogether
from test import test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--train_set')
    parser.add_argument('--test_set')
    parser.add_argument('--log_dir')
    parser.add_argument('--weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{torch.cuda.device_count()} cuda device available.')
    print(f'Using {device} device.')

    batch_size = args.batch_size
    if torch.cuda.device_count() > 1:
        batch_size *= torch.cuda.device_count()

    writer = SummaryWriter(args.log_dir)
    writer_log_dir = Path(writer.log_dir)

    trainset = ImageMaskFolder(args.train_set,
                          transforms=ComposeTogether([RandomRotationTogether(180, expand=True),
                                                      RandomCropTogether(512),
                                                      RandomHorizontalFlipTogether(),
                                                      RandomVerticalFlipTogether()]),
                          transform=ComposeTogether([ToTensor()]),
                          target_transform=ComposeTogether([Dilation(5),
                                                            ToTensor()]))
    testset = ImageMaskFolder(args.test_set,
                         transforms=ComposeTogether([ToTensorTogether()]))
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=batch_size,
                             pin_memory=True,
                             drop_last=True)
    testloader = DataLoader(testset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)

    net = UNet(3, 1)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(net.parameters())
    best_f1 = 0
    best_f1_epoch = 0
    epoch = 0

    if args.weights:
        state = torch.load(args.weights)
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        best_f1 = state['best_f1']
        best_f1_epoch = state['best_f1_epoch']
        epoch = state['epoch']

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    for epoch in count(epoch):
        # Train here.
        net.train()

        total_loss = 0
        propagation_counter = 0

        tq = tqdm(trainloader)
        for image, label, _ in tq:
            image = image.to(device)
            label = label.to(device)

            output = net(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            propagation_counter += 1

            tq.set_description(f'Training epoch {epoch}, loss {loss.item()}')
            writer.add_scalar('train/loss', loss.item(), epoch)

        writer.add_scalar('train/mean_loss', total_loss / propagation_counter, epoch)

        prec, reca, f1 = test(net, testloader, device,
                              save_to=writer_log_dir.joinpath(f'test_{epoch}/'))

        writer.add_scalar('test/Precision', prec, epoch)
        writer.add_scalar('test/Recall', reca, epoch)
        writer.add_scalar('test/F1', f1, epoch)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch

        if isinstance(net, DataParallel):
            net_state = net.module.state_dict()
        else:
            net_state = net.state_dict()
        torch.save({'net': net_state,
                    'optimizer': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'best_f1_epoch': best_f1_epoch,
                    'epoch': epoch},
                   writer_log_dir.joinpath('last.pth'))
        if best_f1_epoch == epoch:
            shutil.copy(writer_log_dir.joinpath('last.pth'), writer_log_dir.joinpath('best.pth'))

        if epoch - state['epoch'] > 20 and epoch - best_f1_epoch > 20:
            writer.add_text('U-Net', f'Best checkpoint is at epoch {best_f1_epoch}, and F1 is {best_f1}.')
            break
