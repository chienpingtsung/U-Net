from itertools import count

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.APD import APD202004v2crack
from losses.FocalLoss import FocalLoss
from models.UNet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

BATCH_size = 5

trainset = APD202004v2crack("/home/chienping/JupyterLab/datasets/04v2crack_tile_572_388/train/")
testset = APD202004v2crack("/home/chienping/JupyterLab/datasets/04v2crack_tile_572_388/val/")

trainloader = DataLoader(trainset, batch_size=BATCH_size, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_size, shuffle=False)

net = UNet(3, 2).to(device)

criterion = FocalLoss()
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

min_loss = 10000
min_loss_epoch = 0

for epoch in count():
    net.train()
    tq = tqdm(trainloader)
    for data in tq:
        inputs = data['image'].to(device, dtype=torch.float)
        labels = data['mask'].to(device, dtype=torch.long) // 255
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(dim=1))
        loss.backward()
        optimizer.step()

        tq.set_description("Training epoch {:3} loss is {}: ".format(epoch, loss.item()))

    net.eval()
    test_loss = 0.0
    test_times = 0
    tq = tqdm(testloader)
    for data in tq:
        with torch.no_grad():
            inputs = data['image'].to(device, dtype=torch.float)
            labels = data['mask'].to(device, dtype=torch.long) // 255

            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(dim=1))
            test_loss += loss.item()
            test_times += 1

            tq.set_description("Testing epoch {:3}: ".format(epoch))

    test_loss /= test_times
    print("Epoch {:3} test_loss: {}".format(epoch, test_loss))

    if test_loss < min_loss:
        min_loss = test_loss
        min_loss_epoch = epoch
        torch.save(net.state_dict(), "U-Net.weights")

    if epoch - min_loss_epoch > 10:
        break

    scheduler.step(test_loss)

print("Best at epoch {}, loss is {}.".format(min_loss_epoch, min_loss))