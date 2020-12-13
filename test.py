import math
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from models.UNet import UNet
from toolbox.image import Tile, Detile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Now using {torch.cuda.device_count()} {device} divces.")

names = glob("ds/04v2crack/val/images/*.bmp")

net = UNet(3, 1)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)
net.load_state_dict(torch.load("U-Net.weights"))
net.eval()

tile = Tile(572, 388)
detile = Detile()

for name in tqdm(names):
    with torch.no_grad():
        im = Image.open(name).convert('RGB')

        width, height = im.size
        lef = (math.ceil(width / 388) * 388 - width) // 2
        top = (math.ceil(height / 388) * 388 - height) // 2

        im = np.array(im)
        im = tile(im).transpose((0, 3, 1, 2))
        im = torch.from_numpy(im).to(device, dtype=torch.float)

        outputs = net(im)
        outputs = torch.sigmoid(outputs)
        predicted = outputs[:, 0, ...] > 0.5
        predicted = np.uint8(predicted.cpu()) * 255

        ma = detile(predicted, (width, height), lef, top)
        ma.save("ds/pred/{}.png".format(os.path.splitext(os.path.basename(name))[0]))
