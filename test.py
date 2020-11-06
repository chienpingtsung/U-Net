import math
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.UNet import UNet
from toolbox.image import Tile, Detile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

names = glob("/home/chienping/JupyterLab/datasets/04v2crack/val/images/*.bmp")

net = UNet(3, 2).to(device)
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
        outputs = torch.softmax(outputs, dim=1)
        predicted = outputs[:, 1, ...] > 0.5
        predicted = np.uint8(predicted.cpu()) * 255
        print(predicted.max())

        ma = detile(predicted, (width, height), lef, top)
        ma.save("ds/pred/{}.png".format(os.path.splitext(os.path.basename(name))[0]))
