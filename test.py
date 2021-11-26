import logging
import math
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from models.UNet import UNet
from utils.image import Tile
from utils.evaluation import calc_confusion_matrix
from utils.image import detile

logger = logging.getLogger(__name__)


def test(net, device, input_root, target_root, save_root=None):
    assert Path(input_root).exists(), f"Path '{input_root}' doesn't exist."
    assert Path(target_root).exists(), f"Path '{target_root}' doesn't exist."

    if save_root and not os.path.exists(save_root):
        os.makedirs(save_root)

    names = {Path(p).stem
             for p in glob(os.path.join(input_root, '*.bmp'))}

    assert not names ^ {Path(p).stem
                        for p in glob(os.path.join(target_root, '*.png'))}, \
        "Missing file for matching between input and target."

    net.eval()

    tile = Tile(512, 512)

    structure = np.ones((5, 5))
    prec_TP = prec_TPFP = reca_TP = reca_TPFN = 0

    tq = tqdm(names)
    for name in tq:
        with torch.no_grad():
            im = Image.open(os.path.join(input_root, f'{name}.bmp')).convert('RGB')

            width, height = im.size
            left = (math.ceil(width / 512) * 512 - width) // 2
            top = (math.ceil(height / 512) * 512 - height) // 2

            im = np.array(im)
            im = tile(im).transpose((0, 3, 1, 2))
            im = torch.from_numpy(im).to(device, dtype=torch.float)

            output = net(im)
            output = torch.sigmoid(output)
            pred = output[:, 0, ...] > 0.5
            pred = np.uint8(pred.cpu()) * 255
            pred = detile(pred, (width, height), top, left)

            if save_root:
                pred.save(os.path.join(save_root, f'{name}.png'))

            gt = Image.open(os.path.join(target_root, f'{name}.png'))
            pred = np.array(pred)
            gt = np.array(gt)

            p_TP, p_TPFP, r_TP, r_TPFN = calc_confusion_matrix(pred == 255, gt == 1, structure=structure)
            prec_TP += p_TP
            prec_TPFP += p_TPFP
            reca_TP += r_TP
            reca_TPFN += r_TPFN

    infinitesimal = 1e-10
    prec = prec_TP / (prec_TPFP + infinitesimal)
    reca = reca_TP / (reca_TPFN + infinitesimal)
    F1 = (2 * prec * reca) / (prec + reca + infinitesimal)

    return prec, reca, F1


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'{torch.cuda.device_count()} cuda device available.')

    net = UNet(3, 1)
    net.load_state_dict(torch.load('UNet0.pth'))
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    prec, reca, F1 = test(net,
                          device,
                          'data/04v2crack/val/images/',
                          'data/04v2crack/val/labels/')

    logger.info(f'Precision {prec}, recall {reca}, F1 {F1}.')
