from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lib.evaluate import calc_confusion_matrix


def test(net, dataloader, device, threshold=0.5, save_to: Path = None):
    if save_to:
        save_to.mkdir(parents=True, exist_ok=True)

    net.eval()

    prec_TP = prec_TPFP = reca_TP = reca_TPFN = 0

    tq = tqdm(dataloader)
    for image, label, stem in tq:
        with torch.no_grad():
            image = image.to(device)

            output = net(image)
            output = torch.sigmoid(output)
            output = torch.squeeze(output) > threshold

            label = torch.squeeze(label).numpy().astype(bool)
            pred = output.cpu().numpy().astype(bool)
            p_TP, p_TPFP, r_TP, r_TPFN = calc_confusion_matrix(label, pred, tolerance=2)
            prec_TP += p_TP
            prec_TPFP += p_TPFP
            reca_TP += r_TP
            reca_TPFN += r_TPFN

            if save_to:
                Image.fromarray(pred.astype(np.uint8), 'L').convert('1').save(save_to.joinpath(f'{stem[0]}.png'))

    infinitesimal = 1e-10
    prec = prec_TP / (prec_TPFP + infinitesimal)
    reca = reca_TP / (reca_TPFN + infinitesimal)
    f1 = (2 * prec * reca) / (prec + reca + infinitesimal)

    return prec, reca, f1
