import os
from glob import glob

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def calc_confusion_matrix(pred, gt, structure):
    """

    :param pred: Predicted mask.
    :param gt: Groundtruth mask.
    :param structure: Structure for binary_dilation, None for not using tolerance.
    :return: (precision_TP, precision_TPFP, recall_TP, recall_TPFN)
    """
    dilated_pred = ndimage.binary_dilation(pred, structure=structure)
    dilated_gt = ndimage.binary_dilation(gt, structure=structure)

    prec_TP = (pred & dilated_gt).sum()
    prec_TPFP = pred.sum()
    reca_TP = (gt & dilated_pred).sum()
    reca_TPFN = gt.sum()

    return prec_TP, prec_TPFP, reca_TP, reca_TPFN


class PrecisionRecallF1(object):
    """Evaluate model with Precision, Recall and F1."""

    def __init__(self, tolerance=0):
        super(PrecisionRecallF1, self).__init__()
        self.structure = np.ones((1 + 2 * tolerance, 1 + 2 * tolerance))

    def __call__(self, GT_path, PRED_path):
        assert os.path.exists(GT_path) and os.path.exists(PRED_path), "Path doesn't exist."

        GT_names = {os.path.basename(p) for p in glob(os.path.join(GT_path, '*.png'))}
        PRED_names = {os.path.basename(p) for p in glob(os.path.join(PRED_path, '*.png'))}
        assert not GT_names ^ PRED_names, "Missing file for matching between GT and PRED."

        prec_TP = 0
        prec_TPFP = 0
        reca_TP = 0
        reca_TPFN = 0
        names = GT_names & PRED_names
        for n in tqdm(names):
            GT = Image.open(os.path.join(GT_path, n))
            PRED = Image.open(os.path.join(PRED_path, n))
            assert GT.mode in ['1', 'L'], "Unsupported mode {} from {}.".format(GT.mode, n)
            assert PRED.mode in ['1', 'L'], "Unsupported mode {} from {}.".format(PRED.mode, n)

            GT_mode = GT.mode
            PRED_mode = PRED.mode

            GT = np.array(GT)
            PRED = np.array(PRED)

            if GT_mode == 'L':
                assert np.logical_or(GT == 0, GT == 255).all(), "Only 0 and 255 are valid values for binary images."
                GT = GT // 255
            if PRED_mode == 'L':
                assert np.logical_or(PRED == 0, PRED == 255).all(), "Only 0 and 255 are valid values for binary images."
                PRED = PRED // 255

            p_TP, p_TPFP, r_TP, r_TPFN = calc_confusion_matrix(PRED == 1, GT == 1, structure=self.structure)

            prec_TP += p_TP
            prec_TPFP += p_TPFP
            reca_TP += r_TP
            reca_TPFN += r_TPFN

        infinitesimal = 1.e-9
        prec = prec_TP / (prec_TPFP + infinitesimal)
        reca = reca_TP / (reca_TPFN + infinitesimal)
        F1 = (2 * prec * reca) / (prec + reca + infinitesimal)

        return prec, reca, F1
