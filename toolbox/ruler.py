import os
from glob import glob

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


class PrecisionRecallF1(object):
    """Evaluate model with Precision, Recall and F1."""

    def __init__(self, tolerance=0):
        super(PrecisionRecallF1, self).__init__()
        self.tolerance = tolerance

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

            GT_dilate = GT.filter(ImageFilter.MaxFilter(1 + 2 * self.tolerance))
            PRED_dilate = PRED.filter(ImageFilter.MaxFilter(1 + 2 * self.tolerance))

            GT_mode = GT.mode
            PRED_mode = PRED.mode

            GT = np.array(GT)
            PRED = np.array(PRED)
            GT_dilate = np.array(GT_dilate)
            PRED_dilate = np.array(PRED_dilate)

            if GT_mode == 'L':
                assert np.logical_or(GT == 0, GT == 255).all(), "Only 0 and 255 are valid values for binary images."
                GT = GT // 255
                GT_dilate = GT_dilate // 255
            if PRED_mode == 'L':
                assert np.logical_or(PRED == 0, PRED == 255).all(), "Only 0 and 255 are valid values for binary images."
                PRED = PRED // 255
                PRED_dilate = PRED_dilate // 255

            prec_TP += ((PRED == 1) & (GT_dilate == 1)).sum()
            prec_TPFP += (PRED == 1).sum()
            reca_TP += ((GT == 1) & (PRED_dilate == 1)).sum()
            reca_TPFN += (GT == 1).sum()

        infinitesimal = 1.e-9
        prec = prec_TP / (prec_TPFP + infinitesimal)
        reca = reca_TP / (reca_TPFN + infinitesimal)
        F1 = (2 * prec * reca) / (prec + reca + infinitesimal)

        return prec, reca, F1


if __name__ == '__main__':
    pass