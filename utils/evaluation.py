from scipy import ndimage


def calc_confusion_matrix(pred, gt, structure=None):
    """

    :param pred: Predicted mask.
    :param gt: Groundtruth mask.
    :param structure: Structure for binary_dilation, None for not using tolerance.
    :return: (precision_TP, precision_TPFP, recall_TP, recall_TPFN)
    """
    dilated_pred = pred
    dilated_gt = gt

    if structure:
        dilated_pred = ndimage.binary_dilation(pred, structure=structure)
        dilated_gt = ndimage.binary_dilation(gt, structure=structure)

    prec_TP = (pred & dilated_gt).sum()
    prec_TPFP = pred.sum()
    reca_TP = (gt & dilated_pred).sum()
    reca_TPFN = gt.sum()

    return prec_TP, prec_TPFP, reca_TP, reca_TPFN
